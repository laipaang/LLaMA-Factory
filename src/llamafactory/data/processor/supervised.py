# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        assert len(messages) == len(encoded_pairs) * 2
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                if messages[turn_idx * 2 + 1].get("weight", 1.0) > 0.0:
                    target_label = target_ids
                else:
                    target_label = [IGNORE_INDEX] * target_len

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs


@dataclass
class TargetingDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_src"])):
            source_ids = self.tokenizer.encode(examples["_src"][i][0], add_special_tokens=False)
            target_ids = self.tokenizer.encode(examples["_tgt"][i][0], add_special_tokens=False)
            #padding
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            source_label = [IGNORE_INDEX] * source_len
            target_label = target_ids
            input_ids = source_ids + target_ids
            label_ids = source_label + target_label
            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                label_ids += [self.tokenizer.eos_tok]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(label_ids)
            model_inputs["is_use_sft_loss"].append(examples["_is_use_sft_loss"][i])
            model_inputs["cls_soft_label"].append(examples["_cls_soft_label"][i][0])
            model_inputs["is_use_cls_loss"].append(examples["_is_use_cls_loss"][i])
            model_inputs["tw_soft_label"].append(examples["_tw_soft_label"][i][0])
            model_inputs["is_use_tw_loss"].append(examples["_is_use_tw_loss"][i])
	    model_inputs["cls_mask"].append(len(input_ids)-1)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:{}\n".format(example["input_ids"]))
        print("inputs:{}\n".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:{}\n".format(example["labels"]))
        print(f"labels:{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
        print("is_use_sft_loss:{}\n".format(example["is_use_sft_loss"]))
        print("cls_soft_label:{}\n".format(example["cls_soft_label"]))
        print("is_use_cls_loss:{}\n".format(example["is_use_cls_loss"]))
        print("tw_soft_label:{}\n".format(example["tw_soft_label"]))
        print("is_use_tw_loss:{}\n".format(example["is_use_tw_loss"]))


@dataclass
class RlhfDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.

        # 考虑到 反馈 的特殊性, 这个只支持 preprocessing_batch_size = 1
        # huggingface 的 datasets 库处理 Map 函数，会将返回的字典中的列表拆开成独立样本
        model_inputs = defaultdict(list)

        src_list = examples["_src"][0]
        tgt_list = examples["_tgt"][0]
        use_sft_loss_list = examples["_is_use_sft_loss"][0]
        use_cls_loss_list = examples["_is_use_cls_loss"][0]
        use_tw_loss_list = examples["_is_use_tw_loss"][0]
        cls_soft_label_list = examples["_cls_soft_label"][0]
        tw_soft_label_list = examples["_tw_soft_label"][0]
        scores_list = examples["_scores"][0]
        rank_list = examples["_rank"][0]

        num_pairs = len(src_list)

        input_ids_batch, labels_batch, attention_mask_batch = [], [], []
        use_sft_loss_batch, use_cls_loss_batch, use_tw_loss_batch = [], [], []
        cls_soft_label_batch, tw_soft_label_batch, scores_batch, rank_batch = [], [], [], []

        for pair_idx in range(num_pairs):
            # 处理单个 src-tgt 对
            src = src_list[pair_idx]
            tgt = tgt_list[pair_idx]
            
            src_msg = self.template.format_user.apply(content=src)
            tgt_msg = self.template.format_assistant.apply(content=tgt)
            source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
            target_ids = self.tokenizer.encode(tgt_msg[0], add_special_tokens=False)

            # 截断到最大长度
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]

            # 构建输入和标签
            input_ids = source_ids + target_ids
            label_ids = [IGNORE_INDEX] * len(source_ids) + target_ids
            if self.template.efficient_eos:
                input_ids.append(self.tokenizer.eos_token_id)
                label_ids.append(self.tokenizer.eos_token_id)
            
            input_ids_batch.append(input_ids)
            labels_batch.append(label_ids)
            attention_mask_batch.append([1] * len(input_ids))
            use_sft_loss_batch.append(use_sft_loss_list[pair_idx])
            use_cls_loss_batch.append(use_cls_loss_list[pair_idx])
            use_tw_loss_batch.append(use_tw_loss_list[pair_idx])
            cls_soft_label_batch.append(cls_soft_label_list[pair_idx])
            tw_soft_label_batch.append(tw_soft_label_list[pair_idx])
            scores_batch.append(scores_list[pair_idx])
            rank_batch.append(rank_list[pair_idx])

        # 添加到输出
        model_inputs["input_ids"].append(input_ids_batch)
        model_inputs["attention_mask"].append(attention_mask_batch)
        model_inputs["labels"].append(labels_batch)

        # 处理其他字段（例如取当前 pair_idx 的值）
        model_inputs["is_use_sft_loss"].append(use_sft_loss_batch)
        model_inputs["is_use_cls_loss"].append(use_cls_loss_batch)
        model_inputs["is_use_tw_loss"].append(use_tw_loss_batch)
        model_inputs["cls_soft_label"].append(cls_soft_label_batch)
        model_inputs["tw_soft_label"].append(tw_soft_label_batch)
        model_inputs["scores"].append(scores_batch)
        model_inputs["rank"].append(rank_batch) 

        # print("model_inputs: ", model_inputs)
        # model_inputs 确保了是一条原始样本的
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        example 形如
        {
            'input_ids': [
                [seq_1 token_id], [seq_2 token_id]
            ], 
            'attention_mask': [
                [mask_1], [mask_2]
            ], 
            'labels': [
                [label_1 token_id], [label_2 token_id]
            ], 
            'is_use_sft_loss': [seq_1 sft tag, seq_2 sft tag], 
            'is_use_cls_loss': 类似, 
            'is_use_tw_loss': 类似, 
            'cls_soft_label': [
                [qlq_score_1], [qlq_score_2]
            ], 
            'tw_soft_label': [
                [tw_score_1], [tw_score_2]
            ], 
            'scores': [click_1, click_2], 
            'rank': [rank_1, rank_2]
        }
        """
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:{}\n".format(example["input_ids"]))
        print("inputs:{}\n".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:{}\n".format(example["labels"]))
        print(f"labels:{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")
        print("is_use_sft_loss:{}\n".format(example["is_use_sft_loss"]))
        print("cls_soft_label:{}\n".format(example["cls_soft_label"]))
        print("is_use_cls_loss:{}\n".format(example["is_use_cls_loss"]))
        print("tw_soft_label:{}\n".format(example["tw_soft_label"]))
        print("is_use_tw_loss:{}\n".format(example["is_use_tw_loss"]))
        print("scores:{}\n".format(example["scores"]))
        print("rank:{}\n".format(example["rank"]))

