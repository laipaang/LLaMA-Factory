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
class NluHeadDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_src"])):
            src_msg = self.template.format_user.apply(content=examples["_src"][i][0])
            tgt_msg = self.template.format_assistant.apply(content=examples["_tgt"][i][0])
            source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
            target_ids = self.tokenizer.encode(tgt_msg[0], add_special_tokens=False)
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
class PackedNluHeadDatasetProcessor(NluHeadDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0 
        batch_input_ids, batch_labels, batch_is_use_sft_loss, batch_cls_soft_label, batch_is_use_cls_loss, batch_tw_soft_label, batch_is_use_tw_loss = [], [], [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_src"])):
            src_msg = self.template.format_user.apply(content=examples["_src"][i][0])
            tgt_msg = self.template.format_assistant.apply(content=examples["_tgt"][i][0])
            source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
            target_ids = self.tokenizer.encode(tgt_msg[0], add_special_tokens=False)
            #padding
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            source_label = [IGNORE_INDEX] * source_len
            target_label = target_ids
            input_ids = source_ids + target_ids
            labels = source_label + target_label
            if self.template.efficient_eos:
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_tok]

            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_is_use_sft_loss.append(examples["_is_use_sft_loss"][i])
                batch_cls_soft_label.append(examples["_cls_soft_label"][i][0])
                batch_is_use_cls_loss.append(examples["_is_use_cls_loss"][i])
                batch_tw_soft_label.append(examples["_tw_soft_label"][i][0])
                batch_is_use_tw_loss.append(examples["_is_use_tw_loss"][i])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_is_use_sft_loss, packed_cls_soft_label, packed_is_use_cls_loss, packed_tw_soft_label, packed_is_use_tw_loss = [], [], [], [], []
            packed_sample_last_indices = []
            total_sample_len = 0 
    
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_sample_last_indices.append(total_sample_len + len(batch_input_ids[index]) - 1)
                total_sample_len += len(batch_input_ids[index])

                packed_is_use_sft_loss += batch_is_use_sft_loss[index] * len(batch_input_ids[index])
                packed_cls_soft_label.append(batch_cls_soft_label[index])
                #packed_cls_soft_label += batch_cls_soft_label[index]
                packed_is_use_cls_loss += batch_is_use_cls_loss[index]
                packed_tw_soft_label.append(batch_tw_soft_label[index])
                #packed_tw_soft_label += batch_tw_soft_label[index]
                packed_is_use_tw_loss += batch_is_use_tw_loss[index]

                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            #print(f"before pad, input len: {len(packed_input_ids)}, position ids: {len(packed_position_ids)}")
            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn
            #print(f"after pad, input len: {len(packed_input_ids)}, position ids: {len(packed_position_ids)}")

            if len(packed_sample_last_indices) < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - len(packed_sample_last_indices) + 1
                packed_sample_last_indices += [0] * pad_length
                #print(f"in model input, packed last indices: {packed_sample_last_indices}")

            if len(packed_is_use_sft_loss) < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - len(packed_is_use_sft_loss) + 1
                packed_is_use_sft_loss += [0.0] * pad_length

            if len(packed_cls_soft_label) < (self.data_args.cutoff_len + 1) * 4:
                pad_length = self.data_args.cutoff_len - len(packed_cls_soft_label) + 1
                packed_cls_soft_label += [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]] * pad_length

            if len(packed_is_use_cls_loss) < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - len(packed_is_use_cls_loss) + 1
                packed_is_use_cls_loss += [0.0] * pad_length

            if len(packed_tw_soft_label) < (self.data_args.cutoff_len + 1) * 3:
                pad_length = self.data_args.cutoff_len - len(packed_tw_soft_label) + 1
                packed_tw_soft_label += [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]] * pad_length

            if len(packed_is_use_tw_loss) < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - len(packed_is_use_tw_loss) + 1
                packed_is_use_tw_loss += [0.0] * pad_length

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")
            
            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["is_use_sft_loss"].append(packed_is_use_sft_loss)
            model_inputs["cls_soft_label"].append(packed_cls_soft_label)
            model_inputs["is_use_cls_loss"].append(packed_is_use_cls_loss)
            model_inputs["tw_soft_label"].append(packed_tw_soft_label)
            model_inputs["is_use_tw_loss"].append(packed_is_use_tw_loss)
            model_inputs["sample_last_indices"].append(packed_sample_last_indices)

        return model_inputs