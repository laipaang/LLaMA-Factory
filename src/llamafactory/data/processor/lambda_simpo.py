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
class LambdaSimpoDatasetProcessor(DatasetProcessor):
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
        sample_length = []

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
            sample_length.append(len(input_ids))

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
        model_inputs["sample_length"].append(sample_length)

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
        # 首先检查并打印 input_ids 的结构
        print("input_ids type:", type(example["input_ids"]))
        if isinstance(example["input_ids"], list):
            print("input_ids length:", len(example["input_ids"]))
            if len(example["input_ids"]) > 0:
                print("first element type:", type(example["input_ids"][0]))

        # 解码所有 input_ids 列表中的文本
        decoded_texts = []
        if isinstance(example["input_ids"], list):
            for i, input_ids_list in enumerate(example["input_ids"]):
                if isinstance(input_ids_list, list):
                    try:
                        decoded_text = self.tokenizer.decode(input_ids_list, skip_special_tokens=False)
                        decoded_texts.append(f"文本 {i}: {decoded_text}")
                    except Exception as e:
                        decoded_texts.append(f"文本 {i}: 解码错误 - {e}")
                else:
                    decoded_texts.append(f"元素 {i} 不是列表: {input_ids_list}")

        print("解码后的文本列表:")
        for text in decoded_texts:
            print(text)

        # 打印其他字段
        print("labels: {}\n".format(example["labels"]))
        print("is_use_sft_loss: {}\n".format(example["is_use_sft_loss"]))
        print("cls_soft_label: {}\n".format(example["cls_soft_label"]))
        print("is_use_cls_loss: {}\n".format(example["is_use_cls_loss"]))
        print("tw_soft_label: {}\n".format(example["tw_soft_label"]))
        print("is_use_tw_loss: {}\n".format(example["is_use_tw_loss"]))
        print("scores: {}\n".format(example["scores"]))
        print("length: {}\n".format(example["sample_length"]))
        print("rank: {}\n".format(example["rank"]))

