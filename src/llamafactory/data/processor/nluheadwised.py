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


