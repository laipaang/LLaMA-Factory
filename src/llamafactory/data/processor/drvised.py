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
class DrDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_src"])):
            #Query for DR
            src = examples["_src"][i][0]
            src = "QUERYFORDR" + src
            src_msg = self.template.format_user.apply(content=src)
            source_ids = self.tokenizer.encode(src_msg[0], add_special_tokens=False)
            tgt_text = examples["_tgt"][i][0]
            tgt_text_ls = tgt_text.split("[SEP]")
            tokens_target = []
            for i, x in enumerate(tgt_text_ls):
                tokens_target.extend(x)
                if i != len(tgt_text_ls)-1:
                    tokens_target.append("[SEP]")
            sep_idx = tokens_target.index("[SEP]")
            tokens_target_forward = tokens_target[:sep_idx]
            tokens_target_forward = ''.join(tokens_target_forward)
            tokens_target_forward_msg = self.template.format_assistant.apply(content=tokens_target_forward)
            target_ids = self.tokenizer.encode(tokens_target_forward_msg[0], add_special_tokens=False)
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            source_label = [IGNORE_INDEX] * source_len
            target_label = target_ids
            input_id = source_ids + target_ids
            label_id = source_label + target_label
            if self.template.efficient_eos:
                input_id += [self.tokenizer.eos_token_id]
                label_id += [self.tokenizer.eos_tok]
            model_inputs["input_ids"].append(input_id)
            model_inputs["attention_mask"].append([1] * len(input_id))
            model_inputs["labels"].append(label_id)
            
            #agent for dr
            tokens_src_backward = tokens_target[sep_idx + 1:]
            tokens_src_backward = ''.join(tokens_src_backward)
            tokens_src_backward = "AGENTFORDR" + tokens_src_backward
            tokens_src_backward_msg = self.template.format_user.apply(content=tokens_src_backward)
            source_ids = self.tokenizer.encode(tokens_src_backward_msg[0], add_special_tokens=False)
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.data_args.cutoff_len)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            source_label = [IGNORE_INDEX] * source_len
            target_label = target_ids
            input_id1 = source_ids + target_ids
            label_id1 = source_label + target_label
            if self.template.efficient_eos:
                input_id1 += [self.tokenizer.eos_token_id]
                label_id1 += [self.tokenizer.eos_tok]
            model_inputs["input_ids"].append(input_id1)
            model_inputs["attention_mask"].append([1] * len(input_id1))
            model_inputs["labels"].append(label_id1)
            
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:{}\n".format(example["input_ids"]))
        print("inputs:{}\n".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:{}\n".format(example["labels"]))
        print(f"labels:{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}\n")


