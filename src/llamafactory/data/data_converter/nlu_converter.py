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

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ...extras import logging
from ..data_utils import Role
from .base_converter import DatasetConverter

if TYPE_CHECKING:
    from ..datasets import Dataset, IterableDataset
    from ..transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)

@dataclass
class NluHeadDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        src = []
        if self.dataset_attr.src and example[self.dataset_attr.src]:
            src.append(example[self.dataset_attr.src])

        tgt = []
        is_use_sft_loss, is_use_cls_loss, is_use_tw_loss, cls_softlabel, tw_softlabel = [], [], [], [], []
        if self.dataset_attr.tgt and example[self.dataset_attr.tgt]:
            tgt_reward = example[self.dataset_attr.tgt]
            tgt_reward_str_list = tgt_reward.split("[SEP]")
            tgt.append(tgt_reward_str_list[0])

            score_list = tgt_reward_str_list[1].split(" ")
            is_sft = float(tgt_reward_str_list[2])
            tag = int(tgt_reward_str_list[3])
            is_use_sft_loss.append(is_sft)
            qlq_list = [float(i) for i in score_list[0].split('_')]
            is_dishang = float(score_list[1])
            score_list.append(float(score_list[-1]))
            cls_softlabel.append(qlq_list)
            tw_softlabel.append([1.0 - is_dishang, is_dishang])
            if tag == 0:
                is_use_cls_loss.append(0.0)
                is_use_tw_loss.append(0.0)
            elif tag == 1:
                is_use_cls_loss.append(1.0)
                is_use_tw_loss.append(0.0)
            elif tag == 2:
                is_use_cls_loss.append(0.0)
                is_use_tw_loss.append(1.0)
            elif tag == 3:
                is_use_cls_loss.append(1.0)
                is_use_tw_loss.append(1.0)
            tgt.append(example[self.dataset_attr.tgt])

        output = {
            "_src": src,
            "_tgt": tgt,
            "_is_use_sft_loss": is_use_sft_loss,
            "_cls_soft_label": cls_softlabel,
            "_is_use_cls_loss": is_use_cls_loss,
            "_tw_soft_label": tw_softlabel,
            "_is_use_tw_loss": is_use_tw_loss,
        }
        return output

