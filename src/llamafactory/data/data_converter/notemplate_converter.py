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


if TYPE_CHECKING:
    from ..datasets import Dataset, IterableDataset
    from ..transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)

@dataclass
class NoTemplateDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        src = []
        if self.dataset_attr.src and example[self.dataset_attr.src]:
            src.append(example[self.dataset_attr.src])

        tgt = []
        if self.dataset_attr.tgt and example[self.dataset_attr.tgt]:
            tgt.append(example[self.dataset_attr.tgt])

        output = {
            "_src": src,
            "_tgt": tgt,
        }
        return output