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
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else None

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )
            if self.dataset_attr.weight_tag and self.dataset_attr.weight_tag in message:
                aligned_messages[-1]["weight"] = message[self.dataset_attr.weight_tag]

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output



