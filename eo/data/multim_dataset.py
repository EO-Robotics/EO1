# Copyright 2025 EO-Robotics Team. All rights reserved.
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

import copy
import math
import random
import re

import torch
import ujson as json
from lerobot.constants import ACTION, OBS_STATE
from torch.utils.data import Dataset

from eo.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_ACTION_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_STATE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    LLAVA_VLA_TOKEN,
    PASS_ACTION_TOKEN,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    TASK_VLA_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from eo.data.lerobot_dataset import MultiLeRobotDataset
from eo.data.schema import MMDatasetConfig


class MultimodaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_configs: list[MMDatasetConfig],
        max_seq_length: int = 16384,
        meta_dataset: MultiLeRobotDataset = None,
        max_action_dim: int = 32,
        chunk_size: int = 50,
        sample_actions: bool = True,
    ):
        super().__init__()
        self.data_configs = data_configs
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size

        list_data_dict, dataset_lens = [], []
        for i, dataset in enumerate(data_configs):
            json_path = dataset.json_path
            sampling_strategy = dataset.sampling_strategy
            sampling_number = None

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                for line in open(json_path):
                    cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                cur_data_dict = json.load(open(json_path))
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            # NOTE: filter out lines above MAX_SEQ_LENGTH
            cur_data_dict = [line for line in cur_data_dict if line.get("seq_length", 0) <= max_seq_length]

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(
                        float(sampling_number.split("%")[0]) * len(cur_data_dict) / 100
                    )
                else:
                    sampling_number = int(sampling_number)

            # sampling
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            dataset_lens.append(len(cur_data_dict))
            for data in cur_data_dict:
                data["vision_base_idx"] = i
            list_data_dict.extend(cur_data_dict)

        self.data = list_data_dict
        self.dataset_lens = dataset_lens
        self.__set_metadata(meta_dataset)

        # set to false during calculating lengths
        self.sample_actions = sample_actions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        sources = self.data[i]

        if "lerobot" in sources:
            items = []
            for le in sources["lerobot"]:
                repo_id, idx, chunk_size = le.split(" ")
                items += [self.__get_metadata(repo_id, int(idx))]
            sources = build_interleaved_prompt(
                items,
                sources,
                max_action_dim=self.max_action_dim,
                chunk_size=self.chunk_size,
                sample_actions=self.sample_actions,
            )
            transformed_source = sources
        else:
            transformed_source = copy.deepcopy(sources)
        transformed_source["conversations"] = llava_to_openai(
            transformed_source["conversations"], "video" in sources
        )
        return transformed_source

    def __get_metadata(self, repo_id, idx):
        """Get the metadata from lerobot dataset."""
        # raise NotImplementedError("__get_metadata is not implemented")
        return self.meta_dataset.getitem_by_id(repo_id, idx)

    def __set_metadata(self, meta_dataset: MultiLeRobotDataset):
        """Set the metadata for the sources."""
        self.meta_dataset = meta_dataset

    @property
    def vision_base_paths(self):
        return [dataset.vision_base_path for dataset in self.data_configs]


def build_interleaved_prompt(
    items: list[dict],
    sources: dict = None,
    max_action_dim: int = 32,
    chunk_size: int = 50,
    sample_actions: bool = False,
):
    """construct a openai style multimodal conversation for lerobot item or mm item containing <action>"""
    view = sources.get("view")
    conversation_len = len(sources["conversations"]) // 2

    truncate_ids = [
        i for i in range(conversation_len) if LLAVA_VLA_TOKEN in sources["conversations"][i * 2]["value"]
    ]
    denoise_idx = random.choice(truncate_ids + [conversation_len])

    sources = {
        "conversations": copy.deepcopy(sources["conversations"]) if sources else [],
        "action": [],
        "state": [],
        "image": [],
        "action_is_pad": [],
    }

    idx = 0
    for i in range(conversation_len):
        human_conversation = sources["conversations"][i * 2]["value"]
        conversation_image_n = human_conversation.count(LLAVA_IMAGE_TOKEN)

        le_image_n = 0
        while le_image_n < conversation_image_n:
            item = items[idx]
            actions, states = [], []
            images = [item[v] for v in view[idx]]

            for k, v in item.items():
                if k.startswith(ACTION) and "is_pad" not in k:
                    actions.append(v.unsqueeze(-1) if v.dim() == 1 else v)
                elif k.startswith(OBS_STATE):
                    states.append(v)
                elif k.startswith(ACTION) and "is_pad" in k:
                    action_is_pad = v

            # in the order of select_action_keys
            states = pad_vector(torch.cat(states, dim=-1), max_action_dim)
            actions = pad_vector(torch.cat(actions, dim=-1), max_action_dim)
            action_is_pads = action_is_pad.clone()

            idx += 1
            le_image_n += len(images)
            sources["image"].extend(images)

        gpt_conversation = sources["conversations"][i * 2 + 1]["value"]
        if human_conversation.endswith(LLAVA_VLA_TOKEN) and gpt_conversation.endswith(LLAVA_ACTION_TOKEN):
            sources["action"].append(actions)
            sources["state"].append(states)
            sources["action_is_pad"].append(action_is_pads)
            sources["conversations"][i * 2]["value"] = human_conversation.replace(
                LLAVA_VLA_TOKEN, TASK_VLA_TOKEN
            )

            if sample_actions:
                if i < denoise_idx:
                    replacement = f"{ACTION_START_TOKEN}{PASS_ACTION_TOKEN * chunk_size}{ACTION_END_TOKEN}"
                    sources["conversations"][i * 2 + 1]["value"] = gpt_conversation.replace(
                        LLAVA_ACTION_TOKEN, replacement
                    )
                elif i == denoise_idx:  # truncate
                    sources["conversations"] = sources["conversations"][: (i + 1) * 2]
                    return sources
    return sources


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\s*" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\s*" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_action_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_ACTION_TOKEN) + r"\n?"
    replacement = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def replace_state_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_STATE_TOKEN) + r"\n?"
    replacement = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_action_tokens(transformed_content)
        transformed_content = replace_state_tokens(transformed_content)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)
    return transformed_data


def pad_vector(vector, new_dim):
    """Can be (b s e) or (b e)"""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector
