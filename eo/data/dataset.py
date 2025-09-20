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

import os
import random

import torch
import transformers
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers.data.data_collator import DefaultDataCollator

from eo.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from eo.data.lerobot_dataset import MultiLeRobotDataset
from eo.data.multim_dataset import MultimodaDataset, pad_vector
from eo.data.schema import DataConfig, LerobotConfig
from eo.data.transforms import ImageTransforms, ImageTransformsConfig
from eo.train.pipeline_config import TrainPipelineConfig

"""multimodal lerobot datasets"""


class MultimodaLeRobotDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args: TrainPipelineConfig,
        processor: transformers.ProcessorMixin,
        padding=True,
    ):
        super().__init__()
        self.args = args
        self.processor = processor

        mm_dataset = []
        lerobot_dataset = []
        if args.data_path.endswith(".yaml"):
            data_configs = DataConfig.from_yaml(args.data_path)
            if args.train_lerobot_only:
                data_configs.mm_datasets = []
        else:
            data_configs = DataConfig(
                lerobot_datasets=[LerobotConfig(repo_id=args.data_path)],
                mm_datasets=[],
            )

        # load lerobot datasets
        if len(data_configs.lerobot_datasets) > 0:
            lerobot_dataset = MultiLeRobotDataset(
                data_configs=data_configs.lerobot_datasets,
                image_transforms=ImageTransforms(ImageTransformsConfig()),
                video_backend=args.lerobot_data_video_backend,
                state_mode=args.state_mode,
                chunk_size=args.chunk_size,
            )

        # load mm datasets
        if len(data_configs.mm_datasets) > 0:
            mm_dataset = MultimodaDataset(
                data_configs=data_configs.mm_datasets,
                # max_packed_length=args.max_packed_length,
                max_action_dim=args.max_action_dim,
                meta_dataset=lerobot_dataset,
                chunk_size=args.chunk_size,
            )

        # multi-modal datasets
        self.mm_dataset = mm_dataset
        self.lerobot_dataset = lerobot_dataset

        self.fps = args.fps
        self.padding = padding
        self.image_min_pixel = args.image_min_pixels
        self.image_max_pixel = args.image_max_pixels
        self.video_min_pixel = args.video_min_pixels
        self.video_max_pixel = args.video_max_pixels
        self.image_resized_w = args.image_resized_width
        self.image_resized_h = args.image_resized_height
        self.video_resized_w = args.video_resized_width
        self.video_resized_h = args.video_resized_height
        self.vision_base_paths = self.mm_dataset.vision_base_paths if mm_dataset else None

    @property
    def lengths(self):
        """group the lengths of the datasets, we set sample_actions to False \
            to avoid action sampling damaging the length of the dataset
            After the length is calculated, reset it back to True
        """
        if getattr(self, "cached_lengths", None):
            return self.cached_lengths
        return []

    def __len__(self):
        if self.args.train_mm_only:
            return len(self.mm_dataset)
        else:
            return len(self.mm_dataset) + len(self.lerobot_dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i < len(self.mm_dataset):
            sources = self.mm_dataset[i]
        else:
            item = self.lerobot_dataset[i - len(self.mm_dataset)]
            images, actions, states = [], [], []
            for k, v in item.items():
                if k.startswith(OBS_IMAGE):
                    images.append(v)
                elif k.startswith(ACTION) and "is_pad" not in k:
                    actions.append(v.unsqueeze(-1) if v.dim() == 1 else v)
                elif k.startswith(OBS_STATE):
                    states.append(v)
                elif k.startswith(ACTION) and "is_pad" in k:
                    action_is_pad = v
            states = pad_vector(torch.cat(states, dim=-1), self.args.max_action_dim)
            actions = pad_vector(torch.cat(actions, dim=-1), self.args.max_action_dim)
            action_is_pads = action_is_pad.clone()
            image_replacement = f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}" * len(images)
            states_replacement = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
            sources = {
                "conversations": [
                    {
                        "role": "user",
                        "content": f"{image_replacement}{states_replacement}{item['task']}{TASK_VLA_TOKEN}",
                    },
                    {
                        "role": "assistant",
                        "content": f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}",
                    },
                ],
                "action": [actions],
                "state": [states],
                "image": images,
                "action_is_pad": [action_is_pads],
            }
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            for image_file in image_files:
                if isinstance(image_file, str) and not image_file.startswith("http"):
                    image_folder = self.vision_base_paths[sources["vision_base_idx"]]
                    image_file = os.path.join(image_folder, image_file)
                elif isinstance(image_file, torch.Tensor):  # lerobot dataset
                    image_file = Image.fromarray((image_file * 255).to(torch.uint8).permute(1, 2, 0).numpy())
                images.append(
                    get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                    )
                )
        elif "video" in sources:
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            video_files = sources["video"]
            video_folder = self.vision_base_paths[sources["vision_base_idx"]]
            if isinstance(video_files, str):
                video_files = [video_files]
            videos = []
            for video_file in video_files:
                if isinstance(video_file, str) and not video_file.startswith("http"):
                    video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.video_min_pixels,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.args.fps,
                )
                videos.append(video_input)
        else:
            grid_key = pixel_key = images = videos = None

        actions = sources.get("action", [])
        states = sources.get("state")
        action_is_pad = sources.get("action_is_pad")
        conversations = sources["conversations"]

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        img_start = 0
        for _, j in enumerate(range(0, len(conversations), 2)):
            user_input = conversations[j]
            gpt_response = conversations[j + 1]
            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                img_num = user_input.count(DEFAULT_IMAGE_TOKEN)
                inputs = self.processor(
                    text=[user_input],
                    images=images[img_start : img_start + img_num] if images else None,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                img_start += img_num
            elif DEFAULT_VIDEO_TOKEN in user_input:
                inputs = self.processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
                all_second_gird.extend(inputs["second_per_grid_ts"])
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            else:
                prompt_input_ids = self.processor.tokenizer(
                    user_input, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"]

            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            response_input_ids = self.processor(text=[gpt_response], padding=False, return_tensors="pt")[
                "input_ids"
            ]
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            # ignore the action token
            cached_action_ids = torch.tensor([self.processor.action_token_id, self.processor.action_pass_id])
            action_mask = torch.isin(labels, cached_action_ids)
            labels[action_mask] = IGNORE_INDEX

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = {
            "input_ids": input_ids,
            "seq_length": input_ids.shape[0],
            "attention_mask": attention_mask,
        }

        if not self.args.train_lerobot_only:
            data_dict["labels"] = labels

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        if len(actions) > 0:
            states = torch.stack(states, dim=0)
            actions = torch.stack(actions, dim=0)
            action_is_pad = torch.stack(action_is_pad, dim=0)

            data_dict["states"] = states
            data_dict["actions"] = actions
            data_dict["action_is_pad"] = action_is_pad

        return data_dict

    def info_qwen_vision_fetch(self):
        from qwen_vl_utils import smart_resize

        if not self.lerobot_dataset:
            return

        print(f"qwen2.5 vl min pixel {self.args.image_min_pixels}, max pixel {self.args.image_max_pixels}")
        for dataset in self.lerobot_dataset._datasets:
            meta_features, video_key = dataset.meta.features, dataset.select_video_keys
            for k in video_key:
                h, w = meta_features[k]["shape"][0], meta_features[k]["shape"][1]
                h_bar, w_bar = smart_resize(
                    h, w, min_pixels=self.args.image_min_pixels, max_pixels=self.args.image_max_pixels
                )
                print(f"{dataset.repo_id:<40} | {k:<40} | resize from {h, w} to {h_bar, w_bar} |")


class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples as part of the dataloader.

    We may randomly sample some examples from the mini action set to avoid parameter
    tracking issue(lm_head and flow head) during training.
    """

    def __init__(
        self,
        dataset: Dataset,
        pack_length: int = 8192,
        mini_action_set_length: int = 512,
    ) -> None:
        self.dataset = dataset
        self.pack_length = pack_length
        self.packed_indices = []
        self.packed_lengths = []

        self.mini_action_set = []
        self.mini_action_set_length = mini_action_set_length

    def _pack(self) -> None:
        import binpacking

        lengths = self.dataset.lengths
        print(f"packing data with length {len(lengths)} ...")
        grouped_indices = binpacking.to_constant_volume(
            list(enumerate(lengths)),
            self.pack_length,
            weight_pos=1,
        )

        for group in grouped_indices:
            indices = [idx for idx, _ in group]
            length = sum(length for _, length in group)
            self.packed_indices.append(indices)
            self.packed_lengths.append(length)

        mini_action_set = [
            idx
            for idx in range(len(self.dataset.mm_dataset), len(lengths))
            if lengths[idx] < self.mini_action_set_length
        ]
        self.mini_action_set.extend(mini_action_set)
        print(f"* mini action set: {len(self.mini_action_set)}")

    def __len__(self):
        return len(self.packed_indices)

    def __getitem__(self, index: int):
        indices = self.packed_indices[index]
        items = []
        no_actions = True

        for i in indices:
            data = self.dataset[i]
            items.append(data)
            if "actions" in data:
                no_actions = False

        if no_actions and len(self.mini_action_set) > 0:
            select_data = self.dataset[random.choice(self.mini_action_set)]
            select_data["pad_action_example"] = True
            items.append(select_data)
        return items

    @property
    def lerobot_dataset(self):
        return self.dataset.lerobot_dataset

    def info_qwen_vision_fetch(self):
        self.dataset.info_qwen_vision_fetch()


""" Data Collators """


class MultimodaDataCollator(DefaultDataCollator):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        batch_actions = []
        batch_states = []
        batch_action_is_pad = []

        is_labels_provided = "labels" in examples[0]
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            if is_labels_provided:
                batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            if "actions" in keys:
                batch_actions.append(example["actions"])
                batch_states.append(example["states"])
                batch_action_is_pad.append(example["action_is_pad"])

        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = input_ids != self.pad_token_id

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if is_labels_provided:
            labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)
            data_dict["labels"] = labels

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_actions) > 0:
            actions = torch.cat(batch_actions, dim=0)
            states = torch.cat(batch_states, dim=0)
            action_is_pad = torch.cat(batch_action_is_pad, dim=0)  # (b s)

            data_dict["actions"] = actions
            data_dict["states"] = states
            data_dict["action_is_pad"] = action_is_pad
        return data_dict


class MultimodaPackedDataCollator(DefaultDataCollator):
    """Collate features for supervised fine-tuning."""

    def __init__(
        self,
        *args,
        return_position_ids=True,
        separator_id=-100,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id

    def __call__(self, features, return_tensors=None, separator_id=None):
        return_tensors = return_tensors or self.return_tensors
        separator_id = separator_id or self.separator_id
        is_labels_provided = "labels" in features[0][0]

        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        batch_actions = []
        batch_states = []
        batch_action_is_pad = []

        batch_position_ids = []
        padded_action_example = False

        for examples in features:
            if examples[-1].get("pad_action_example"):
                if padded_action_example:  # only add one action example to avoid OOM
                    examples.pop(-1)
                padded_action_example = True

            for example in examples:
                keys = example.keys()
                batch_input_ids.append(example["input_ids"])
                batch_position_ids.append(torch.arange(len(example["input_ids"])))

                if "pixel_values_videos" in keys:
                    batch_pixel_video_values.append(example["pixel_values_videos"])
                    batch_video_thw.append(example["video_grid_thw"])

                elif "pixel_values" in keys:
                    batch_pixel_values.append(example["pixel_values"])
                    batch_image_thw.append(example["image_grid_thw"])

                if is_labels_provided:
                    batch_label_ids.append(example["labels"])

                if "second_per_grid_ts" in keys:
                    batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

                if "actions" in keys:
                    batch_actions.append(example["actions"])
                    batch_states.append(example["states"])
                    batch_action_is_pad.append(example["action_is_pad"])

        data_dict = {
            "input_ids": torch.cat(batch_input_ids, dim=0).unsqueeze(0),  # (b=1, s)
            "packed_position_ids": batch_position_ids,
        }
        if is_labels_provided:
            data_dict["labels"] = torch.cat(batch_label_ids, dim=0).unsqueeze(0)  # (b=1, s)

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_actions) > 0:
            actions = torch.cat(batch_actions, dim=0)
            states = torch.cat(batch_states, dim=0)
            action_is_pad = torch.cat(batch_action_is_pad, dim=0)  # (b s)

            data_dict["actions"] = actions
            data_dict["states"] = states
            data_dict["action_is_pad"] = action_is_pad
        return data_dict


def make_supervised_data_module(processor, args: TrainPipelineConfig):
    """build datasets and collator"""
    dataset = MultimodaLeRobotDataset(args=args, processor=processor)
    if args.pack_dataset:
        dataset = PackedDataset(dataset, args.max_packed_length, args.mini_action_set_length)
        data_collator = MultimodaPackedDataCollator()
    else:
        data_collator = MultimodaDataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": data_collator}


""" Helper Functions """


def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel, width, height):
    content = {"type": "image", "image": image_path, "min_pixels": min_pixel, "max_pixels": max_pixel}

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    image_input, _ = process_vision_info(messages)
    return image_input[0]


def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_frames": 30,
        "max_frames": 60,
        "fps": fps,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    messages = [{"role": "user", "content": [content]}]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs
