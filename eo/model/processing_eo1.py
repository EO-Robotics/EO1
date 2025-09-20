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

from typing import Union

import numpy as np
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import OBS_STATE
from lerobot.datasets.utils import cast_stats_to_numpy
from lerobot.policies.normalize import Normalize, Unnormalize
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

"""constants"""
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

ACTION_START_TOKEN = "<|action_start|>"
DEFAULT_ACTION_TOKEN = "<|action_pad|>"
PASS_ACTION_TOKEN = "<|action_pass|>"
ACTION_END_TOKEN = "<|action_end|>"

STATE_START_TOKEN = "<|state_start|>"
DEFAULT_STATE_TOKEN = "<|state_pad|>"
STATE_END_TOKEN = "<|state_end|>"
TASK_VLA_TOKEN = "<|vla|>"

RobotInput = Union[np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]]


class EO1VisionVideosProcessorKwargs(VideosKwargs, total=False):
    fps: list[float] | float


class EO1VisionImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None


class EO1VisionTextKwargs(TextKwargs):
    noise_token_num: int | None
    noise_prompt: str | None


class EO1VisionProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: EO1VisionTextKwargs
    images_kwargs: EO1VisionImagesKwargs
    videos_kwargs: EO1VisionVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class EO1VisionProcessor(ProcessorMixin):
    """EEO1Vision Processor for Image, Text, Video, and Robotic Action Processing"""

    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        robot_config=None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", DEFAULT_IMAGE_TOKEN)
        self.video_token = getattr(tokenizer, "video_token", DEFAULT_VIDEO_TOKEN)
        self.action_token = getattr(tokenizer, "action_token", DEFAULT_ACTION_TOKEN)
        self.state_token = getattr(tokenizer, "state_token", DEFAULT_STATE_TOKEN)

        # robot policy
        self.action_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN) or 151666
        self.action_pass_id = tokenizer.convert_tokens_to_ids(PASS_ACTION_TOKEN) or 151667
        self.robot_config = robot_config or {}
        self.set_normalization(self.robot_config)

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def set_normalization(self, robot_config: dict):
        features, stats, state_mode = (
            robot_config.get("features"),
            robot_config.get("stats"),
            robot_config.get("state_mode"),
        )
        if None in [features, stats, state_mode]:
            return
        else:
            normalization_mapping = {
                "STATE": NormalizationMode(state_mode),
                "ACTION": NormalizationMode(state_mode),
            }
            normalize_inputs, unnormalize_outputs = {}, {}
            for repo_id, fea in features.items():
                stat = cast_stats_to_numpy(stats[repo_id])
                fea = dataset_to_policy_features(fea)

                input_features = {k: v for k, v in fea.items() if v.type == FeatureType.STATE}
                output_features = {k: v for k, v in fea.items() if v.type == FeatureType.ACTION}

                normalize_inputs[repo_id] = Normalize(input_features, normalization_mapping, stat)
                unnormalize_outputs[repo_id] = Unnormalize(output_features, normalization_mapping, stat)

            self.robot_config = dict(robot_config)
            self.normalize_inputs, self.unnormalize_outputs = normalize_inputs, unnormalize_outputs

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        states: RobotInput = None,
        actions: RobotInput = None,
        **kwargs: Unpack[EO1VisionProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            EO1VisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        noise_token_num = output_kwargs["text_kwargs"].pop("noise_token_num", None)
        output_kwargs["text_kwargs"].pop("noise_prompt", None)

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to \
                        the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # noise tokens
        noise_token_num = noise_token_num or self.robot_config.get("action_chunk_size")
        for i in range(len(text)):
            while self.action_token in text[i]:
                text[i] = text[i].replace(
                    self.action_token,
                    "<|placeholder|>" * noise_token_num,
                    1,
                )
            text[i] = text[i].replace("<|placeholder|>", self.action_token)

        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        # robot inputs
        robot_inputs = {}

        if states is not None:
            if isinstance(states, list):
                states = torch.stack(states, dim=0)
            if states.ndim == 1:
                states = states.unsqueeze(0)
            robot_inputs.update({"states": states})

        if actions is not None:
            if isinstance(actions, list):
                actions = torch.stack(actions, dim=0)
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            robot_inputs.update({"actions": actions})

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **robot_inputs},
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"] + ["states", "actions"]

    @torch.no_grad
    def _prepare_robot_inputs(self, batch: dict):
        """Prepare model inputs from raw robot batch"""
        batch_messages = []
        batch_states = []
        max_state_dim = self.robot_config.get("max_state_dim", 32)

        state_keys = [x for x in batch.keys() if x.startswith(OBS_STATE)]
        batch_size = len(batch[state_keys[0]])

        if "repo_id" in batch:
            repo_ids = batch.pop("repo_id")
        else:
            print("no repo_id found, use the first one in normalize_inputs")
            repo_ids = list(self.normalize_inputs.keys())[0]
        repo_ids = [repo_ids] * batch_size if isinstance(repo_ids, str) else repo_ids

        for i, repo_id in enumerate(repo_ids):
            mini_batch = {k: v[i] for k, v in batch.items()}

            normalize_inputs = self.normalize_inputs[repo_id]
            select_video_keys = self.robot_config["select_video_keys"][repo_id]
            select_state_keys = self.robot_config["select_state_keys"][repo_id]

            for k in normalize_inputs.features:
                if not isinstance(mini_batch[k], torch.Tensor):
                    mini_batch[k] = torch.tensor(mini_batch[k], dtype=torch.float32)

            mini_batch = normalize_inputs(mini_batch)
            states = torch.concat([mini_batch[k] for k in select_state_keys])
            batch_states.append(pad_vector(states, max_state_dim))
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": mini_batch[k]} for k in select_video_keys),
                        {"type": "state", "state": []},  # chat template state token
                        {"type": "text", "text": f"{mini_batch['task']}{TASK_VLA_TOKEN}"},
                    ],
                }
            ]
            batch_messages += [messages]
        return repo_ids, batch_messages, batch_states

    def _process_robot_outputs(self, repo_ids: list[str], actions: torch.Tensor):
        """Process model outputs back to robot format"""
        output_actions = []
        for i, repo_id in enumerate(repo_ids):
            unnormalize_outputs = self.unnormalize_outputs[repo_id]
            select_action_keys = self.robot_config["select_action_keys"][repo_id]
            features = unnormalize_outputs.features
            cum_dims = [0] + np.cumsum([features[k].shape[0] for k in select_action_keys]).tolist()
            origin_action = torch.tensor(actions[i], dtype=torch.float32)[..., : cum_dims[-1]]
            batch = {
                k: origin_action[..., cum_dims[m] : cum_dims[m + 1]] for m, k in enumerate(select_action_keys)
            }
            unnorm_actions = unnormalize_outputs(batch)
            unnorm_actions = torch.concat([unnorm_actions[k] for k in select_action_keys], -1)
            output_actions.append(unnorm_actions)
        output_actions = torch.stack(output_actions, dim=0)

    @torch.no_grad
    def select_action(self, model, batch: dict, **kwargs):
        repo_ids, batch_messages, batch_states = self._prepare_robot_inputs(batch)

        noise_prompt = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"
        inputs = self.apply_chat_template(
            batch_messages,
            states=batch_states,
            add_generation_prompt=True,
            noise_prompt=noise_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True)
        actions = outputs.actions.cpu()
        output_actions = self._process_robot_outputs(repo_ids, actions)
        return BatchFeature({"action": output_actions})


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Lerobot robot policy features"""
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = ft["names"]
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key.startswith("action"):
            type = FeatureType.ACTION
        else:
            continue
        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )
    return policy_features


def pad_vector(vector, new_dim=32):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


EO1VisionProcessor.register_for_auto_class()
