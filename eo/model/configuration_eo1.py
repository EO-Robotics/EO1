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

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)


class EO1VisionVLTextConfig(Qwen2_5_VLTextConfig):
    def __init__(
        self,
        state_token_id=None,
        action_token_start_id=None,
        action_token_id=None,
        action_pass_id=None,
        vision_token_start_id=None,
        image_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_token_id = state_token_id
        self.action_token_start_id = action_token_start_id
        self.action_token_id = action_token_id
        self.action_pass_id = action_pass_id

        self.vision_token_start_id = vision_token_start_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class EO1VisionFlowMatchingConfig(Qwen2_5_VLConfig):
    model_type = "onevision_fm"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": EO1VisionVLTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        # flow matching specific
        action_chunk_size=50,
        max_action_dim=32,
        num_denoise_steps=5,
        action_act="linear",
        num_action_layers=2,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            **kwargs,
        )
        self.action_chunk_size = action_chunk_size
        self.max_action_dim = max_action_dim
        self.num_denoise_steps = num_denoise_steps
        self.action_act = action_act
        self.num_action_layers = num_action_layers


EO1VisionFlowMatchingConfig.register_for_auto_class()
