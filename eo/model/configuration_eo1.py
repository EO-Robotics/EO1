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

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)


class EO1VisionFlowMatchingConfig(PretrainedConfig):
    model_type = "eo1"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        action_chunk_size=50,
        max_action_dim=32,
        num_denoise_steps=10,
        action_act="linear",
        num_action_layers=2,
        state_token_id=151670,
        action_token_id=151666,
        action_pass_id=151667,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"](
                hidden_size=1280,
                out_hidden_size=2048,
                tokens_per_second=2,
            )

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.state_token_id = state_token_id
        self.action_token_id = action_token_id
        self.action_pass_id = action_pass_id

        self.action_chunk_size = action_chunk_size
        self.max_action_dim = max_action_dim
        self.num_denoise_steps = num_denoise_steps
        self.action_act = action_act
        self.num_action_layers = num_action_layers

        super().__init__(**kwargs)


EO1VisionFlowMatchingConfig.register_for_auto_class()
