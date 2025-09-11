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

"""This module defines constants used throughout the application,
including system messages and various special tokens for language
and vision models.
These tokens are used to demarcate different types of input such
as images, videos, actions, and states, with specific sets for
different model architectures like LLaVA and datasets like LeRobot.
"""

SYSTEM_MESSAGE = "You are a helpful physical assistant."

# qwen2.5-vl special tokens
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

# EO-1 special tokens
ACTION_START_TOKEN = "<|action_start|>"
DEFAULT_ACTION_TOKEN = "<|action_pad|>"
PASS_ACTION_TOKEN = "<|action_pass|>"
ACTION_END_TOKEN = "<|action_end|>"
STATE_START_TOKEN = "<|state_start|>"
DEFAULT_STATE_TOKEN = "<|state_pad|>"
STATE_END_TOKEN = "<|state_end|>"
TASK_VLA_TOKEN = "<|vla|>"

# llava style special tokens
IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
LLAVA_ACTION_TOKEN = "<action>"
LLAVA_STATE_TOKEN = "<state>"
LLAVA_VLA_TOKEN = "<vla>"
