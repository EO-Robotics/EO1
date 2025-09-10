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

"""Schema for the dataset configuration."""

from dataclasses import dataclass, field

import yaml


@dataclass
class MMDatasetConfig:
    json_path: str
    sampling_strategy: str = "all"
    vision_base_path: str | None = None


@dataclass
class LerobotConfig:
    repo_id: str
    root: str
    episodes: list[int] | None = None
    delta_action: bool = False
    state_mode: str = "MEAN_STD"

    train_subtask: str | bool | None = False  # Optional[true, false, mix:0.5, cumulate]
    select_video_keys: list[str] = None
    select_action_keys: list[str] = None
    select_state_keys: list[str] = None
    effector_indices: list[int] = None
    weight: float | None = None


@dataclass
class DataConfig:
    mm_datasets: list[MMDatasetConfig] = field(default_factory=list)
    lerobot_datasets: list[LerobotConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            mm_datasets=[MMDatasetConfig(**d) for d in raw.get("mm_datasets") or []],
            lerobot_datasets=[LerobotConfig(**d) for d in raw.get("lerobot_datasets") or []],
        )
