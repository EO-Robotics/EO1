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

import math
import random
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.transforms import ImageTransformConfig, RandomSubsetApply, SharpnessJitter
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2._utils import query_size


class RandomScaleCrop:
    """Randomly crop the image according to the scale parameter, keeping the original aspect ratio.
    Args:
        scale: The range of size (as a fraction of the original size) to sample from.
    """

    def __init__(self, scale: tuple[float, float]):
        self.scale = scale

    def __call__(self, img: Any) -> Any:
        h, w = query_size(img)

        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area

        aspect_ratio = w / h
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(target_area / crop_h))

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        return v2.functional.crop(img, i, j, crop_h, crop_w)


@dataclass
class ImageTransformsConfig:
    """Image and Wrist transforms for the LeRobot dataset."""

    enable: bool = True
    max_num_transforms: int = 3
    random_order: bool = True
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "crop": ImageTransformConfig(
                type="RandomScaleCrop",
                kwargs={"scale": (0.95, 1.0)},
            ),
            "rotate": ImageTransformConfig(
                type="RadomRotate",
                kwargs={"degrees": (-3, 3)},
            ),
        }
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.weights = []
        self.transforms = {}

        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue
            match tf_cfg.type:
                case "Identity":
                    self.transforms[tf_name] = v2.Identity(**tf_cfg.kwargs)
                case "ColorJitter":
                    self.transforms[tf_name] = v2.ColorJitter(**tf_cfg.kwargs)
                case "SharpnessJitter":
                    self.transforms[tf_name] = SharpnessJitter(**tf_cfg.kwargs)
                case "RadomRotate":
                    self.transforms[tf_name] = v2.RandomRotation(**tf_cfg.kwargs)
                case "RandomScaleCrop":
                    self.transforms[tf_name] = RandomScaleCrop(**tf_cfg.kwargs)
                case _:
                    self.transforms[tf_name] = v2.Identity(**tf_cfg.kwargs)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        outputs = self.tf(*inputs)
        return outputs


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
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
