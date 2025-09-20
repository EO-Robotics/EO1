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

"""This module provides custom LeRobot dataset implementations that extend the base classes from the `lerobot` library.

The `LeRobotDataset` class adds features such as:
- Subtask training modes.
- Selection of specific video, state, and action keys.
- Dataset weighting for sampling.
- Delta action calculation.
- State and action normalization.

The `MultiLeRobotDataset` class is a wrapper for loading and combining multiple `LeRobotDataset` instances,
potentially from different repositories. It supports parallel data loading to speed up the process.
"""

import bisect
import multiprocessing
import os
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import datasets
import torch
from datasets import load_dataset
from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset as BaseLeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset as BaseMultiLeRobotDataset,
)
from lerobot.datasets.utils import hf_transform_to_torch, serialize_dict
from lerobot.policies.normalize import Normalize
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from .schema import LerobotConfig
from .transforms import dataset_to_policy_features

"""lerobot datasets"""


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        state_mode: str = "MEAN_STD",
        select_video_keys: list[str] | None = None,
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
        train_subtask: str | None = None,  # ["cumulate", "mixture:0.5", "true"]
        delta_action: bool = False,
        effector_indices: list[int] | None = None,
        weight: float | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        # set weight for the dataset
        self.set_weight(weight)

        # remove unused features for efficiency
        self.set_feature_keys(select_video_keys, select_state_keys, select_action_keys)

        # set delta action mode
        self.set_delta_action(delta_action, effector_indices)

        # set nomalizer for multiple datasets
        self.set_normalization(state_mode)

        # calculate substake indices
        self.set_train_subtask(train_subtask)

        # set episode from index
        self.get_episode_from_index(episodes)

    def set_train_subtask(self, train_subtask: str | None = None):
        """set train subtask mode for lerobot dataset"""
        self.train_subtask = train_subtask
        if train_subtask is None:
            return

        # calculate the start and end indices of each episode
        task_sizes = {}
        try:
            for _, ep in self.meta.episodes.items():
                task_sizes[ep["episode_index"]] = [item["end_frame"] for item in ep["action_config"]]
        except Exception as e:
            print(f"[warn] {self.repo_id} failed to calculate episode subtask cumulate: {e}")
            self.train_subtask = None

        self.task_sizes = task_sizes
        print(f"* set train_subtask {self.train_subtask} for {self.repo_id}")

    def set_feature_keys(self, video_keys=None, state_keys=None, action_keys=None):
        """select video, state and action keys from the dataset"""
        self.select_video_keys = video_keys or self.meta.video_keys
        self.select_state_keys = state_keys or [x for x in self.meta.features if x.startswith(OBS_STATE)]
        self.select_action_keys = action_keys or [x for x in self.meta.features if x.startswith(ACTION)]
        self.select_feature_keys = self.select_video_keys + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{k}_is_pad" for k in self.select_action_keys]

    def set_weight(self, weight: float | None):
        """set weight for lerobot dataset"""
        self.weight = weight
        if weight is not None:
            self._num_frames_weight = int(self.num_frames * weight)
            print(
                f"* set weight {weight} for {self.repo_id}, num_frames: {self.num_frames} -> {self._num_frames_weight}"
            )
        else:
            self._num_frames_weight = self.num_frames

    def set_delta_action(self, delta_action: bool, effector_indices: list[int] | None = None):
        """set delta action mode for lerobot dataset"""
        self.delta_action = delta_action
        self.effector_indices = effector_indices or []

        if not delta_action:
            return

        import numpy as np

        print(f"* set delta action mode for {self.repo_id} ...")

        acum_idx = 0
        cumulative_lengths = self.episode_data_index["to"]
        for k in self.select_action_keys:
            action = torch.stack(self.hf_dataset[k]).numpy()
            if action.ndim == 1:
                action = action[:, None]
            delta_action = np.diff(action, axis=0)

            delta_action = np.concatenate([delta_action, delta_action[-1:]], axis=0)
            for end_idx in cumulative_lengths:
                delta_action[end_idx - 1] = delta_action[end_idx - 2]

            # set effector indices
            dim = action.shape[-1]
            mask = np.array([False] * dim)
            for i in self.effector_indices:
                idx = i - acum_idx
                if 0 <= idx < dim:
                    mask[idx] = True
            delta_action[:, mask] = action[:, mask]
            acum_idx += dim

            # update hf dataset, only list with numpy is supported
            self.hf_dataset = self.hf_dataset.remove_columns(k)
            self.hf_dataset = self.hf_dataset.add_column(k, list(delta_action.squeeze()))

            self.meta.stats[k]["min"] = delta_action.min(0)
            self.meta.stats[k]["max"] = delta_action.max(0)
            self.meta.stats[k]["mean"] = delta_action.mean(0)
            self.meta.stats[k]["std"] = delta_action.std(0)

    def set_normalization(self, state_mode: str = "MEAN_STD"):
        """set normalization mode for lerobot dataset"""
        features = dataset_to_policy_features(self._features)
        mapping = {"STATE": NormalizationMode(state_mode), "ACTION": NormalizationMode(state_mode)}
        self.normalizer = Normalize(features, mapping, self._stats)

    def get_episode_from_index(self, episodes: list[int] | None = None):
        """
        episodes: list of episode indices
        """
        if episodes is None:
            self.episode_from_index = None
        else:
            self.episode_from_index = {ep_idx: i for i, ep_idx in enumerate(episodes)}

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train", keep_in_memory=False)
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", keep_in_memory=False)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.select_video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def __len__(self):
        return self._num_frames_weight

    def _get_query_indices(
        self, idx: int, ep_idx: int, delta_indices: dict = None
    ) -> tuple[dict[str, list[int | bool]]]:
        if self.episode_from_index is not None:
            ep_idx = self.episode_from_index[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        delta_indices = delta_indices or self.delta_indices
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in delta_indices.items()  # {"action": [0-50)}
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        if self.weight is not None:
            idx = random.randint(0, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx, delta_indices)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.select_video_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        if self.train_subtask and len(self.task_sizes[ep_idx]) > 0:
            try:
                if self.train_subtask == "cumulate":
                    task_text = " ".join(
                        [item["action_text"] for item in self.meta.episodes[ep_idx]["action_config"]]
                    )
                else:
                    sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], item["frame_index"])
                    sub_idx = min(sub_idx, len(self.task_sizes[ep_idx]) - 1)
                    task_text = self.meta.episodes[ep_idx]["action_config"][sub_idx]["action_text"]

                    if self.train_subtask.startswith("mixture"):
                        global_text = self.meta.tasks[task_idx]
                        w = float(self.train_subtask.split(":")[-1])
                        # random select from [global_text, subtask_text]
                        task_text = random.choices([task_text, global_text], weights=[w, 1 - w])[0]

            except Exception as e:
                print(f"[warn] {self.repo_id} failed to get subtask {idx} / {len(self.hf_dataset)}: {e}")
                task_text = self.meta.tasks[task_idx]

        else:
            task_text = self.meta.tasks[task_idx]

        item["task"] = task_text
        return self.post_process(item)

    @property
    def _stats(self) -> datasets.Features:
        return {k: self.meta.stats[k] for k in (self.select_state_keys + self.select_action_keys)}

    @property
    def _features(self) -> dict[str, dict]:
        return {k: self.meta.features[k] for k in self.select_feature_keys}

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """sort the keys in the order of select_feature_keys"""
        item = {k: item[k] for k in (self.select_feature_keys + ["task"] + self.select_action_is_pad_keys)}
        item = self.normalizer(item)
        return item


class MultiLeRobotDataset(BaseMultiLeRobotDataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`.
    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        data_configs: list[LerobotConfig],
        state_mode: str = "MEAN_STD",
        image_transforms: Callable | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        chunk_size: int = 32,
        **kwargs: Any,
    ):
        self.data_configs = data_configs
        self.chunk_size = chunk_size
        self.disabled_features = set()

        # load lerobot datasets
        num_processes = int(os.environ.get("DATASET_NUM_PROCESSES", 8))
        print(f"* load {len(data_configs)} lerobot datasets with {num_processes} processes ...")
        pool = multiprocessing.Pool(processes=num_processes)
        fn = partial(
            _load_single_lerobot_dataset,
            data_configs=self.data_configs,
            image_transforms=image_transforms,
            download_videos=download_videos,
            video_backend=video_backend,
            chunk_size=chunk_size,
        )
        datasets = list(
            tqdm(
                pool.imap(fn, range(len(data_configs))),
                total=len(data_configs),
                desc="Loading lerobot datasets",
            )
        )
        pool.close()
        pool.join()

        self._datasets = [ds for ds in datasets if ds is not None]
        self.repo_ids = [ds.repo_id for ds in self._datasets]
        print(f"successfully load dataset {len(self.repo_ids)}/{len(data_configs)}:\n{self.repo_ids} ")

        self._repo_ids_index = {repo_id: i for i, repo_id in enumerate(self.repo_ids)}
        self.cumulative_sizes = ConcatDataset.cumsum(self._datasets)
        self.image_transforms = image_transforms

        # set select feature keys
        self.state_mode = state_mode
        self._select_video_keys = {
            ds.repo_id.replace("/", "."): ds.select_video_keys for ds in self._datasets
        }
        self._select_state_keys = {
            ds.repo_id.replace("/", "."): ds.select_state_keys for ds in self._datasets
        }
        self._select_action_keys = {
            ds.repo_id.replace("/", "."): ds.select_action_keys for ds in self._datasets
        }

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        item = self._datasets[dataset_idx][sample_idx]
        return self.post_process(item)

    def getitem_by_id(self, repo_id: str, idx: int, chunk_size: int = None) -> dict[str, torch.Tensor]:
        """get an item by repo_id and index."""
        dataset_idx = self._repo_ids_index.get(repo_id)
        if dataset_idx is None:
            raise ValueError(f"invalid dataset: {repo_id}. available dataset: {self.repo_ids}")
        lerobot_dataset = self._datasets[dataset_idx]

        delta_indices = None
        if chunk_size is not None:
            delta_indices = {k: list(range(0, chunk_size)) for k in lerobot_dataset.select_action_keys}

        item = lerobot_dataset.__getitem__(idx, delta_indices=delta_indices)
        return self.post_process(item)

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """unify video keys across datasets."""
        return item

    @property
    def _features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            features[repo_id_suffix] = dataset._features
        return features

    @property
    def _stats(self) -> datasets.Features:
        stats = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            stats[repo_id_suffix] = dataset._stats
        return stats

    @property
    def configuration(self) -> dict:
        return {
            "features": self._features,
            "stats": serialize_dict(self._stats),
            "state_mode": self.state_mode,
            "select_video_keys": self._select_video_keys,
            "select_state_keys": self._select_state_keys,
            "select_action_keys": self._select_action_keys,
        }


def _load_single_lerobot_dataset(
    idx,
    data_configs: list[LerobotConfig],
    image_transforms: Callable | None = None,
    download_videos: bool = True,
    video_backend: str | None = None,
    chunk_size: int = 32,
):
    """load a single lerobot dataset"""
    try:
        data_config = data_configs[idx]
        data_path = Path(data_config.root or HF_LEROBOT_HOME) / data_config.repo_id
        meta = LeRobotDatasetMetadata(data_config.repo_id, data_path)
        select_action_keys = data_config.select_action_keys or [
            k for k in meta.features if k.startswith(ACTION)
        ]
        delta_timestamps = {k: [i / meta.fps for i in range(0, chunk_size)] for k in select_action_keys}
        dataset = LeRobotDataset(
            data_config.repo_id,
            root=data_path,
            episodes=data_config.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            download_videos=download_videos,
            video_backend=video_backend,
            select_video_keys=data_config.select_video_keys,
            select_state_keys=data_config.select_state_keys,
            select_action_keys=data_config.select_action_keys,
            state_mode=data_config.state_mode,
            train_subtask=data_config.train_subtask,
            delta_action=data_config.delta_action,
            effector_indices=data_config.effector_indices,
            weight=data_config.weight,
        )
    except Exception as e:
        print(f"[warn] read dataset {data_config.repo_id} failed, skipped!")
        print(e)
        return None
    return dataset
