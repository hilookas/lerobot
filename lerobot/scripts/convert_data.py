#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
python lerobot/scripts/convert_data.py \
--raw-repo-id lookas/astra_test \
--repo-id lookas/astra_test_converted
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.video_utils import transcode_video
import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.common.datasets.utils import flatten_dict

from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, save_meta_data
import tqdm

import gc
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, save_images_to_video


def convert_data(
    raw_repo_id: str,
    repo_id: str,
    push_to_hub: bool = True,
    root: str = "data",
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    episodes: list[int] | None = None,
    force_override: bool = False,
    resume: bool = False,
    cache_dir: Path = Path("/tmp"),
    tests_data_dir: Path | None = None,
    encoding: dict | None = None,
):
    check_repo_id(repo_id)
    user_id, dataset_id = repo_id.split("/")

    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    if local_dir:
        # Robustify when `local_dir` is str instead of Path
        local_dir = Path(local_dir)

        # Send warning if local_dir isn't well formated
        if local_dir.parts[-2] != user_id or local_dir.parts[-1] != dataset_id:
            warnings.warn(
                f"`local_dir` ({local_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht'). Following this naming convention is advised, but not mandatory.",
                stacklevel=1,
            )

        # Check we don't override an existing `local_dir` by mistake
        if local_dir.exists():
            if force_override:
                shutil.rmtree(local_dir)
            elif not resume:
                raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override 1`.")

        meta_data_dir = local_dir / "meta_data"
        videos_dir = local_dir / "videos"
    else:
        # Temporary directory used to store images, videos, meta_data
        meta_data_dir = Path(cache_dir) / "meta_data"
        videos_dir = Path(cache_dir) / "videos"
    
    # converting
    
    raw_dataset = LeRobotDataset(raw_repo_id)
    for episode_index in tqdm.tqdm(range(raw_dataset.num_episodes)):
        for key in raw_dataset.video_frame_keys:
            tmp_fname = f"{key}_episode_{episode_index:06d}.mp4"
            tmp_video_path = raw_dataset.videos_dir / tmp_fname
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = videos_dir / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            transcode_video(tmp_video_path, video_path, filter=["-filter:v", "scale=640:-1"], overwrite=True)
    
    modified_rows = []
    for row in raw_dataset.hf_dataset:
        modified_row = {}
        modified_row['observation.state'] = torch.concatenate([
            row["observation.state.arm_l"],
            row["observation.state.gripper_l"],
            row["observation.state.arm_r"],
            row["observation.state.gripper_r"],
            row["observation.state.base"],
        ])
        modified_row['action'] = torch.concatenate([
            row["action.arm_l"],
            row["action.gripper_l"],
            row["action.arm_r"],
            row["action.gripper_r"],
            row["action.base"],
        ])
        
        for key in row:
            if 'observation.state.' not in key and 'action.' not in key:
                modified_row[key] = row[key]
        modified_rows.append(modified_row)
        
    features = {}
    for key in modified_row:
        if key == 'observation.state' or key == 'action':
            features[key] = Sequence(
                length=modified_row[key].shape[0], feature=Value(dtype="float32", id=None)
            )
        else:
            features[key] = raw_dataset.features[key]

    hf_dataset = Dataset.from_list(modified_rows, features=Features(features))   
    hf_dataset.set_transform(hf_transform_to_torch)
        
    episode_data_index = raw_dataset.episode_data_index
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": raw_dataset.info["fps"],
        "video": video,
    }
    if video:
        info["encoding"] = {'vcodec': 'libsvtav1', 'pix_fmt': 'yuv420p', 'g': 2, 'crf': 30}

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if local_dir:
        hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
        hf_dataset.save_to_disk(str(local_dir / "train"))

    if push_to_hub or local_dir:
        # mandatory for upload
        save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if push_to_hub:
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
        api = HfApi()
        api.create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    if tests_data_dir:
        # get the first episode
        num_items_first_ep = episode_data_index["to"][0] - episode_data_index["from"][0]
        test_hf_dataset = hf_dataset.select(range(num_items_first_ep))
        episode_data_index = {k: v[:1] for k, v in episode_data_index.items()}

        test_hf_dataset = test_hf_dataset.with_format(None)
        test_hf_dataset.save_to_disk(str(tests_data_dir / repo_id / "train"))

        tests_meta_data = tests_data_dir / repo_id / "meta_data"
        save_meta_data(info, stats, episode_data_index, tests_meta_data)

        # copy videos of first episode to tests directory
        episode_index = 0
        tests_videos_dir = tests_data_dir / repo_id / "videos"
        tests_videos_dir.mkdir(parents=True, exist_ok=True)
        for key in lerobot_dataset.video_frame_keys:
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            shutil.copy(videos_dir / fname, tests_videos_dir / fname)

    if local_dir is None:
        # clear cache
        shutil.rmtree(meta_data_dir)
        shutil.rmtree(videos_dir)

    return lerobot_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-repo-id",
        type=str,
        required=True,
        help="Directory containing input raw datasets (e.g. `data/aloha_mobile_chair_raw` or `data/pusht_raw).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload to hub.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        help="When provided, only converts the provided episodes (e.g `--episodes 2 3 4`). Useful to test the code on 1 episode.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="When set to 1, removes provided output directory if it already exists. By default, raises a ValueError exception.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="When set to 1, resumes a previous run.",
    )
    parser.add_argument(
        "--tests-data-dir",
        type=Path,
        help=(
            "When provided, save tests artifacts into the given directory "
            "(e.g. `--tests-data-dir tests/data` will save to tests/data/{--repo-id})."
        ),
    )

    args = parser.parse_args()
    convert_data(**vars(args))


if __name__ == "__main__":
    main()
