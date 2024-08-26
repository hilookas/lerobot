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
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
import numpy

import fractions
import av
import av.container
import av.stream
import av.video.frame
import queue
import threading

logging.getLogger('libav').setLevel(logging.ERROR)
logging.getLogger().setLevel(5)


def load_from_videos(
    item: dict[str, torch.Tensor],
    video_frame_keys: list[str],
    videos_dir: Path,
    tolerance_s: float,
    backend: str = "pyav",
):
    """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
    in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a Segmentation Fault.
    This probably happens because a memory reference to the video loader is created in the main process and a
    subprocess fails to access it.
    """
    # since video path already contains "videos" (e.g. videos_dir="data/videos", path="videos/episode_0.mp4")
    data_dir = videos_dir.parent

    for key in video_frame_keys:
        if isinstance(item[key], list):
            # load multiple frames at once (expected when delta_timestamps is not None)
            timestamps = [frame["timestamp"] for frame in item[key]]
            paths = [frame["path"] for frame in item[key]]
            if len(set(paths)) > 1:
                raise NotImplementedError("All video paths are expected to be the same for now.")
            video_path = data_dir / paths[0]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames
        else:
            # load one frame
            timestamps = [item[key]["timestamp"]]
            video_path = data_dir / item[key]["path"]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames[0]

    return item


def decode_video_frames_torchvision(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()
        
        for stream in reader.container.streams:
            stream.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def get_video_encoder(
    video_path: Path, 
    fps: int = 30, 
    width: int = 1280, 
    height: int = 720,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    options: dict[str, str] | None = None,
    nice: int = 0,
):
    q = queue.Queue()
    
    def thread():
        nonlocal options

        # ffmpeg -f image2 -r 30 -i imgs_dir/frame_%06d.png -vcodec libsvtav1 -pix_fmt yuv420p -g 2 -crf 30 -loglevel error -y imgs_dir.mp4
        video_path.parent.mkdir(parents=True, exist_ok=True)

        container: av.container.OutputContainer = av.open(file=str(video_path), mode="w")
        
        if options is None:
            options = {
                # "g": str(2), # GOP will not work with libsvtav1
                "crf": str(30),
                # "preset": str(10),
            }
        
        stream: av.stream.Stream = container.add_stream(vcodec, rate=fps, options=options)
        stream.pix_fmt = pix_fmt

        stream.width = width
        stream.height = height

        VIDEO_PTIME = 1 / fps
        VIDEO_CLOCK_RATE = 90000

        timestamp = 0

        while True:
            if nice:
                time.sleep(nice)
            image = q.get()
            if image is None:
                break
            # print(timestamp)

            frame = av.video.VideoFrame.from_ndarray(image)
            
            frame.pts = timestamp
            frame.time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
            
            timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)

            for packet in stream.encode(frame):
                container.mux(packet)

            q.task_done()
            
        # Stop
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()
        for stream in container.streams:
            stream.close()
        container = None
        
        q.task_done()
    
    threading.Thread(target=thread, args=(), daemon=True).start()

    # usage: q.put(image)

    return q


def save_images_to_video(
    imgs_array: numpy.array, 
    video_path: Path, 
    fps: int = 30, 
    width: int = 1280, 
    height: int = 720,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    options: dict[str, str] | None = None,
    nice: int = 0,
):
    q = get_video_encoder(video_path, fps, width, height, vcodec, pix_fmt, options, nice)

    for img_array in imgs_array:
        q.put(img_array)
        
    q.put(None)
    q.join()


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")
