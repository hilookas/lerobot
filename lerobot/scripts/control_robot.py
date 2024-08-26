"""
Examples of usage:

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py record_dataset \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay_episode \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record_dataset \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --run-compute-stats 1 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

- Train on this dataset with the ACT policy:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py run_policy \
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import json
import logging
import os
import platform
import queue
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import tqdm
from huggingface_hub import create_branch
from omegaconf import DictConfig
from termcolor import colored

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.common.datasets.video_utils import get_video_encoder
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, save_meta_data

########################################################################################
# Utilities
########################################################################################

def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def log_control_info(robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items += [f"ep:{episode_index}"]
    if frame_index is not None:
        log_items += [f"frame:{frame_index}"]

    def log_dt(shortname, dt_val_s):
        nonlocal log_items
        log_items += [f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"]

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)
    
    robot.log_control_info(log_dt)

    info_str = " ".join(log_items)
    if fps is not None:
        actual_fps = 1 / dt_s
        if actual_fps < fps - 1:
            info_str = colored(info_str, "yellow")
    logging.info(info_str)


def get_is_headless():
    if platform.system() == "Linux":
        display = os.environ.get("DISPLAY")
        if display is None or display == "":
            return True
    return False


########################################################################################
# Control modes
########################################################################################


def teleoperate(robot: Robot, fps: int | None = None, teleop_time_s: float | None = None, use_busy_wait = False):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    start_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - now
            if 1 / fps - dt_s > 0:
                if use_busy_wait:
                    busy_wait(1 / fps - dt_s)
                else:
                    time.sleep(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s, fps=fps)

        if teleop_time_s is not None and time.perf_counter() - start_time > teleop_time_s:
            break


@torch.no_grad()
def record_dataset(
    robot: Robot,
    fps: int | None = None,
    root="data",
    repo_id="lerobot/debug",
    warmup_time_s=0,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    num_image_writers=8,
    force_override=False,
    say_out = True,
    enable_keyboard = True,
    use_busy_wait = False
):
    # TODO(rcadene): Add option to record logs

    if not video:
        raise NotImplementedError()

    if not robot.is_connected:
        robot.connect()

    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    rec_info_path = episodes_dir / "data_recording_info.json"
    if rec_info_path.exists():
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        episode_index = rec_info["last_episode_index"] + 1
    else:
        episode_index = 0

    # Execute a few seconds without recording data, to give times
    # to the robot devices to connect and start synchronizing.
    timestamp = 0
    start_time = time.perf_counter()
    if warmup_time_s > 0:
        logging.info("Warming up (no data recording)")
        if say_out:
            os.system('say "Warmup" &')
    while timestamp < warmup_time_s:
        now = time.perf_counter()
        observation, action, done = robot.teleop_step(record_data=True)

        dt_s = time.perf_counter() - now
        if 1 / fps - dt_s > 0:
            if use_busy_wait:
                busy_wait(1 / fps - dt_s)
            else:
                time.sleep(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_time

    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    exit_early = False
    rerecord_episode = False
    stop_recording = False

    # Only import pynput if not in a headless environment
    if get_is_headless():
        logging.info("Headless environment detected. Keyboard input will not be available.")
        enable_keyboard = False
    
    if enable_keyboard:
        from pynput import keyboard

        def on_press(key):
            nonlocal exit_early, rerecord_episode, stop_recording
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Exiting loop...")
                    exit_early = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    rerecord_episode = True
                    exit_early = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    stop_recording = True
                    exit_early = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    # Wait if necessary
    if not episode_index == num_episodes and not reset_time_s >= 0:
        assert hasattr(robot, "wait_for_reset")
        robot.wait_for_reset()

    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    # Using only 4 worker threads to avoid blocking the main thread.
    # Start recording all episodes
    while episode_index < num_episodes:
        logging.info(f"Recording episode {episode_index}")
        if say_out:
            os.system(f'say "Recording episode {episode_index}" &')
        ep_dict = {}
        frame_index = 0
        timestamp = 0
        start_time = time.perf_counter()
        
        encoder_q: dict[queue.Queue] | None = None
        
        while timestamp < episode_time_s or episode_time_s < 0:
            now = time.perf_counter()
            observation, action, done = robot.teleop_step(record_data=True)
            
            if done:
                exit_early = True

            image_keys = [key for key in observation if "image" in key]
            not_image_keys = [key for key in observation if "image" not in key]
            
            if encoder_q is None:
                encoder_q = {}
                for key in image_keys:
                    fname = f"{key}_episode_{episode_index:06d}.mp4"
                    video_path = local_dir / "videos" / fname
                    if video_path.exists():
                        video_path.unlink()
                    encoder_q[key] = get_video_encoder(video_path, fps, observation[key].shape[2], observation[key].shape[1], options={
                        "crf": str(30),
                        "preset": str(12),
                    })
            
            for key in image_keys:
                encoder_q[key].put(observation[key].permute((1, 2, 0)).numpy()) # CHW to HWC

            for key in not_image_keys:
                if key not in ep_dict:
                    ep_dict[key] = []
                ep_dict[key].append(observation[key])

            for key in action:
                if key not in ep_dict:
                    ep_dict[key] = []
                ep_dict[key].append(action[key])

            frame_index += 1

            dt_s = time.perf_counter() - now
            
            if 1 / fps - dt_s > 0:
                if use_busy_wait:
                    busy_wait(1 / fps - dt_s)
                else:
                    time.sleep(1 / fps - dt_s)

            dt_s = time.perf_counter() - now
            log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_time

            if exit_early:
                exit_early = False
                break

        if not stop_recording:
            # Start resetting env while the executor are finishing
            logging.info("Reset the environment")
            if say_out:
                os.system('say "Reset the environment" &')

        timestamp = 0
        start_time = time.perf_counter()

        # During env reset we save the data and encode the videos
        num_frames = frame_index

        for key in image_keys:
            # Store the reference to the video frame, even tho the videos are not yet encoded
            ep_dict[key] = []
            for i in range(num_frames):
                ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / fps})

        for key in not_image_keys:
            ep_dict[key] = torch.stack(ep_dict[key])

        for key in action:
            ep_dict[key] = torch.stack(ep_dict[key])

        ep_dict["episode_index"] = torch.tensor([episode_index] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True
        ep_dict["next.done"] = done

        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        print("Saving episode dictionary...")
        torch.save(ep_dict, ep_path)
        
        if encoder_q:
            for key in encoder_q:
                encoder_q[key].put(None)
            for key in encoder_q:
                if encoder_q[key].qsize() > 0:
                    with tqdm.tqdm(total=encoder_q[key].qsize(), desc="Saving images") as pbar:
                        while encoder_q[key].qsize() > 0:
                            pbar.update((pbar.total - encoder_q[key].qsize()) - pbar.n)
                            time.sleep(0.1)
                encoder_q[key].join()

        rec_info = {
            "last_episode_index": episode_index,
        }
        with open(rec_info_path, "w") as f:
            json.dump(rec_info, f)

        is_last_episode = stop_recording or (episode_index == (num_episodes - 1))
        
        print(f"Done #{episode_index}")

        # Wait if necessary
        if not is_last_episode:
            if reset_time_s >= 0:
                with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
                    while timestamp < reset_time_s:
                        time.sleep(1)
                        timestamp = time.perf_counter() - start_time
                        pbar.update(1)
                        if exit_early:
                            exit_early = False
                            break
            else:
                assert hasattr(robot, "wait_for_reset")
                robot.wait_for_reset()

        # Skip updating episode index which forces re-recording episode
        if rerecord_episode:
            rerecord_episode = False
            continue

        episode_index += 1

        if is_last_episode:
            logging.info("Done recording")
            if say_out:
                os.system('say "Done recording" &')
            if enable_keyboard:
                listener.stop()
            break

    num_episodes = episode_index

    logging.info("Concatenating episodes")
    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = {'vcodec': 'libsvtav1', 'pix_fmt': 'yuv420p', 'crf': 30}

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    if run_compute_stats:
        logging.info("Computing dataset statistics")
        if say_out:
            os.system('say "Computing dataset statistics" &')
        stats = compute_stats(lerobot_dataset, max_num_samples=320)
        lerobot_dataset.stats = stats
    else:
        logging.info("Skipping computation of the dataset statistrics")
        stats = {}

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if push_to_hub:
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
        create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    logging.info("Exiting")
    if say_out:
        os.system('say "Exiting" &')

    return lerobot_dataset


def replay_episode(robot: Robot, episode: int, fps: int | None = None, root="data", repo_id="lerobot/debug", use_busy_wait = False):
    # TODO(rcadene): Add option to record logs
    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()

    if not robot.is_connected:
        robot.connect()

    for idx in range(from_idx, to_idx):
        now = time.perf_counter()

        action = items[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - now
        if 1 / fps - dt_s > 0:
            if use_busy_wait:
                busy_wait(1 / fps - dt_s)
            else:
                time.sleep(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s, fps=fps)


def run_policy(robot: Robot, policy: torch.nn.Module, hydra_cfg: DictConfig, run_time_s: float | None = None, use_busy_wait = False):
    # TODO(rcadene): Add option to record eval dataset and logs

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    fps = hydra_cfg.env.fps

    if not robot.is_connected:
        robot.connect()

    start_time = time.perf_counter()
    while True:
        now = time.perf_counter()

        observation = robot.capture_observation()

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and hydra_cfg.use_amp
            else nullcontext(),
        ):
            # add batch dimension to 1
            for name in observation:
                observation[name] = observation[name].unsqueeze(0)

            if device.type == "mps":
                for name in observation:
                    observation[name] = observation[name].to(device)

            action = policy.select_action(observation)

            # remove batch dimension
            action = action.squeeze(0)

        robot.send_action(action.to("cpu"))

        dt_s = time.perf_counter() - now
        if 1 / fps - dt_s > 0:
            if use_busy_wait:
                busy_wait(1 / fps - dt_s)
            else:
                time.sleep(1 / fps - dt_s)

        dt_s = time.perf_counter() - now
        log_control_info(robot, dt_s, fps=fps)

        if run_time_s is not None and time.perf_counter() - start_time > run_time_s:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot",
        type=str,
        default="koch",
        help="Name of the robot provided to the `make_robot(name)` factory function.",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )

    parser_record = subparsers.add_parser("record_dataset", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=2,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=10,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=5,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--num-image-writers",
        type=int,
        default=8,
        help="Number of threads writing the frames as png images on disk. Don't set too much as you might get unstable fps due to main thread being blocked.",
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "--disable-sayout",
        action='store_false', dest='say_out',
        help="Whether use `say` command to speak out current status or not.",
    )
    parser_record.add_argument(
        "--disable-keyboard",
        action='store_false', dest='enable_keyboard',
        help="Enable keyboard control",
    )

    parser_replay = subparsers.add_parser("replay_episode", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")

    parser_policy = subparsers.add_parser("run_policy", parents=[base_parser])
    parser_policy.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_policy.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_name = args.robot
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot"]

    robot = make_robot(robot_name)
    if control_mode == "teleoperate":
        teleoperate(robot, **kwargs)
    elif control_mode == "record_dataset":
        record_dataset(robot, **kwargs)
    elif control_mode == "replay_episode":
        replay_episode(robot, **kwargs)

    elif control_mode == "run_policy":
        pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)
        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", args.overrides)
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
        run_policy(robot, policy, hydra_cfg)
