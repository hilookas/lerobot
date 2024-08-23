import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional

from astra_controller.astra_controller import AstraController

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

########################################################################
# Astra robot arm
########################################################################


@dataclass
class AstraRobotConfig:
    """
    Example of usage:
    ```python
    AstraRobotConfig()
    ```
    """

    # Define all components of the robot
    astra_controller: AstraController = field(default=None)


class AstraRobot:
    """
    Example of highest frequency teleoperation without camera:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    astra_controller = AstraController(
        
    )
    robot = AstraRobot(astra_controller)

    # Connect motors buses and cameras if any (Required)
    robot.connect()

    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection without camera:
    ```python
    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = AstraRobot(astra_controller)
    robot.connect()
    while True:
        observation, action, done = robot.teleop_step(record_data=True)
    ```

    Example of highest frequency data collection with cameras:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the mackbookpro and the iphone (connected in USB to the macbookpro)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "macbookpro": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "iphone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }

    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = AstraRobot(astra_controller)
    robot.connect()
    while True:
        observation, action, done = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy (without running multiple policies in parallel to ensure highest frequency):
    ```python
    # Assumes leader and follower arms + cameras have been instantiated already (see previous example)
    robot = AstraRobot(astra_controller)
    robot.connect()
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: AstraRobotConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = AstraRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.astra_controller: AstraController = self.config.astra_controller
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "AstraRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.astra_controller:
            raise ValueError(
                "AstraRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        self.astra_controller.connect()

        self.is_connected = True
    
    def wait_for_reset(self):
        self.astra_controller.wait_for_reset()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AstraRobot is not connected. You need to run `robot.connect()`."
            )
        
        assert record_data, "Please use Astra Web Teleop"

        # Prepare to assign the positions of the leader to the follower
        now = time.perf_counter()
        action_arm_l, action_gripper_l, action_arm_r, action_gripper_r, action_base, action_eef_l, action_eef_r = self.astra_controller.read_leader_present_position()
        self.logs[f"read_leader_pos_dt_s"] = time.perf_counter() - now
        
        # Leader-follower process will be automatically handle in astra controller.
        # Reason for that is we want to deliver image from device camera to the operator as soon as possible.
        # Also, delay of arm is all over the place. Strictly do as aloha does may not be necessary.
        # TODO delay consideration
            
        obs_dict = self.capture_observation()

        action_dict = {}
        action_dict["action.arm_l"] = torch.from_numpy(np.array(action_arm_l))
        action_dict["action.gripper_l"] = torch.from_numpy(np.array(action_gripper_l))
        action_dict["action.arm_r"] = torch.from_numpy(np.array(action_arm_r))
        action_dict["action.gripper_r"] = torch.from_numpy(np.array(action_gripper_r))
        action_dict["action.base"] = torch.from_numpy(np.array(action_base))
        action_dict["action.eef_l"] = torch.from_numpy(np.array(action_eef_l))
        action_dict["action.eef_r"] = torch.from_numpy(np.array(action_eef_r))

        return obs_dict, action_dict, self.astra_controller.done

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AstraRobot is not connected. You need to run `robot.connect()`."
            )

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        now = time.perf_counter()
        state_arm_l, state_gripper_l, state_arm_r, state_gripper_r, state_base, state_odom = self.astra_controller.read_present_position()
        self.logs[f"read_follower_pos_dt_s"] = time.perf_counter() - now

        # Capture images from cameras
        now = time.perf_counter()
        images = self.astra_controller.read_cameras()
        self.logs[f"read_camera_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state.arm_l"] = torch.from_numpy(np.array(state_arm_l))
        obs_dict["observation.state.gripper_l"] = torch.from_numpy(np.array(state_gripper_l))
        obs_dict["observation.state.arm_r"] = torch.from_numpy(np.array(state_arm_r))
        obs_dict["observation.state.gripper_r"] = torch.from_numpy(np.array(state_gripper_r))
        obs_dict["observation.state.base"] = torch.from_numpy(np.array(state_base))
        obs_dict["observation.state.odom"] = torch.from_numpy(np.array(state_odom))
        # Convert to pytorch format: channel first and float32 in [0,1]
        for name in images:
            obs_dict[f"observation.images.{name}"] = torchvision.transforms.functional.pil_to_tensor(images[name]) # shape (3, 480, 640)

        return obs_dict

    def send_action(self, action: torch.Tensor):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AstraRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = len(self.astra_controller.motor_names)
        follower_goal_pos = action[from_idx:to_idx].numpy()
        from_idx = to_idx

        self.astra_controller.write_goal_position(follower_goal_pos.astype(np.int32))

    def log_control_info(self, log_dt):
        key = f"read_leader_pos_dt_s"
        if key in self.logs:
            log_dt("dtRlead", self.logs[key])

        key = f"write_follower_goal_pos_dt_s"
        if key in self.logs:
            log_dt("dtRfoll", self.logs[key])

        key = f"read_follower_pos_dt_s"
        if key in self.logs:
            log_dt("dtWfoll", self.logs[key])

        key = f"read_camera_dt_s"
        if key in self.logs:
            log_dt(f"dtRcam", self.logs[key])

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AstraRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.astra_controller.disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
