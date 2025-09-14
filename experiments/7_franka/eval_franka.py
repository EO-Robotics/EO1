import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import collections
import copy
import dataclasses
import os.path as osp
import time
from datetime import datetime
from pathlib import Path

import cv2
import deoxys.utils.transform_utils as dft
import imageio
import numpy as np
import torch
import tqdm
import tyro
from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from PIL import Image
from realsense_camera import MultiCamera
from transformers import AutoModel, AutoProcessor

# Add your camera serial numbers here
EGO_CAMERA = "213522070137"
THIRD_CAMERA = "243222074139"

reset_joint_positions = [
    0.0760389047913384,
    -1.0362613022620384,
    -0.054254247684777324,
    -2.383951857286591,
    -0.004505598470154735,
    1.3820559157131187,
    0.784935455988679,
]


def save_rollout_video(rollout_images, save_dir):
    """Saves an MP4 replay of an episode."""
    date_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    mp4_path = Path(save_dir) / f"{date_time}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model parameters
    #################################################################################################################
    resize_size: int = 224
    replan_steps: int = 5
    model_path: str = ""
    repo_id: str = ""
    task: str = ""

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Path = Path("experiments/7_franka/videos")  # Path to save videos
    max_timesteps: int = 300  # Number of timesteps to run


def convert_gripper_action(action):
    action[-1] = 1 - action[-1]
    if action[-1] < 0.5:
        action[-1] = -1

    return action


def get_robot_interface():
    robot_interface = FrankaInterface(osp.join(config_root, "charmander.yml"))
    controller_cfg = YamlConfig(osp.join(config_root, "osc-pose-controller.yml")).as_easydict()
    controller_type = "OSC_POSE"

    return robot_interface, controller_cfg, controller_type


def main(args: Args):
    multi_camera = MultiCamera()
    robot_interface, controller_cfg, controller_type = get_robot_interface()

    model = (
        AutoModel.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    while True:
        action_plan = collections.deque()
        input("Press Enter to start episode ...")
        reset_joints_to(robot_interface, reset_joint_positions)

        replay_images = []
        bar = tqdm.tqdm(
            range(args.max_timesteps),
            position=0,
            leave=True,
            ncols=80,
            desc="Rollout steps",
        )

        for _ in bar:
            try:
                images = multi_camera.get_frame()
                last_state = robot_interface._state_buffer[-1]
                last_gripper_state = robot_interface._gripper_state_buffer[-1]
                frame, _ = images[THIRD_CAMERA]
                ego_frame, _ = images[EGO_CAMERA]

                if not action_plan:
                    frame = copy.deepcopy(frame)
                    ego_frame = copy.deepcopy(ego_frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ego_frame = cv2.cvtColor(ego_frame, cv2.COLOR_BGR2RGB)
                    replay_images.append(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ego_frame_rgb = cv2.cvtColor(ego_frame, cv2.COLOR_BGR2RGB)
                    replay_images.append(frame_rgb.copy())

                    eef_pose = np.asarray(last_state.O_T_EE, dtype=np.float32).reshape(4, 4).T
                    eef_state = np.concatenate(
                        (
                            eef_pose[:3, -1],
                            dft.mat2euler(
                                eef_pose[:3, :-1],
                            ),
                        ),
                        axis=-1,
                    )
                    gripper_state = np.array([last_gripper_state.width])
                    state_data = np.concatenate([eef_state.flatten(), np.array([0]), gripper_state])

                    frame_resized = cv2.resize(frame_rgb, (args.resize_size, args.resize_size))
                    ego_frame_resized = cv2.resize(ego_frame_rgb, (args.resize_size, args.resize_size))

                    ego_frame = Image.fromarray(ego_frame_resized)
                    frame = Image.fromarray(frame_resized)

                    # NOTE: Change the keys to match your model
                    batch = {
                        "observation.images.image": [frame],
                        "observation.images.wrist_image": [ego_frame],
                        "observation.state": [state_data],
                        "task": [args.task],
                        "repo_id": [args.repo_id],
                    }
                    ov_out = processor.select_action(
                        model,
                        batch,
                    )
                    action_chunk = ov_out.action[0].numpy()
                    assert len(action_chunk) >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                pred_action_chunk = action_plan.popleft()
                action = pred_action_chunk
                rotation_matrix = dft.euler2mat(action[3:6])
                quat = dft.mat2quat(rotation_matrix)
                axis_angle = dft.quat2axisangle(quat)
                action[3:6] = axis_angle
                action = convert_gripper_action(action)

                robot_interface.control(
                    controller_type=controller_type, action=action, controller_cfg=controller_cfg
                )

            except KeyboardInterrupt:
                break

        # saving video
        video_save_path = args.video_out_path / args.task.replace(" ", "_")
        video_save_path.mkdir(parents=True, exist_ok=True)
        curr_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = video_save_path / f"{curr_time}.mp4"
        video = np.stack(replay_images)
        imageio.mimsave(save_path, video, fps=20)

        if input("Do one more eval (default y)? [y/n]").lower() == "n":
            break


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
