"""
This script shows how we evaluated a finetuned EO-1 on a real WidowX robot, which is adapted from https://github.com/octo-models/octo/blob/main/examples/04_eval_finetuned_on_robot.py.
While the exact specifics may not be applicable to your use case, this script serves as a didactic example of how to use EO-1 in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/HaomingSong/bridge_data_robot.git)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import dataclasses
import pathlib
import time
from datetime import datetime

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import tqdm
import tyro
from PIL import Image
from transformers import AutoModel, AutoProcessor
from widowx_env import RHCWrapper, WidowXGym
from widowx_envs.widowx_env_service import WidowXConfigs


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model parameters
    #################################################################################################################
    im_size: int = 224
    action_horizon: int = 2
    model_path: str = ""
    repo_id: str = ""

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    robot_ip: str = "10.6.8.122"  # IP address of the robot
    robot_port: int = 5556  # Port of the robot
    initial_eep: tuple[float, float, float] = (0.3, 0.0, 0.25)  # Initial position
    # initial_eep: tuple[float, float, float] = (0.15, 0.0, 0.1)  # Initial position
    blocking: bool = False  # Use the blocking controller
    max_timesteps: int = 120  # Number of timesteps to run
    default_instruction: str = "Put the eggplant in the basket"  # Default instruction

    #################################################################################################################
    # Utils
    #################################################################################################################
    show_image: bool = False  # Show image
    roll_out_path: pathlib.Path = pathlib.Path("experiments/5_widowx/logs")  # Path to save videos


##############################################################################
STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/D435/color/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def eval_bridge(args: Args) -> None:
    curr_time = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    base_save_path = args.roll_out_path / pathlib.Path(args.default_instruction.replace(" ", "_")) / curr_time

    # set up the widowx client
    start_state = np.concatenate([args.initial_eep, (0, 0, 0, 1)])
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["start_state"] = list(start_state)

    env = WidowXGym(
        env_params,
        host=args.robot_ip,
        port=args.robot_port,
        im_size=args.im_size,
        blocking=args.blocking,
        sticky_gripper_num_steps=STICKY_GRIPPER_NUM_STEPS,
    )
    if not args.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE
    results_df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    model = (
        AutoModel.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    env = RHCWrapper(env, args.action_horizon)

    while True:
        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        if input(f"Use default instruction: {args.default_instruction}? (default y) [y/n]").lower() == "n":
            instruction = input("Enter instruction: ")
        else:
            instruction = args.default_instruction

        # do rollout
        images = []
        images.append(obs["full_image"])
        last_tstep = time.time()
        bar = tqdm.tqdm(
            range(args.max_timesteps),
            position=0,
            leave=True,
            ncols=80,
            desc="Rollout steps",
        )

        for t_step in bar:
            try:
                bar.set_description(f"Step {t_step}/{args.max_timesteps}")
                if args.show_image:
                    cv2.imshow("img_view", obs["full_image"])
                    cv2.waitKey(1)

                # prepare observation
                # image = torch.from_numpy(obs["image_primary"] / 255).permute(2, 0, 1)
                # [::-1, ::-1]
                image = cv2.resize(obs["full_image"], (256, 256), interpolation=cv2.INTER_LINEAR)
                # image = np.ascontiguousarray(obs["image_primary"])

                # print("image",image.shape)
                img = Image.fromarray(image)
                batch = {
                    "observation.images.image": [img],
                    "observation.images.wrist_image": [img],
                    "observation.state": [obs["proprio"]],
                    "task": [str(instruction)],
                    "repo_id": [args.repo_id],
                }
                ov_out = processor.select_action(model, batch)
                action_chunk = ov_out.action.squeeze(0).numpy()

                assert len(action_chunk) >= args.action_horizon, (
                    f"We want to replan every {args.action_horizon} steps, but policy only predicts {len(action_chunk)} steps."
                )

                # perform environment step
                obs, _, _, truncated, infos = env.step(action_chunk)

                # recording history images
                for history_obs in infos["observations"]:
                    image = history_obs["full_image"]
                    images.append(image)
                if truncated:
                    break

                # match the step duration
                elapsed_time = time.time() - last_tstep
                if elapsed_time < STEP_DURATION:
                    time.sleep(STEP_DURATION - elapsed_time)

            except KeyboardInterrupt:
                break
            time.sleep(0.2)

        # logging rollouts
        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%, a float value 0-1, or a numeric value 0-100 based on the evaluation spec)"
            )
            try:
                if success == "y":
                    success = 1.0
                elif success == "n":
                    success = 0.0
                else:
                    success = float(success)
            except Exception:
                success = 0.0

            video_save_path = (
                base_save_path
                / "videos"
                / f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_success_{success:.2f}.mp4"
            )

            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 1] but got: {success}")

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            {
                                "instruction": instruction,
                                "success": success,
                                "duration": t_step,
                                "video_filename": video_save_path,
                                "model_path": args.model_path,
                                "repo_id": args.repo_id,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        # saving video
        video = np.stack(images)
        video_save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_save_path, video, fps=1.0 / STEP_DURATION * 3)

        if (
            input(f"Already eval {len(results_df)} rollouts. Do one more eval (default y)? [y/n]").lower()
            == "n"
        ):
            break

    # save results
    csv_filename = base_save_path / "results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    # print avg
    print(f"Avg success: {results_df['success'].mean()}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    args: Args = tyro.cli(Args)
    eval_bridge(args)
