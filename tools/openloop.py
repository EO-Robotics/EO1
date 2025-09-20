import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.normalize import Unnormalize
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

from eo.data.lerobot_dataset import LeRobotDataset
from eo.model.processing_eo1 import EO1VisionProcessor

argparser = argparse.ArgumentParser()
argparser.add_argument("--repo_id", type=str, default="libero_spatial_no_noops_1.0.0_lerobot", help="repo id")
argparser.add_argument("--root", type=str, default="./demo_data", help="root path")
argparser.add_argument(
    "--model_path",
    type=str,
    default="path/to/your/model",
    help="model path",
)
argparser.add_argument("--num_step", type=int, default=10, help="model path")
argparser.add_argument("--train_subtask", type=bool, default=False, help="model path")
argparser.add_argument("--delta_action", type=bool, default=False, help="delta action")
args = argparser.parse_args()

num_step = args.num_step

# load models and set keys
# processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

processor = EO1VisionProcessor.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, dtype=torch.bfloat16).eval().cuda()
action_horizon = processor.robot_config.get("action_chunk_size", 50)

select_video_keys = processor.robot_config["select_video_keys"][args.repo_id]
select_state_keys = processor.robot_config["select_state_keys"][args.repo_id]
select_action_keys = processor.robot_config["select_action_keys"][args.repo_id]
state_mode = processor.robot_config["state_mode"]

# load dataset
meta = LeRobotDatasetMetadata(args.repo_id, root=f"{args.root}/{args.repo_id}")
dataset = LeRobotDataset(
    args.repo_id,
    root=f"{args.root}/{args.repo_id}",
    delta_timestamps={
        k: [i / meta.fps for i in range(action_horizon)]
        for k in map(lambda x: x, select_action_keys)  # noqa: C417
    },
    state_mode=state_mode,
    train_subtask=args.train_subtask,
    select_action_keys=select_action_keys,
    delta_action=args.delta_action,
    effector_indices=[14, 15],
)

# helper functions
fn = lambda x: Image.fromarray((x.permute(1, 2, 0) * 255).numpy().astype(np.uint8))  # noqa: E731
unnormalizer = Unnormalize(dataset.normalizer.features, dataset.normalizer.norm_map, dataset.normalizer.stats)
actions = []
actions_data = []

for i in tqdm(range(num_step)):
    data = dataset[action_horizon * i]
    # model
    data = unnormalizer(data)
    batch = {
        **{k: [fn(data[k])] for k in select_video_keys},
        **{k: [data[k]] for k in select_state_keys},
        "task": [data["task"]],
        "repo_id": [args.repo_id],
    }
    selected_actions = processor.select_action(model, batch).action.squeeze(0).cpu().numpy()
    # raw
    actions_data += [
        torch.cat(
            [data[k].unsqueeze(-1) if data[k].ndim == 1 else data[k] for k in select_action_keys], dim=1
        ).numpy()
    ]

    if args.delta_action:
        selected_states = []
        for k in select_action_keys:
            state_key = k.replace("action", "observation.state")
            selected_states.append(
                data[state_key].unsqueeze(0) if data[state_key].ndim == 1 else data[state_key]
            )
        selected_states = torch.cat(selected_states, dim=1).numpy()

        accumulated_actions = np.cumsum(selected_actions, axis=0)
        exec_actions = selected_states + accumulated_actions
        exec_actions[..., -2:] = selected_actions[..., -2:]
    else:
        exec_actions = selected_actions

    actions += [exec_actions]


actions = np.concatenate(actions, axis=0)
actions_data = np.concatenate(actions_data, axis=0)

# plot actions
fig, axs = plt.subplots(actions.shape[-1], 1, figsize=(12 * 4, 3 * 4 * actions.shape[-1]))
for i in range(actions.shape[-1]):
    axs[i].plot(range(num_step * action_horizon), actions_data[:, i], color="tab:green")  # , linestyle="--")
    axs[i].plot(range(num_step * action_horizon), actions[:, i], color="tab:red")  # , linestyle=":")

fig.suptitle(f"{args.model_path}", fontsize=16)
fig.legend(
    labels=["Dataset Action", "Model Action"],
    loc="center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.05),
    frameon=False,
)
step = args.model_path.split("/")[-1].split("-")[-1]

# save visualization
plt.savefig(f"{args.model_path}/openloop_{step}.png", dpi=100, bbox_inches="tight")
print(f"save to {args.model_path}/openloop_{step}.png")
