import argparse

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_path",
    type=str,
    default="outputs/your_path",
    help="Path to the pretrained model",
)
argparser.add_argument(
    "--repo_id",
    type=str,
    default="libero_spatial_no_noops_1.0.0_lerobot",
    help="Class name of the model",
)
args = argparser.parse_args()


def eval_policy():
    # set the observation (image, state, etc.)
    image0 = "demo_data/example.png"
    image1 = Image.open("demo_data/example.png")

    model = AutoModel.from_pretrained(args.model_path, dtype=torch.bfloat16).eval().cuda()

    processor = AutoProcessor.from_pretrained(args.model_path)

    batch = {
        "observation.images.image": [image0],
        "observation.images.wrist_image": [image1],
        "observation.state": [torch.rand(8)],
        "task": ["put the object in the box."],
        "repo_id": [args.repo_id],
    }
    ov_output = processor.select_action(
        model,
        batch,
    )
    print(ov_output)


if __name__ == "__main__":
    eval_policy()
