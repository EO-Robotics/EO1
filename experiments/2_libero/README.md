

# Libero Benchmark Training and Evaluation

This directory contains the implementation for training and evaluating EO-1 on the Libero benchmark, a comprehensive suite of manipulation tasks designed to test robotic learning algorithms across different task categories.

## Overview

The Libero benchmark consists of four main task suites:
- **Libero Spatial**: Spatial reasoning tasks requiring understanding of 3D spatial relationships
- **Libero Object**: Object manipulation tasks focusing on object-specific behaviors
- **Libero Goal**: Goal-conditioned tasks with varying objectives
- **Libero 10/90**: Subsets with 10 and 90 tasks respectively for different evaluation scales

## Dataset Preparation

### 1. Download Libero Datasets

The Libero datasets are available on Hugging Face and can be downloaded using the Hugging Face CLI:

```bash
cd YOUR_PATH_TO_DATASET

# Install Hugging Face CLI if not already installed
pip install huggingface-cli

# Download all Libero datasets
datasets=(
    IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
    IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
    IPEC-COMMUNITY/libero_90_no_noops_lerobot
    IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
)

for dataset in ${datasets[@]};
do
  echo "Downloading ${dataset}..."
  huggingface-cli download \
  --repo-type dataset --resume-download --local-dir-use-symlinks False \
  ${dataset} \
  --local-dir ${dataset}
done
```

**Note**: The datasets are quite large (several GB each), so ensure you have sufficient disk space and a stable internet connection.

### 2. Configure Dataset Paths

Update the dataset configuration in `experiments/2_libero/data-libero.yaml`:

```yaml
mm_datasets: # leave empty

lerobot_datasets:
  
  - repo_id: libero_spatial_no_noops_1.0.0_lerobot
    root: HF_LEROBOT_HOME
  
  - repo_id: libero_90_no_noops_lerobot
    root: HF_LEROBOT_HOME
  
  - repo_id: libero_object_no_noops_1.0.0_lerobot
    root: HF_LEROBOT_HOME
  
  - repo_id: libero_10_no_noops_1.0.0_lerobot
    root: HF_LEROBOT_HOME
```

Make sure to set the `HF_LEROBOT_HOME` to your dataset directory.

## Training

### Training Configuration

The training script (`train.sh`) is configured with the following hyperparameters:
- **GPUs**: 8 GPUs for distributed training
- **Batch Size**: 64 per device (total effective batch size: 512)
- **Learning Rates**: 
  - backbone: 1e-4
  - merger: 1e-4  
  - vision tower: 2e-5
- **Epochs**: 50
- **Chunk Size**: 8 (for sequence processing)
- **Optimization**: AdamW with cosine learning rate scheduling
- **Precision**: BF16 with TF32 enabled

### Start Training

```bash
bash experiments/2_libero/train.sh
```

The training will:
- Use the Qwen2.5-VL-3B-Instruct vision-language model as the base
- Train on all four Libero datasets simultaneously
- Save checkpoints every 5000 steps
- Use gradient checkpointing and flash attention for memory efficiency
- Log training progress every 100 steps

## Evaluation

### Evaluation Configuration

The evaluation script supports testing on different task suites with configurable parameters:
- **Number of trials per task**: 50 (for statistical significance)
- **Replanning steps**: 8 (for action sequence planning)
- **Video recording**: Enabled for visualization
- **Random seed**: 7 (for reproducibility)

### Run Evaluation

```bash
# Set the task suite to evaluate
task_suite_name=libero_spatial  # Options: libero_spatial, libero_90, libero_object, libero_10

# Set the path to your trained checkpoint
ckpt_path=PATH_TO_CHECKPOINT

# Run evaluation
python experiments/2_libero/eval_libero.py \
  --args.pretrained-path ${ckpt_path} \
  --args.task-suite-name ${task_suite_name} \
  --args.job-name ${task_suite_name} \
  --args.num-trials-per-task 50 \
  --args.replan-steps 8
```

### Evaluation Output

The evaluation will:
- Generate detailed logs in `experiments/2_libero/logs/`
- Save evaluation videos in `experiments/2_libero/logs/videos/`
- Report success rates for each task and overall performance
- Provide action-level analysis and failure case insights

## Results

### Performance Comparison

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|----------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| **EO-1 @ 50 epochs (finetuned)** | **99.7** | **99.8** | **99.2** | **94.8** | **98.2** |

## File Structure

```
experiments/2_libero/
├── README.md                # This file
├── train.sh                 # Training script
├── eval_libero.py           # Evaluation script
├── data-libero.yaml         # Dataset configuration
└── logs/                    # Training and evaluation logs
    └── videos/              # Evaluation videos
```