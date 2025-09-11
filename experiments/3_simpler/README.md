# SimplerEnv Benchmark Training and Evaluation

This directory contains the implementation for training and evaluating EO-1 on the SimplerEnv benchmark, a comprehensive suite of robotic manipulation tasks designed to test real-world deployment capabilities across different robot platforms and task complexities.

## Quick Start

```bash
# 1. Download datasets
bash experiments/3_simpler/download_datasets.sh

# 2. Train on Bridge dataset
bash experiments/3_simpler/train_bridge.sh

# 3. Train on Fractal dataset  
bash experiments/3_simpler/train_fractal.sh

# 4. Setup evaluation environment
git clone https://github.com/DelinQu/SimplerEnv-OpenVLA
cd SimplerEnv-OpenVLA
# Follow installation instructions

# 5. Run evaluation
bash scripts/run_eo.sh
```

For detailed instructions, see the sections below.

## Overview

The SimplerEnv benchmark consists of two main datasets and multiple evaluation scenarios:

### Datasets
- **Bridge Dataset**: Real-world manipulation data collected from various robot platforms
- **Fractal Dataset**: Large-scale synthetic manipulation data for robust training

### Evaluation Benchmarks

#### WidowX Benchmark (4 tasks)
- **Put Spoon on Towel**: Place a spoon on a specific towel location
- **Put Carrot on Plate**: Position a carrot on a designated plate
- **Stack Blocks**: Stack multiple blocks in a specific configuration
- **Put Eggplant in Basket**: Place an eggplant inside a basket

#### Google Robot Benchmark (4 tasks, 2 evaluation modes)
- **Pick Coke Can**: Grasp and lift a coke can from the table
- **Move Near**: Navigate the robot arm to a specific position
- **Open⋅Close Drawer**: Open and close a drawer mechanism
- **Drawer Apple**: Place an apple inside a drawer

**Evaluation Modes:**
- **Matching Mode**: Test on similar visual conditions as training data
- **Aggregation Mode**: Test on diverse visual conditions and object variations

The benchmark tests the model's ability to:
- Generalize across different robot platforms (WidowX vs Google Robot)
- Handle real-world visual variations and lighting conditions
- Perform precise manipulation tasks with varying complexity
- Adapt to different action spaces and control paradigms
- Maintain performance across different evaluation scenarios

## Dataset Preparation

### 1. Download SimplerEnv Datasets

The SimplerEnv datasets are available on Hugging Face and can be downloaded using the Hugging Face CLI:

```bash
cd YOUR_PATH_TO_DATASET

# Install Hugging Face CLI if not already installed
pip install huggingface-cli

# Download all SimplerEnv datasets
datasets=(
  IPEC-COMMUNITY/fractal20220817_data_lerobot
  IPEC-COMMUNITY/bridge_orig_lerobot
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

Update the dataset configuration files:

**For Bridge dataset** (`experiments/3_simpler/data-bridge.yaml`):
```yaml
mm_datasets:

lerobot_datasets:
  - repo_id: bridge_orig_lerobot
    root: HF_LEROBOT_HOME
    select_video_keys: [observation.images.image_0]
```

**For Fractal dataset** (`experiments/3_simpler/data-fractal.yaml`):
```yaml
mm_datasets:

lerobot_datasets:
  - repo_id: fractal20220817_data_lerobot
    root: HF_LEROBOT_HOME
    select_video_keys: [observation.images.image]
```

Make sure to set the `HF_LEROBOT_HOME` environment variable to point to your dataset directory.

## Training

### Training Configuration

The training is performed separately for each dataset with optimized hyperparameters:

#### Bridge Dataset Training
- **GPUs**: 8 GPUs (A100 80GB) for distributed training
- **Batch Size**: 128 per device (total effective batch size: 1024)
- **Epochs**: 20
- **Chunk Size**: 4 (for sequence processing)

#### Fractal Dataset Training
- **GPUs**: 8 GPUs (A100 80GB) for distributed training
- **Batch Size**: 256 per device (total effective batch size: 2048)
- **Epochs**: 10
- **Chunk Size**: 4 (for sequence processing)

#### Common Hyperparameters
- **Learning Rates**: 
  - Backbone: 1e-4
  - Merger: 1e-4  
  - Vision tower: 2e-5
- **Optimization**: AdamW with cosine learning rate scheduling
- **Precision**: BF16 with TF32 enabled
- **Weight Decay**: 0.1
- **Warmup Ratio**: 0.03

### Start Training

Train on each dataset separately:

```bash
# Train on Bridge dataset
bash experiments/3_simpler/train_bridge.sh

# Train on Fractal dataset  
bash experiments/3_simpler/train_fractal.sh
```

The training will:
- Use the Qwen2.5-VL-3B-Instruct vision-language model as the base
- Train on each dataset independently with optimized settings
- Save checkpoints every 5000 steps
- Use gradient checkpointing and flash attention for memory efficiency
- Log training progress every 100 steps

## Evaluation

### Evaluation Setup

The evaluation requires setting up the SimplerEnv environment:

**Install SimplerEnv Dependencies**:
```bash
# Clone and setup SimplerEnv-OpenVLA
git clone https://github.com/DelinQu/SimplerEnv-OpenVLA
cd SimplerEnv-OpenVLA
# Follow the installation instructions in the repository
```

~~**Copy EO-1 Model Files**:~~ NOTE: SimplerEnv-OpenVLA have already supported the EO-1 model implementation.
```bash
cp -r experiments/3_simpler/simpler_env/eo SimplerEnv-OpenVLA/simpler_env/policies/eo
cp experiments/3_simpler/simpler_env/eval_simpler.sh SimplerEnv-OpenVLA/scripts/run_eo.sh
```

### Evaluation Configuration

The evaluation script supports multiple task suites with configurable parameters:
- **Action Ensemble Temperature**: 4 (for action sampling diversity)
- **Number of Trials**: Variable per task (typically 10-50)
- **Video Recording**: Enabled for visualization and analysis
- **Environment**: Headless rendering for distributed evaluation

### Run Evaluation

```bash
# Update checkpoint path in eval_simpler.sh
ckpt_path=YOUR_CHECKPOINT_PATH

# Run evaluation on all task suites
bash SimplerEnv-OpenVLA/scripts/run_eo.sh
```

The evaluation will:
- Test on 9 different task configurations
- Generate detailed logs and success metrics
- Save evaluation videos for analysis
- Calculate overall performance statistics

## Results

### Performance Comparison

#### WidowX Benchmark Results
| Task | Put Spoon on Towel | Put Carrot on Plate | Stack Blocks | Put Eggplant in Basket | Overall |
|------|-------------------|-------------------|--------------|----------------------|---------|
| $\pi_{0}$ | 83.8 | 52.5 | 52.5 | 87.9 | 69.2 |
| **EO-1** | **63.6** | **54.5** | **81.8** | **90.9** | **72.7** |

#### Google Robot Benchmark - Matching Mode
| Task | Pick Coke Can | Move Near | Open⋅Close Drawer | Drawer Apple | Overall |
|------|---------------|-----------|-------------------|--------------|---------|
| $\pi_{0}$ | 97.9 | 78.7 | 62.25 | 46.6 | 71.4 |
| **EO-1** | **98.0** | **83.8** | **71.3** | **52.8** | **76.5** |

#### Google Robot Benchmark - Aggregation Mode
| Task | Pick Coke Can | Move Near | Open\Close Drawer | Drawer Apple | Average |
|------|---------------|-----------|-------------------|--------------|---------|
| $\pi_{0}$ | 90.1 | 80.7 | 27.6 | 20.5 | 54.7 |
| **EO-1** | **91.6** | **81.7** | **55.0** | **23.8** | **63.0** |

### Key Achievements

- **Strong Generalization**: EO-1 achieves competitive performance across different robot platforms (WidowX and Google Robot)
- **Robust Manipulation**: Consistent performance on complex manipulation tasks like stacking and drawer operations
- **Real-world Adaptation**: Effective handling of visual variations and real-world conditions
- **Multi-modal Understanding**: Superior performance on tasks requiring spatial reasoning and object manipulation

### Analysis

The results demonstrate EO-1's capabilities in:
1. **Cross-platform Generalization**: Strong performance on both WidowX and Google Robot platforms
2. **Complex Manipulation**: Excellent results on challenging tasks like stacking blocks (81.8%) and drawer operations
3. **Visual Robustness**: Consistent performance across different visual conditions and object appearances
4. **Action Precision**: Effective handling of fine-grained manipulation tasks requiring precise control

## File Structure

```
experiments/3_simpler/
├── README.md                    # This file
├── train_bridge.sh             # Bridge dataset training script
├── train_fractal.sh            # Fractal dataset training script
├── data-bridge.yaml            # Bridge dataset configuration
├── data-fractal.yaml           # Fractal dataset configuration
└── simpler_env/                # Evaluation environment
    ├── eo/                     # EO-1 model implementation
    │   ├── eo_model.py         # Main model class
    │   └── geometry.py          # Geometry utilities
    ├── eval_simpler.sh         # Evaluation script
    └── main_inference.py       # Inference interface
```