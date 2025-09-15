# Franka Emika Panda Robot Control with EO-1

This directory contains the implementation for controlling Franka Emika Panda robots using the EO-1 model. The system enables real-time robot manipulation tasks through vision-language-action integration.

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**

- Franka Emika Panda robot arm
- RealSense cameras (or compatible RGB cameras)
- **NUC**: Configured with real-time kernel for robot control
- **Workstation**: Equipped with GPU for model inference

**Software Requirements:**

- Ubuntu 20.04+ with CUDA support
- Python 3.10+
- Real-time kernel configuration on NUC
- Deoxys control system properly configured

### Installation

1. **Setup submodules:**

```bash
git submodule update --init --recursive experiments/7_franka/deoxys_control
```

2. **Configure robot control system:**
   Follow the [Deoxys Documentation](https://zhuyifengzju.github.io/deoxys_docs/html/index.html) to configure your NUC and workstation for Franka control.

3. **Install dependencies on workstation**

```bash
# Create conda environment
conda create -n eo python=3.10
conda activate eo

# Install deoxys for workstation
pip install -e experiments/7_franka/deoxys_control/deoxys

# Install additional requirements
pip install -r experiments/7_franka/requirements.txt
```

**Note**: The NUC handles real-time robot control while the workstation runs the EO-1 model inference. Both systems must be properly configured according to the Deoxys documentation.

## 🤖 Running Robot Control

### Basic Usage

```bash
python experiments/7_franka/eval_franka.py \
    --model-path "path/to/your/model" \
    --repo-id libero_spatial_no_noops_1.0.0_lerobot \
    --task "Pick and place a cube" \
    --video-out-path experiments/7_franka/videos \
    --max-timesteps 300
```

### Parameters

| Parameter          | Description                                  | Default                                 |
| ------------------ | -------------------------------------------- | --------------------------------------- |
| `--model-path`     | Path to the trained EO-1 model checkpoint    | Required                                |
| `--repo-id`        | Dataset repository ID for task specification | `libero_spatial_no_noops_1.0.0_lerobot` |
| `--task`           | Natural language description of the task     | `"Pick and place a cube"`               |
| `--video-out-path` | Directory to save recorded videos            | `experiments/7_franka/videos`           |
| `--max-timesteps`  | Maximum number of control steps              | `300`                                   |
| `--resize-size`    | Image resize dimensions for model input      | `224`                                   |
| `--replan-steps`   | RHC control horizon                          | `5`                                     |

### Camera Configuration

The system supports multiple camera inputs. Update the camera serial numbers in `eval_franka.py`:

```python
# Camera serial numbers (update these with your actual camera IDs)
EGO_CAMERA = "213522070137"  # Wrist camera
THIRD_CAMERA = "243222074139"  # External camera
```

## 🔒 Safety Considerations

- Always ensure proper workspace setup before operation
- Monitor robot movements and be ready to use emergency stop
- Verify camera positioning for optimal visual coverage

## 📝 Notes

- The system requires both wrist and external cameras for optimal performance
- Model performance depends on lighting conditions and camera positioning
- Regular calibration of the robot and cameras is recommended
- Check the video output directory for recorded demonstrations
