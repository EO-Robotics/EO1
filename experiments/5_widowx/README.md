# WidowX 250s with EO-1

This directory contains the implementation for controlling WidowX 250s robots using the EO-1 model. The system enables real-time robot manipulation tasks through vision-language-action integration.

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**

- WidowX 250s robot arm
- RealSense D435 camera (or compatible RGB camera)
- Compute options:
  - Single GPU workstation (runs both ROS control and model inference)
  - OR: NUC + GPU workstation (NUC for arm control, workstation for model inference)

**Software Requirements:**

- Ubuntu 20.04+ with CUDA support
- Python 3.10+
- Docker (recommended for running the WidowX ROS control node on a workstation in single-machine mode)
- BridgeData WidowX controller stack properly configured

Notes on architecture:

- `Single-machine mode`: Run the WidowX ROS control node in Docker on the same GPU workstation used for EO-1 inference.
- `Dual-machine mode`: Use a NUC for robot control and a GPU workstation for model inference. For WidowX, the NUC does not require a real-time kernel in this setup.

### Installation

1. **Setup submodules:**

```bash
git submodule update --init --recursive experiments/5_widowx/bridge_data_robot
git submodule update --init --recursive experiments/5_widowx/edgeml
```

2. **Configure robot control system:**
   Follow the BridgeData WidowX controller setup in [bridge_data_robot](https://github.com/HaomingSong/bridge_data_robot?tab=readme-ov-file#setup) to configure your NUC/workstation for WidowX 250s control:

3. **Install dependencies on workstation**

```bash
# Create conda environment
conda create -n eo python=3.10
conda activate eo

# Install WidowX envs for workstation
pip install -e experiments/5_widowx/bridge_data_robot/widowx_envs
pip install -e experiments/5_widowx/edgeml

# Install additional requirements
pip install -r experiments/5_widowx/requirements.txt
```

**Note**: In dual-machine mode, ensure the workstation can reach the control host (robot IP/port) over the network. In single-machine mode, ensure Docker has access to USB and camera devices.

## 🤖 Running Robot Control

### Basic Usage

```bash
python experiments/5_widowx/eval_widowx.py \
    --model-path "path/to/your/model" \
    --repo-id libero_spatial_no_noops_1.0.0_lerobot \
    --default-instruction "Put the eggplant in the basket" \
    --robot-ip 10.6.8.122 \
    --robot-port 5556 \
    --max-timesteps 120
```

### Parameters

| Parameter               | Description                               | Default                          |
| ----------------------- | ----------------------------------------- | -------------------------------- |
| `--model-path`          | Path to the trained EO-1 model checkpoint | Required                         |
| `--repo-id`             | Dataset/repo ID for task specification    | Required                         |
| `--default-instruction` | Default natural language instruction      | "Put the eggplant in the basket" |
| `--roll-out-path`       | Directory to save rollouts/videos         | experiments/5_widowx/logs        |
| `--max-timesteps`       | Maximum number of control steps           | 120                              |
| `--im-size`             | Image size for model input                | 224                              |
| `--action-horizon`      | Receding-horizon (RHC) execution steps    | 2                                |
| `--blocking`            | Use blocking control for step execution   | False                            |
| `--robot-ip`            | Robot/control host IP                     | 10.6.8.122                       |
| `--robot-port`          | Robot/control host port                   | 5556                             |

### Camera Configuration

- Default color topic for RealSense D435 is `/D435/color/image_raw` (see `CAMERA_TOPICS` in `eval_widowx.py`).
- Mount and wire the D435 according to the hardware guide: [BridgeData V2 Hardware Setup](https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit?tab=t.0).
- If your camera topic differs, update `CAMERA_TOPICS` or the controller configuration accordingly.

## 🔒 Safety Considerations

- Always ensure proper workspace setup and clear the workspace before operation.
- Monitor robot movements and be ready to use the emergency stop.
- Verify camera positioning and exposure for optimal visual coverage.

## 📝 Notes

- This setup uses a single external D435 stream by default; wrist camera is optional.
- Model performance depends on lighting, viewpoint, and calibration quality.
- Regular calibration of the robot and camera(s) is recommended.
- Rollouts and videos are saved under `--roll-out-path`.
