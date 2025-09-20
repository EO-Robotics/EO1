# VLLM Evaluation on Vision-Language Benchmarks

This directory contains the implementation for evaluating EO-1 on multiple vision-language benchmarks using the VLMEvalKit framework.

## Overview

The evaluation covers three comprehensive benchmarks designed to test different aspects of vision-language understanding:

### Benchmarks

- **EO-Bench**: End-to-end evaluation benchmark for embodied AI tasks
- **ERQABench**: Embodied Reasoning and Question Answering benchmark
- **RoboVQA**: Robotic Visual Question Answering benchmark

These benchmarks test the model's ability to:

- Understand complex visual scenes and spatial relationships
- Answer questions about robotic manipulation tasks
- Reason about cause-and-effect relationships in embodied scenarios
- Process multi-modal inputs (images, videos, text) effectively

## Quick Start

```bash
# 1. Download model and datasets
bash experiments/8_vllmeval/download_all.sh

# 2. Install VLMEvalKit, TODO: pr to VLMEvalKit to add EO-1-3B model @ Xianqiang Gao
git clone https://github.com/DelinQu/VLMEvalKit
cd VLMEvalKit
pip install -e .

# 3. Configure evaluation
cp experiments/8_vllmeval/dataset-config.json VLMEvalKit/configs/

# 4. Run evaluation
bash experiments/8_vllmeval/run.sh
```

## Dataset and Model Preparation

### 1. Download EO-1-3B Model

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface-cli

# Download EO-1-3B model
huggingface-cli download --resume-download --local-dir-use-symlinks False \
  IPEC-COMMUNITY/EO-1-3B \
  --local-dir EO-1-3B
```

### 2. Download Benchmark Datasets

```bash
cd YOUR_PATH_TO_DATASET

# Download all benchmark datasets
datasets=(
  ERQABench
  RoboVQA
  EO-Bench
)

HF_DATASET_HOME=YOUR_PATH_TO_DATASET

for dataset in ${datasets[@]};
do
  echo "Downloading ${dataset}..."
  huggingface-cli download \
  --repo-type dataset --resume-download --local-dir-use-symlinks False \
  IPEC-COMMUNITY/${dataset} \
  --local-dir ${HF_DATASET_HOME}/${dataset}
done
```

**Note**: The datasets are quite large (several GB each), so ensure you have sufficient disk space and a stable internet connection.

## VLMEvalKit Installation

### Install VLMEvalKit

```bash
# Clone the VLMEvalKit repository
git clone https://github.com/DelinQu/VLMEvalKit
cd VLMEvalKit

# Install in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt

# Verify installation
python -c "import vlmeval; print('VLMEvalKit installed successfully')"
```

## Evaluation Configuration

### Model and Dataset Configuration

Create a configuration file (`dataset-config.json`) with the following settings:

```json
{
  "model": {
    "EO1-3B": {
      "class": "EO1VisionFlowMatchingChat",
      "min_pixels": 50176,
      "max_pixels": 100352,
      "use_custom_prompt": false,
      "model_path": "IPEC-COMMUNITY/EO-1-3B"
    }
  },
  "data": {
    "EOBench": {
      "class": "EOBench",
      "dataset": "EOBench",
      "data_file": "IPEC-COMMUNITY/EO-Bench/benchmark_v1.jsonl",
      "data_root": "IPEC-COMMUNITY/EO-Bench"
    },
    "ERQABench": {
      "class": "ERQABench",
      "dataset": "ERQABench",
      "data_root": "IPEC-COMMUNITY/ERQABench",
      "data_file": "IPEC-COMMUNITY/ERQABench/benchmark_v1.jsonl"
    },
    "RoboVQA": {
      "class": "RoboVQA",
      "dataset": "RoboVQA",
      "data_root": "IPEC-COMMUNITY/RoboVQA",
      "data_file": "IPEC-COMMUNITY/RoboVQA/benchmark_v1.jsonl",
      "fps": 1
    }
  }
}
```

### Configuration Parameters

- **min_pixels**: 50176 - Minimum image resolution for processing
- **max_pixels**: 100352 - Maximum image resolution for processing
- **use_custom_prompt**: false - Use default prompting strategy
- **fps**: 1 - Frame rate for video processing (RoboVQA only)

## Running Evaluation

### Basic Evaluation

```bash
cd VLMEvalKit

# Run evaluation on all configured benchmarks
bash scripts/run.sh
```

### Custom Evaluation

```bash
# Run evaluation on specific benchmarks
python vlmeval/run.py --model EO1-3B --data EOBench
python vlmeval/run.py --model EO1-3B --data ERQABench
python vlmeval/run.py --model EO1-3B --data RoboVQA
```

### Evaluation with Custom Settings

```bash
# Run with custom batch size and device settings
python vlmeval/run.py \
  --model EO1-3B \
  --data EOBench \
  --batch_size 8 \
  --device cuda:0
```

## Results and Analysis

### Expected Output

The evaluation will generate:

- **Detailed logs** for each benchmark
- **Per-question results** with model predictions
- **Overall accuracy scores** for each benchmark
- **Performance metrics** including response time and memory usage

### Results Directory Structure

```
VLMEvalKit/
├── results/
│   ├── EO1-3B/
│   │   ├── EOBench/
│   │   ├── ERQABench/
│   │   └── RoboVQA/
│   └── logs/
```
