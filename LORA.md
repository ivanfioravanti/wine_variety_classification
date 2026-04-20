# Training Instructions

This document describes how to generate training data and train the wine variety classification model using MLX.

## Data Generation

The training data is generated from wine reviews using the `generate_data.py` script. The script pulls the [spawn99/wine-reviews](https://huggingface.co/datasets/spawn99/wine-reviews) dataset from Hugging Face and creates JSONL files for training, validation, and testing.

### Prerequisites
- Python with `datasets` and `pandas` installed
- No manual dataset download needed — data is fetched from Hugging Face automatically

### Generate Training Data

1. Basic usage (filters for specific country defined in config.py):
```bash
python -m train.generate_data
```

2. Include wines from all countries:
```bash
python -m train.generate_data --all-countries
```

The script will:
- Load and filter the wine dataset from Hugging Face
- Remove rare varieties (less than 5 examples)
- Create prompt-completion pairs for training
- Split data into train (80%), test (10%), and validation (10%) sets
- Save the data in `train/data/` as:
  - `train.jsonl`
  - `test.jsonl`
  - `valid.jsonl`

## Model Training

The project uses MLX-LM for fine-tuning models using LoRA (Low-Rank Adaptation), with live validation monitoring during training.

### Prerequisites
- MLX and MLX-LM installed
- Generated training data from the previous step

### Available Training Configs

The `train/` directory contains configs for several models:

| Config file | Model |
|---|---|
| `qwen_lora_config.yaml` | `mlx-community/Qwen3-0.6B-bf16` |
| `gemma_lora_config.yaml` | `mlx-community/gemma-4-e2b-it-bf16` |
| `phi_4_lora_config.yaml` | Phi-4 |
| `phi_lora_config.yaml` | Phi-3.5-mini-instruct |
| `llama3_3_lora_config.yaml` | Llama 3.3 |
| `mistral_lora_config_sample.yaml` | Mistral |

### Start Training

Run training with live validation monitoring using the config of your choice:

```bash
# Example: Qwen3 0.6B
python ./train/lora_training_monitor.py -c ./train/qwen_lora_config.yaml

# Example: Gemma 4 e2B
python ./train/lora_training_monitor.py -c ./train/gemma_lora_config.yaml
```

The monitor script wraps `mlx_lm.lora` and computes accuracy on validation batches during training.

The training process will:
- Initialize the model and apply LoRA adapters
- Save checkpoints periodically (configurable via `save_every`)
- Log training progress and validation metrics
- Save the final adapter weights in the `adapters` directory

### Training Configuration

Each YAML config file controls the following key parameters:

- `model`: Hugging Face model identifier
- `data`: Path to the training data directory
- `batch_size`: Minibatch size (e.g., 4)
- `learning_rate`: AdamW learning rate (e.g., 1e-5)
- `iters`: Number of training iterations (e.g., 4000)
- `max_seq_length`: Maximum sequence length (e.g., 256)
- `num_layers`: Number of layers to fine-tune (`-1` = all)
- `lora_parameters`: LoRA rank, scale, dropout, and target layers
- `optimizer_config`: AdamW settings (betas, eps, weight_decay)
- `lr_schedule`: Learning rate schedule (e.g., cosine decay with warmup)

Monitor the training progress through the logged metrics and validation results.
