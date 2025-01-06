# Training Instructions

This document describes how to generate training data and train the wine variety classification model using MLX.

## Data Generation

The training data is generated from wine reviews using the `generate_data.py` script. The script processes wine review data and creates JSONL files for training, validation, and testing.

### Prerequisites
- Wine review dataset (`data/winemag-data-130k-v2.csv`)
- Python with pandas installed

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
- Load and filter the wine dataset
- Remove rare varieties (less than 5 examples)
- Create prompt-completion pairs for training
- Split data into train (80%), test (10%), and validation (10%) sets
- Save the data in `train/data/` as:
  - `train.jsonl`
  - `test.jsonl`
  - `valid.jsonl`

## Model Training

The project uses MLX-LM for fine-tuning the Phi-3.5-mini model using LoRA (Low-Rank Adaptation).

### Prerequisites
- MLX and MLX-LM installed
- Generated training data from the previous step

### Start Training

1. Review and modify the configuration in `train/phi_lora_config_sample.yaml` if needed. Key parameters include:
   - `batch_size`: 8 (default)
   - `learning_rate`: 2e-5 (default)
   - `iters`: 1000 (default)
   - `max_seq_length`: 2048 (default)

2. Start the training:
```bash
mlx_lm.lora --config ./train/phi_lora_config_sample.yaml
```

The training process will:
- Initialize the Phi-3.5-mini model
- Apply LoRA fine-tuning
- Save checkpoints every 300 iterations (configurable)
- Log training progress and validation metrics
- Save the final adapter weights in the `adapters` directory

### Training Configuration

Key configuration options in `phi_lora_config_sample.yaml`:
- `model`: Uses microsoft/Phi-3.5-mini-instruct
- `data`: Points to the training data directory
- `lora_parameters`:
  - Applies to attention layers (qkv_proj and o_proj)
  - rank: 64
  - dropout: 0
- Learning rate schedule:
  - Cosine decay with warmup
  - Initial warmup steps: 100
  - Learning rate range: 1e-6 to 1e-4

Monitor the training progress through the logged metrics and validation results. 