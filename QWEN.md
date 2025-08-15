# Qwen Integration Guide

This document provides comprehensive information about Qwen model integration in the wine variety classification project, including both inference and fine-tuning capabilities.

> **Note:** This guide also covers **Groq integration** for Qwen models, providing cloud-based inference with high-performance Qwen models like `qwen-2.5-72b-instruct` and `qwen-qwq-32b`.

## Overview

The project supports Qwen models through multiple providers and offers dedicated fine-tuning support via MLX. Qwen models are particularly well-suited for this task due to their strong multilingual capabilities and excellent performance on classification tasks.

## Supported Qwen Models

### Inference Models

#### Groq Provider (`wine_groq.py`)
**Cloud-based Qwen models with ultra-fast inference:**
- **qwen-qwq-32b** - Latest Qwen reasoning model (32B parameters)
- **qwen-2.5-72b-instruct** - Large 72B parameter model for high accuracy
- **qwen-2.5-32b-instruct** - Balanced performance and speed
- **qwen-2.5-7b-instruct** - Fast inference for development

#### Ollama Provider (`wine_ollama.py`)
- **qwen2.5:72b-instruct** - Large 72B parameter model for high accuracy
- Performance: 72% accuracy on Italian wines, 75.6% on French wines

#### MLX Provider (`wine_mlx_server_unstructured.py`)
Full Qwen3 model suite with various quantization levels:
- **mlx-community/Qwen3-0.6B-8bit** & **mlx-community/Qwen3-0.6B-bf16**
- **mlx-community/Qwen3-1.7B-4bit** & **mlx-community/Qwen3-1.7B-8bit**
- **mlx-community/Qwen3-4B-4bit** & **mlx-community/Qwen3-4B-8bit**
- **mlx-community/Qwen3-8B-4bit** & **mlx-community/Qwen3-8B-8bit**
- **mlx-community/Qwen3-14B-4bit** & **mlx-community/Qwen3-14B-8bit**
- **mlx-community/Qwen3-30B-A3B-4bit** & **mlx-community/Qwen3-30B-A3B-8bit**
- **mlx-community/Qwen3-32B-4bit** & **mlx-community/Qwen3-32B-8bit**
- **mlx-community/Qwen3-235B-A22B-4bit** & **mlx-community/Qwen3-235B-A22B-8bit**

### Fine-tuning Models

#### Base Model for LoRA Training
- **mlx-community/Qwen3-0.6B-bf16** - Default base model for fine-tuning
- Configured in `train/qwen_lora_config.yaml`

## Configuration Files

### Fine-tuning Configuration
**File:** `train/qwen_lora_config.yaml`

```yaml
# Model Configuration
model: "mlx-community/Qwen3-0.6B-bf16"
data: "/Users/ifioravanti/projects/wine_variety_classification/train/data"

# Training Parameters
batch_size: 16
learning_rate: 1e-5
iters: 16000
max_seq_length: 256

# LoRA Configuration
lora_parameters:
  keys: ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"]
  rank: 64
  scale: 20
  dropout: 0

# Optimization
optimizer: "adafactor"
optimizer_config:
  adafactor:
    scale_parameter: true
    relative_step: false
    clip_threshold: 1.0
    decay_rate: -0.997

# Scheduling
lr_schedule:
  name: cosine_decay
  warmup: 50
  warmup_init: 5e-5
  arguments: [1e-4, 500, 5e-5]
```

## Usage Instructions

### 1. Running Qwen Models for Inference

#### Via Groq (Cloud-based, Fastest)
```bash
# Set your Groq API key in .env file
echo "GROQ_API_KEY=your_key_here" >> .env

# Run Groq provider with Qwen models
python -m providers.wine_groq

# Or run comprehensive testing
python wine_all.py  # Includes Groq Qwen models
```

#### Via Ollama (Local)
```bash
# Pull Qwen models
ollama pull qwen2.5:72b-instruct

# Run Ollama provider
python -m providers.wine_ollama
```

#### Via MLX (Local, Fine-tuned)
```bash
# Run MLX Qwen models (various sizes)
python -m providers.wine_mlx_server_unstructured
```

### 2. Fine-tuning Qwen Models

#### Step 1: Generate Training Data
```bash
# Generate training data from wine reviews
python -m train.generate_data

# Include all countries (optional)
python -m train.generate_data --all-countries
```

#### Step 2: Start Fine-tuning
```bash
# Basic training
mlx_lm.lora --config ./train/qwen_lora_config.yaml

# With monitoring (recommended)
python train/lora_training_monitor.py --config ./train/qwen_lora_config.yaml
```

#### Step 3: Monitor Training Progress
The training monitor provides real-time validation outputs:
- Shows decoded model predictions during training
- Tracks learning progress with sample wine predictions
- Compatible with Qwen models (tested with Qwen3-0.6B variants)

### 3. Using Fine-tuned Models

After training, use the fine-tuned adapter:

```bash
# The adapter weights are saved in the 'adapters' directory
# Use with MLX provider (automatically loads adapters)
python -m providers.wine_mlx_server_unstructured
```

## Performance Benchmarks

### Qwen Model Performance Comparison
| Provider | Model | Italian Wines | French Wines | Speed |
|----------|-------|---------------|--------------|--------|
| Groq | qwen-2.5-72b-instruct | TBD | TBD | Ultra-fast |
| Groq | qwen-2.5-32b-instruct | TBD | TBD | Fast |
| Ollama | qwen2.5:72b-instruct | 72.0% | 75.6% | Medium |

### Provider Comparison
| Provider | Type | Best For | Qwen Models Available |
|----------|------|----------|----------------------|
| **Groq** | Cloud | Speed & Accuracy | qwen-2.5-72b, qwen-2.5-32b, qwen-2.5-7b, qwen-qwq-32b |
| **Ollama** | Local | Privacy | qwen2.5:72b-instruct |
| **MLX** | Local | Fine-tuning | Full Qwen3 suite (0.6B-235B) |

## Troubleshooting

### Common Issues

#### Model Loading Errors
```
KeyError: Mistral3Config not found
```
**Solution:** Use Qwen models instead of Mistral for better compatibility.

#### Memory Issues During Training
- Reduce `batch_size` in config
- Enable `grad_checkpoint: true`
- Use quantized models (4-bit or 8-bit)

#### Adapter Loading Issues
- Ensure adapter directory exists: `adapters/adapters.safetensors`
- Check file permissions
- Verify model compatibility

### Performance Optimization

#### For Low-End Hardware
```yaml
# Modified config for resource-constrained systems
batch_size: 8
grad_checkpoint: true
max_seq_length: 128
model: "mlx-community/Qwen3-0.6B-4bit"
```

#### For High-Performance Training
```yaml
# Optimized config for powerful systems
batch_size: 32
learning_rate: 2e-5
iters: 20000
model: "mlx-community/Qwen3-4B-bf16"
```

## Integration Notes

### Provider Compatibility
- **Ollama**: Supports qwen2.5 series
- **MLX**: Full Qwen3 support with quantization
- **LM Studio**: Limited Qwen support via MLX integration
- **OpenRouter**: Available through various providers

### Data Format
Training data uses JSONL format:
```json
{"prompt": "Wine description and characteristics...", "completion": "{\"variety\": \"Chardonnay\"}"}
```

### Environment Variables
Required for API access:
```bash
# For Groq cloud models (required)
GROQ_API_KEY=your_groq_api_key_here

# Not required for local Qwen models (Ollama, MLX)
```

## Next Steps

1. **Experiment with different Qwen model sizes** to find optimal performance/resource balance
2. **Try different LoRA configurations** (rank, alpha values)
3. **Test on different wine regions** beyond Italian/French
4. **Compare fine-tuned vs. base model performance**
5. **Explore Qwen2.5 vs Qwen3 differences** in wine classification tasks

## Resources

- [Qwen Model Hub](https://huggingface.co/collections/Qwen/qwen-65f6e9a6d1e3eb9e6a3e2f8f)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [LoRA Training Guide](./LORA.md)
- [Training Data Generation](./train/generate_data.py)