# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running Tests and Providers

```bash
# Run all providers and generate summary
python wine_all.py

# Generate chart from existing results only
python wine_all.py --generate-chart

# Generate chart from specific summary file
python wine_all.py --generate-chart --summary summary_20250105_095642.csv

# Run without generating individual CSV files
python wine_all.py --no-provider-csv

# Run individual providers
python -m providers.wine_ollama
python -m providers.wine_openai
python -m providers.wine_anthropic
python -m providers.wine_gemini
python -m providers.wine_gemini_genai
python -m providers.wine_gemini_openai
python -m providers.wine_deepseek
python -m providers.wine_openrouter
python -m providers.wine_lmstudio
python -m providers.wine_mlx_server_unstructured
python -m providers.wine_mlx_omni_server
python -m providers.wine_openai_unstructured

# Run OpenAI batching job
python -m providers.wine_openai_batching

# Run tests
python -m tests.test_openrouter
```

### Training and Data Generation

```bash
# Generate training data for specific country (default: Italy from config.py)
python -m train.generate_data

# Generate training data for all countries
python -m train.generate_data --all-countries

# Train model with MLX (requires configuration file)
mlx_lm.lora --config ./train/phi_lora_config_sample.yaml
mlx_lm.lora --config ./train/qwen_lora_config.yaml
mlx_lm.lora --config ./train/llama3_3_lora_config.yaml
mlx_lm.lora --config ./train/phi_4_lora_config.yaml

# Train with validation output monitoring (shows model predictions during training)
python train/lora_training_monitor.py --config ./train/qwen_lora_config.yaml
```

## Architecture

### Provider Pattern
Each LLM provider in `/providers/` implements a standardized interface:
- `run_provider(models)` function that returns (DataFrame, results_dict)
- Handles API authentication via environment variables
- Implements retry logic for rate limits
- Uses structured output (JSON) where supported
- Tracks per-row results and overall accuracy
- Prints individual row predictions showing actual vs predicted varieties

### Data Flow
1. `data_utils.py` loads wine data from CSV with configurable filtering (country, sample size)
2. Providers generate prompts asking to predict grape variety from wine description
3. Results are collected with actual vs predicted varieties
4. `wine_all.py` aggregates results from multiple providers
5. Results saved to `/results/` with timestamped CSVs and accuracy charts

### Key Configuration
- `config.py`: Central configuration for country selection (default: Italy), sample size (default: 200), random seed (default: 123)
- `.env`: API keys for various providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY)
- Provider-specific models defined within each provider module

### Structured Output
When adding new providers, prefer structured JSON output using:
- OpenAI: `response_format={"type": "json_schema"}` with schema definition
- Anthropic: Tool/function calling with schema
- Gemini: `response_mime_type="application/json"` with response_schema
- Pydantic models for validation where applicable

### Concurrent Processing
Most providers use `ThreadPoolExecutor` for parallel API calls. Maintain this pattern for consistent performance across providers. Default configuration uses 8 workers for parallel processing.

### Training Data Generation
The `train/generate_data.py` script:
- Filters wine dataset by country (configurable)
- Removes rare varieties (<5 examples)
- Creates prompt-completion pairs in JSONL format
- Splits data into train (80%), test (10%), validation (10%)
- Outputs to `train/data/` directory