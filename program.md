# Autonomous Hyperparameter Search — Wine Variety LoRA Training

You are an autonomous research agent running hyperparameter experiments for a wine variety classification model fine-tuned with MLX LoRA. Your goal is to find the configuration that maximizes classification accuracy.

## Setup

Before starting experiments, do the following once:

1. Ensure dependencies are installed with `uv sync` (this project uses `uv` and `.venv`).
2. Create the results log file if it does not exist: `results/hp_search.jsonl`
3. Read the baseline config: `train/gemma_lora_config.yaml`
4. Read past experiment results from `results/hp_search.jsonl` (if any exist)

## Config file

- **File to modify**: `train/gemma_lora_config.yaml`
- **Do NOT modify**: `prepare.py`, `data_utils.py`, `config.py`, or any provider files
- The agent modifies ONLY the YAML config before each run

## How to run an experiment

This project uses `uv` for dependency and environment management. All training commands **must** be run via `uv run` so the correct virtual environment (`.venv`) is used and `mlx_lm.lora` is found on `PATH`.

```bash
uv run python ./train/lora_training_monitor.py -c ./train/gemma_lora_config.yaml
```

The training will run and produce validation accuracy outputs. At the end of training, look for a line in stdout formatted like:

```
HPSEARCH_RESULT|accuracy=0.7200|val_loss=1.2345|best_accuracy=0.7200|best_iteration=200|config=./train/gemma_lora_config.yaml
```

Parse this line to extract the accuracy and val_loss values.

## How to log results

After each experiment completes, append ONE line to `results/hp_search.jsonl` with all the hyperparameters used and the results. Use this exact JSON format (single line, no pretty-printing):

```json
{"timestamp": "ISO8601", "accuracy": 0.72, "val_loss": 1.23, "model": "mlx-community/gemma-4-e2b-it-bf16", "learning_rate": 1e-5, "rank": 32, "batch_size": 4, "scale": 2, "num_layers": -1, "lora_keys": ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"], "iters": 200, "lr_schedule_warmup": 100, "lr_schedule_warmup_init": 1e-4, "lr_schedule_arguments": [4e-4, 400, 1e-4], "max_seq_length": 256, "notes": "baseline"}
```

## Experiment loop

Follow this loop for each experiment:

1. **Read history**: Check `results/hp_search.jsonl` for all past experiments. Identify the best accuracy so far.
2. **Pick a change**: Based on the history, choose ONE hyperparameter to modify (see search space below). If this is the first run, use the baseline config as-is.
3. **Edit the YAML**: Modify `train/gemma_lora_config.yaml` with your chosen parameter. Keep `iters: 200` for all experiments.
4. **Run training**: Execute the training command above. Wait for it to complete.
5. **Parse result**: Extract accuracy from the `HPSEARCH_RESULT` line in the output.
6. **Log result**: Append the JSON line to `results/hp_search.jsonl`.
7. **Decide next**: If accuracy improved, keep the config and explore nearby values. If worse, revert that parameter and try a different one.
8. **Repeat**: Go back to step 1.

## Search space

These are the hyperparameters you should explore and their valid ranges:

| Parameter | YAML key | Baseline | Values to try |
|-----------|----------|----------|---------------|
| Learning rate | `learning_rate` | 1e-5 | 5e-6, 1e-5, 5e-5, 1e-4, 2e-4 |
| LoRA rank | `lora_parameters.rank` | 32 | 4, 8, 16, 32, 64 |
| Batch size | `batch_size` | 4 | 2, 4, 8, 16 |
| LoRA scale | `lora_parameters.scale` | 2 | 1, 2, 4, 8 |
| Num layers | `num_layers` | -1 | 8, 16, 24, -1 |
| LoRA keys | `lora_parameters.keys` | all 7 | subsets (see below) |
| LR warmup steps | `lr_schedule.warmup` | 100 | 30, 50, 100, 150 |
| LR warmup init | `lr_schedule.warmup_init` | 1e-4 | 1e-5, 1e-4, 1e-3 |
| LR decay args | `lr_schedule.arguments` | [4e-4, 400, 1e-4] | vary decay endpoints |
| Max sequence length | `max_seq_length` | 256 | 128, 256, 512 |

### LoRA keys subsets to try

- **Full** (baseline): `["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"]`
- **Attention only**: `["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"]`
- **MLP only**: `["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"]`
- **QKV only**: `["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]`

## Strategy guidelines

- **One parameter at a time**: Change only ONE hyperparameter per experiment (unless doing a final combined run). This keeps results interpretable.
- **Start with high-impact params**: Sweep learning rate first, then rank, then batch size. These tend to have the largest effect.
- **Be methodical**: Complete a full sweep of one parameter before moving to the next.
- **If stuck**: If 3 consecutive experiments on the same parameter show no improvement, move to the next parameter.
- **Combine best params**: After sweeping all individual parameters, do a combined run with all the best values.
- **Bold changes**: If accuracy is stuck below 40%, try a bolder change (e.g., 10x learning rate or rank 64).
- **Never change iters**: Always keep `iters: 200` during the search phase.
- **Revert on failure**: If training crashes or produces NaN loss, revert to the last working config and try a different change.
- **Clean adapters**: Delete the `adapters/` directory before each new run to avoid loading stale weights: `rm -rf adapters/`

## Fixed parameters (do NOT change)

- `model`: `mlx-community/gemma-4-e2b-it-bf16`
- `data`: the training data path
- `seed`: 123
- `fine_tune_type`: lora
- `iters`: 200 (during search)
- `adapter_path`: `adapters`
- `save_every`: 100
- `grad_checkpoint`: false
- `dropout`: 0
- `optimizer_config`: keep the existing AdamW config

## Success metric

**Maximize accuracy** (higher is better). The `accuracy` field from `HPSEARCH_RESULT` is the primary metric. `val_loss` is secondary (lower is better).

## Final run

After the search phase completes (or you've done 20+ experiments), take the best configuration found and:

1. Update `train/gemma_lora_config.yaml` with the best params
2. Change `iters` back to `4000` for a full-length training run
3. Run the training one final time
4. Log this as the "final" run with `"notes": "final_full_run"` in the JSONL

## Quick reference

- Config file: `train/gemma_lora_config.yaml`
- Training command: `uv run python ./train/lora_training_monitor.py -c ./train/gemma_lora_config.yaml`
- Results log: `results/hp_search.jsonl`
- Output pattern: `HPSEARCH_RESULT|accuracy=...|val_loss=...|best_accuracy=...|best_iteration=...|config=...`
- Clean adapters: `rm -rf adapters/`
