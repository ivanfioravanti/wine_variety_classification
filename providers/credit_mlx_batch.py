"""Batch inference provider using MLX for credit default classification."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from credit_data_utils import (
    DEFAULT_CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    LABEL_COLUMN,
    PROMPT_COLUMN,
    extract_default_label,
    load_credit_data,
    load_jsonl_prompts,
    split_credit_data,
)

try:
    from mlx_lm import batch_generate, load
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "mlx_lm is required for the MLX batch provider. Install it via `pip install mlx-lm`."
    ) from exc


DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-bf16"
SYSTEM_PROMPT = (
    "You are a careful credit risk analyst. Use only the origination-time loan and "
    'credit profile information provided. Respond strictly in JSON as {"default": "yes"} '
    'or {"default": "no"}.'
)
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 16
DEFAULT_JSONL_PATH = DEFAULT_OUTPUT_DIR / "test.jsonl"


np.random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DEFAULT_ADAPTER_SUFFIXES = (".safetensors", ".bin", ".json", ".pt")
DEBUG_PREFIX = "[credit_mlx_batch]"


@dataclass
class BatchRunResult:
    dataframe: pd.DataFrame
    accuracy: float
    predictions: List[Optional[str]]
    tp: int
    tn: int
    fp: int
    fn: int


def _chunk(iterable: Sequence, size: int) -> Iterable[Tuple[int, Sequence]]:
    for start in range(0, len(iterable), size):
        yield start, iterable[start : start + size]


def _encode_prompts(
    prompts: Sequence[str],
    tokenizer,
    system_prompt: Optional[str],
    add_generation_prompt: bool = True,
) -> List[List[int]]:
    encoded: List[List[int]] = []
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError(
                f"Expected prompts to be strings, received {type(prompt)!r} instead."
            )

        if has_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        else:
            combined = prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"
            input_ids = tokenizer.encode(combined, add_special_tokens=True)

        if not isinstance(input_ids, list):
            raise ValueError("Tokenizer.encode must return a list of token ids.")
        encoded.append(input_ids)

    return encoded


def _resolve_adapter_path(
    adapter: Optional[str],
    model_name: str,
) -> Optional[Path]:
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.exists() and DEFAULT_ADAPTERS_DIR.exists():
            adapter_path = DEFAULT_ADAPTERS_DIR / adapter
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path '{adapter}' could not be found.")
        print(f"{DEBUG_PREFIX} Using adapter provided via CLI at {adapter_path}")
        return adapter_path

    if not DEFAULT_ADAPTERS_DIR.exists():
        print(f"{DEBUG_PREFIX} Adapters directory {DEFAULT_ADAPTERS_DIR} does not exist.")
        return None

    default_files = list(DEFAULT_ADAPTERS_DIR.glob("*"))
    has_root_weights = any(
        entry.is_file() and entry.suffix in DEFAULT_ADAPTER_SUFFIXES
        for entry in default_files
    )
    has_config = any(
        entry.is_file() and entry.name in {"adapter_config.json", "adapter_config.yaml"}
        for entry in default_files
    )
    if has_root_weights and has_config:
        print(
            f"{DEBUG_PREFIX} Found adapter files directly under {DEFAULT_ADAPTERS_DIR}, auto-loading them."
        )
        return DEFAULT_ADAPTERS_DIR

    model_suffix = model_name.split("/")[-1]
    candidates: List[Path] = []
    for entry in DEFAULT_ADAPTERS_DIR.iterdir():
        if not entry.is_dir():
            continue
        contents = list(entry.iterdir())
        has_weights = any(
            child.suffix in DEFAULT_ADAPTER_SUFFIXES and child.is_file()
            for child in contents
        )
        if not has_weights:
            continue
        if entry.name == model_name or entry.name == model_suffix:
            candidates.append(entry)
        elif model_suffix in entry.name:
            candidates.append(entry)

    if len(candidates) == 1:
        print(
            f"{DEBUG_PREFIX} Auto-selected adapter directory {candidates[0]} for model {model_name}."
        )
        return candidates[0]

    if len(candidates) > 1:
        print(
            f"{DEBUG_PREFIX} Multiple adapter candidates found for {model_name}: {candidates}. "
            "Please specify one using --adapter."
        )
        return None

    print(
        f"{DEBUG_PREFIX} No adapter directory matched model {model_name} inside {DEFAULT_ADAPTERS_DIR}."
    )
    return None


def _load_eval_dataframe_from_csv(
    csv_path: str | Path,
    evaluation_split: str,
    max_rows: Optional[int],
    limit: Optional[int],
) -> pd.DataFrame:
    df = load_credit_data(csv_path=csv_path, max_rows=max_rows, random_state=RANDOM_SEED)
    train_df, valid_df, test_df = split_credit_data(df, random_state=RANDOM_SEED)

    split_map = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
        "all": df,
    }
    eval_df = split_map[evaluation_split].reset_index(drop=True)
    if limit is not None:
        eval_df = eval_df.head(limit).copy()
    return eval_df.reset_index(drop=True)


def _load_eval_dataframe(
    jsonl_path: Optional[str | Path],
    csv_path: str | Path,
    evaluation_split: str,
    max_rows: Optional[int],
    limit: Optional[int],
) -> pd.DataFrame:
    if jsonl_path is not None:
        df = load_jsonl_prompts(jsonl_path).reset_index(drop=True)
        if limit is not None:
            df = df.head(limit).copy()
        return df.reset_index(drop=True)

    if DEFAULT_JSONL_PATH.exists():
        df = load_jsonl_prompts(DEFAULT_JSONL_PATH).reset_index(drop=True)
        if limit is not None:
            df = df.head(limit).copy()
        return df.reset_index(drop=True)

    return _load_eval_dataframe_from_csv(csv_path, evaluation_split, max_rows, limit)


def run_batch_inference(
    model_name: str,
    batch_size: int,
    max_tokens: int,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    verbose: bool = False,
    trust_remote_code: bool = True,
    adapter: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    csv_path: str = str(DEFAULT_CSV_PATH),
    evaluation_split: str = "test",
    max_rows: Optional[int] = None,
) -> BatchRunResult:
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    df = _load_eval_dataframe(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        evaluation_split=evaluation_split,
        max_rows=max_rows,
        limit=limit,
    )
    prompts = df[PROMPT_COLUMN].tolist()

    load_kwargs = {"path_or_hf_repo": model_name}
    tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
    if tokenizer_config is not None:
        load_kwargs["tokenizer_config"] = tokenizer_config

    adapter_path = _resolve_adapter_path(adapter, model_name)
    if adapter_path is not None:
        load_kwargs["adapter_path"] = str(adapter_path)
        print(f"Using adapter at {adapter_path}")

    model, tokenizer = load(**load_kwargs)

    encoded_prompts = _encode_prompts(prompts, tokenizer, system_prompt)
    predictions: List[Optional[str]] = [None] * len(encoded_prompts)

    for start, chunk in _chunk(encoded_prompts, batch_size):
        response = batch_generate(
            model,
            tokenizer,
            chunk,
            max_tokens=max_tokens,
            verbose=verbose,
            completion_batch_size=min(batch_size, len(chunk)),
            prefill_batch_size=min(batch_size, len(chunk)),
        )

        for offset, text in enumerate(response.texts):
            predictions[start + offset] = extract_default_label(text)
            predicted = predictions[start + offset]
            actual = df[LABEL_COLUMN].iat[start + offset]
            predicted_display = predicted if predicted is not None else "None"
            print(f"Predicted: {predicted_display}, Actual: {actual}")

    df[f"{model_name}-default"] = predictions

    pred_series = df[f"{model_name}-default"].fillna("unknown")
    actual_series = df[LABEL_COLUMN]
    accuracy = float(np.mean(pred_series == actual_series))

    tp = int(np.sum((pred_series == "yes") & (actual_series == "yes")))
    tn = int(np.sum((pred_series == "no") & (actual_series == "no")))
    fp = int(np.sum((pred_series == "yes") & (actual_series == "no")))
    fn = int(np.sum((pred_series == "no") & (actual_series == "yes")))

    return BatchRunResult(df, accuracy, predictions, tp=tp, tn=tn, fp=fp, fn=fn)


def run_provider(
    models: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    verbose: bool = False,
    trust_remote_code: bool = True,
    adapter: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    csv_path: str = str(DEFAULT_CSV_PATH),
    evaluation_split: str = "test",
    max_rows: Optional[int] = None,
):
    models_to_use = list(models) if models is not None else [DEFAULT_MODEL]

    results = {}
    combined_df: Optional[pd.DataFrame] = None

    for model_name in models_to_use:
        print(f"Processing with {model_name}...")
        run_result = run_batch_inference(
            model_name=model_name,
            batch_size=batch_size,
            max_tokens=max_tokens,
            limit=limit,
            system_prompt=system_prompt,
            verbose=verbose,
            trust_remote_code=trust_remote_code,
            adapter=adapter,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
            evaluation_split=evaluation_split,
            max_rows=max_rows,
        )

        print(
            f"{model_name} accuracy: {run_result.accuracy * 100:.2f}% over "
            f"{len(run_result.dataframe)} samples"
        )
        print(
            f"Confusion matrix counts: TP={run_result.tp}, TN={run_result.tn}, "
            f"FP={run_result.fp}, FN={run_result.fn}"
        )

        results[model_name] = {
            "accuracy": run_result.accuracy,
            "sample_size": len(run_result.dataframe),
            "tp": run_result.tp,
            "tn": run_result.tn,
            "fp": run_result.fp,
            "fn": run_result.fn,
        }

        combined_df = run_result.dataframe

    return combined_df, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run credit default classification using MLX batch inference"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model identifier to load with mlx_lm.load",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples processed",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="Optional JSONL evaluation file. Defaults to train/credit_data/test.jsonl if present.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Raw credit CSV to use when a JSONL evaluation file is not supplied",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("train", "valid", "test", "all"),
        default="test",
        help="Split to evaluate when loading directly from the raw CSV",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows used from the raw CSV while preserving time coverage",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trusting remote code when loading the tokenizer",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Run without the default system prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from batch generation",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=(
            "Name or path of adapter to load. If a relative name is provided, "
            "it is resolved against the adapters directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    system_prompt = None if args.no_system_prompt else SYSTEM_PROMPT
    trust_remote_code = not args.no_trust_remote_code

    _, results = run_provider(
        models=[args.model],
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        limit=args.limit,
        system_prompt=system_prompt,
        verbose=args.verbose,
        trust_remote_code=trust_remote_code,
        adapter=args.adapter,
        jsonl_path=args.jsonl_path,
        csv_path=args.csv_path,
        evaluation_split=args.evaluation_split,
        max_rows=args.max_rows,
    )

    model_result = results[args.model]
    elapsed = time.perf_counter() - start_time
    print("\nFinal Results:")
    print(
        f"  Accuracy: {model_result['accuracy'] * 100:.2f}%\n"
        f"  Sample Size: {model_result['sample_size']}\n"
        f"  TP: {model_result['tp']}\n"
        f"  TN: {model_result['tn']}\n"
        f"  FP: {model_result['fp']}\n"
        f"  FN: {model_result['fn']}\n"
        f"  Total Time: {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
