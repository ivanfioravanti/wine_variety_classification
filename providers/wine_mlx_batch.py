"""Batch inference provider using MLX for wine variety classification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import COUNTRY, RANDOM_SEED
from data_utils import prepare_wine_data

try:
    from mlx_lm import batch_generate, load
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "mlx_lm is required for the MLX batch provider. Install it via `pip install mlx-lm`."
    ) from exc


PROMPT_COLUMN = "prompt"
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SYSTEM_PROMPT = (
    "You're a sommelier expert and you know everything about wine. "
    "You answer precisely with the name of the variety/blend in JSON format: "
    '{"variety": "<answer>"}.'
)
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 64


np.random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DEFAULT_ADAPTER_SUFFIXES = (".safetensors", ".bin", ".json", ".pt")
DEBUG_PREFIX = "[wine_mlx_batch]"


@dataclass
class BatchRunResult:
    dataframe: pd.DataFrame
    accuracy: float
    predictions: List[Optional[str]]


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


def _extract_variety(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    candidate = raw_text.strip()

    try:
        return json.loads(candidate)["variety"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)["variety"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return None


def _load_dataset(limit: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    df, varieties = prepare_wine_data()
    if limit is not None:
        df = df.head(limit).copy()
    df = df.reset_index(drop=True)

    if PROMPT_COLUMN not in df.columns:
        df[PROMPT_COLUMN] = [_build_prompt(row) for _, row in df.iterrows()]

    return df, varieties


def _format_field(value, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    value_str = str(value).strip()
    return value_str if value_str else fallback


def _build_prompt(row: pd.Series) -> str:
    winery = _format_field(row.get("winery"), "a winery")
    country = _format_field(row.get("country"), "Unknown country")
    region = _format_field(
        row.get("region_1") or row.get("province") or row.get("region_2"),
        "Unknown region",
    )
    designation = _format_field(row.get("designation"), "an unspecified appellation")
    description = _format_field(row.get("description"), "No description provided.")
    taster = _format_field(row.get("taster_name"), "a reviewer")
    points = _format_field(row.get("points"), "unrated")
    price = row.get("price")
    price_str = (
        f"{price:.0f}"
        if isinstance(price, (int, float)) and not np.isnan(price)
        else _format_field(price, "unknown")
    )

    return (
        "Based on this wine review, guess the grape variety:\n"
        f"This wine is produced by {winery} in the {region} region of {country}.\n"
        f"It was grown in {designation}. It is described as: \"{description}\".\n"
        f"The wine has been reviewed by {taster} and received {points} points.\n"
        f"The price is {price_str}."
    )


def _resolve_adapter_path(
    adapter: Optional[str],
    model_name: str,
) -> Optional[Path]:
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.exists() and DEFAULT_ADAPTERS_DIR.exists():
            adapter_path = DEFAULT_ADAPTERS_DIR / adapter
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter path '{adapter}' could not be found."
            )
        print(f"{DEBUG_PREFIX} Using adapter provided via CLI at {adapter_path}")
        return adapter_path

    if not DEFAULT_ADAPTERS_DIR.exists():
        print(f"{DEBUG_PREFIX} Adapters directory {DEFAULT_ADAPTERS_DIR} does not exist.")
        return None

    default_files = list(DEFAULT_ADAPTERS_DIR.glob("*"))
    has_root_weights = any(
        f.is_file() and f.suffix in DEFAULT_ADAPTER_SUFFIXES for f in default_files
    )
    has_config = any(
        f.is_file() and f.name in {"adapter_config.json", "adapter_config.yaml"}
        for f in default_files
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
    elif len(candidates) > 1:
        print(
            f"{DEBUG_PREFIX} Multiple adapter candidates found for {model_name}: {candidates}."
            " Please specify one using --adapter."
        )
    else:
        print(
            f"{DEBUG_PREFIX} No adapter directory matched model {model_name} inside {DEFAULT_ADAPTERS_DIR}."
        )

    return None


def run_batch_inference(
    model_name: str,
    batch_size: int,
    max_tokens: int,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    verbose: bool = False,
    trust_remote_code: bool = True,
    adapter: Optional[str] = None,
) -> BatchRunResult:
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    df, _ = _load_dataset(limit)
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
            predictions[start + offset] = _extract_variety(text)
            predicted = predictions[start + offset]
            actual = df["variety"].iat[start + offset]
            predicted_display = predicted if predicted is not None else "None"
            if predicted == actual:
                print(f"✅ Predicted: {predicted_display}, Actual: {actual}")
            else:
                print(f"❌ Predicted: {predicted_display}, Actual: {actual}")

    column_name = f"{model_name}-variety"
    df[column_name] = predictions

    pred_series = df[column_name]
    actual_series = df["variety"]
    accuracy = float(np.mean(pred_series == actual_series))

    return BatchRunResult(df, accuracy, predictions)


def run_provider(
    models: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    verbose: bool = False,
    trust_remote_code: bool = True,
    adapter: Optional[str] = None,
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
        )

        print(
            f"{model_name} accuracy: {run_result.accuracy * 100:.2f}% over {len(run_result.dataframe)} samples"
        )

        results[model_name] = {
            "accuracy": run_result.accuracy,
            "sample_size": len(run_result.dataframe),
            "country": COUNTRY,
        }

        combined_df = run_result.dataframe

    return combined_df, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wine variety classification using MLX batch inference"
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
    )

    model_result = results[args.model]
    print("\nFinal Results:")
    print(
        f"  Accuracy: {model_result['accuracy'] * 100:.2f}%\n"
        f"  Sample Size: {model_result['sample_size']}\n"
        f"  Country: {model_result['country']}"
    )


if __name__ == "__main__":
    main()

