from __future__ import annotations

import argparse
from pathlib import Path

from credit_data_utils import (
    DEFAULT_CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    LABEL_COLUMN,
    load_credit_data,
    split_credit_data,
    write_jsonl,
)
from config import RANDOM_SEED


def _print_split_stats(name: str, size: int, yes_count: int, no_count: int) -> None:
    positive_rate = (yes_count / size) * 100 if size else 0.0
    print(
        f"{name}: {size} rows | default=yes: {yes_count} | "
        f"default=no: {no_count} | positive rate: {positive_rate:.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-split train/valid/test JSONL files for credit default prediction"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help="Path to the processed credit CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where train/valid/test JSONL files will be written",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of rows while preserving time coverage",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for sampling and splitting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    print(f"Loading credit dataset from {csv_path}...")
    df = load_credit_data(
        csv_path=csv_path,
        max_rows=args.max_rows,
        random_state=args.seed,
    )

    print(f"Retained {len(df)} rows with binary loan outcomes.")
    print("Creating chronological train/valid/test splits based on issue_d...")
    train_df, valid_df, test_df = split_credit_data(
        df,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print(f"Writing JSONL files to {output_dir}...")
    write_jsonl(train_df, output_dir / "train.jsonl")
    write_jsonl(valid_df, output_dir / "valid.jsonl")
    write_jsonl(test_df, output_dir / "test.jsonl")

    print("\nSplit summary:")
    for split_name, split_df in (
        ("train", train_df),
        ("valid", valid_df),
        ("test", test_df),
    ):
        counts = split_df[LABEL_COLUMN].value_counts()
        _print_split_stats(
            split_name,
            len(split_df),
            int(counts.get("yes", 0)),
            int(counts.get("no", 0)),
        )


if __name__ == "__main__":
    main()
