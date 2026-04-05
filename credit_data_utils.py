from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED


PROMPT_COLUMN = "prompt"
COMPLETION_COLUMN = "completion"
LABEL_COLUMN = "default_label"
TARGET_COLUMN = "loan_status"
SPLIT_DATE_COLUMN = "issue_d"
PARSED_SPLIT_DATE_COLUMN = "_issue_d_parsed"

POSITIVE_STATUS = "Charged Off"
NEGATIVE_STATUS = "Fully Paid"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "loan_processed_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train" / "credit_data"

PROMPT_FIELDS: Sequence[Tuple[str, str]] = (
    ("loan_amnt", "Loan amount"),
    ("term", "Term"),
    ("emp_length", "Employment length"),
    ("home_ownership", "Home ownership"),
    ("annual_inc", "Annual income"),
    ("verification_status", "Verification status"),
    ("purpose", "Loan purpose"),
    ("addr_state", "Borrower state"),
    ("dti", "Debt-to-income ratio"),
    ("delinq_2yrs", "Delinquencies in past 2 years"),
    ("earliest_cr_line", "Earliest credit line"),
    ("fico_range_high", "FICO high range"),
    ("inq_last_6mths", "Credit inquiries in last 6 months"),
    ("open_acc", "Open accounts"),
    ("pub_rec", "Public records"),
    ("revol_bal", "Revolving balance"),
    ("revol_util", "Revolving utilization"),
    ("total_acc", "Total accounts"),
    ("initial_list_status", "Initial list status"),
    ("application_type", "Application type"),
    ("acc_now_delinq", "Accounts currently delinquent"),
    ("tot_cur_bal", "Total current balance"),
    ("acc_open_past_24mths", "Accounts opened in past 24 months"),
    ("bc_open_to_buy", "Bankcard open to buy"),
    ("mort_acc", "Mortgage accounts"),
    ("mths_since_recent_inq", "Months since recent inquiry"),
    ("num_accts_ever_120_pd", "Accounts ever 120+ days past due"),
    ("num_rev_tl_bal_gt_0", "Revolving trades with balance"),
    ("num_tl_op_past_12m", "Trades opened in past 12 months"),
    ("pct_tl_nvr_dlq", "Percent trades never delinquent"),
    ("pub_rec_bankruptcies", "Public record bankruptcies"),
    ("tax_liens", "Tax liens"),
    ("total_bal_ex_mort", "Total balance excluding mortgage"),
    ("disbursement_method", "Disbursement method"),
)


def _format_value(value) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    value_str = str(value).strip()
    return value_str if value_str else "unknown"


def build_credit_prompt(row: pd.Series) -> str:
    lines = [
        "Based only on these loan-origination features, predict whether the loan will eventually default.",
        'Respond in JSON format as {"default": "yes"} or {"default": "no"}.',
        "",
    ]
    for field_name, label in PROMPT_FIELDS:
        lines.append(f"{label}: {_format_value(row.get(field_name))}")
    return "\n".join(lines)


def build_completion(label: str) -> str:
    return json.dumps({"default": label})


def extract_default_label(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    candidate = raw_text.strip()
    try:
        value = json.loads(candidate)["default"]
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                return normalized
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            value = json.loads(snippet)["default"]
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                    return normalized
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    normalized = candidate.lower()
    if POSITIVE_LABEL in normalized:
        return POSITIVE_LABEL
    if NEGATIVE_LABEL in normalized:
        return NEGATIVE_LABEL

    return None


def _sample_dataframe(
    df: pd.DataFrame,
    max_rows: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df.copy()

    if PARSED_SPLIT_DATE_COLUMN in df.columns:
        df_sorted = df.sort_values(by=PARSED_SPLIT_DATE_COLUMN, kind="stable").reset_index(
            drop=True
        )
        indices = np.linspace(0, len(df_sorted) - 1, num=max_rows, dtype=int)
        sampled_df = df_sorted.iloc[indices].copy()
        return sampled_df.reset_index(drop=True)

    sampled_df, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )
    return sampled_df.reset_index(drop=True)


def load_credit_data(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    max_rows: Optional[int] = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Credit dataset not found at {csv_path}")

    usecols = [TARGET_COLUMN, SPLIT_DATE_COLUMN] + [
        field_name for field_name, _ in PROMPT_FIELDS
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[df[TARGET_COLUMN].isin({POSITIVE_STATUS, NEGATIVE_STATUS})].copy()
    df[PARSED_SPLIT_DATE_COLUMN] = pd.to_datetime(
        df[SPLIT_DATE_COLUMN],
        format="%b-%Y",
        errors="coerce",
    )
    df = df.dropna(subset=[PARSED_SPLIT_DATE_COLUMN]).copy()

    df[LABEL_COLUMN] = df[TARGET_COLUMN].map(
        {
            POSITIVE_STATUS: POSITIVE_LABEL,
            NEGATIVE_STATUS: NEGATIVE_LABEL,
        }
    )
    df = _sample_dataframe(df, max_rows=max_rows, random_state=random_state)
    df[PROMPT_COLUMN] = [build_credit_prompt(row) for _, row in df.iterrows()]
    df[COMPLETION_COLUMN] = df[LABEL_COLUMN].map(build_completion)
    return df.reset_index(drop=True)


def split_credit_data(
    df: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if valid_size <= 0 or test_size <= 0 or (valid_size + test_size) >= 1:
        raise ValueError("Validation and test sizes must be positive and sum to less than 1.")

    if PARSED_SPLIT_DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Expected column {PARSED_SPLIT_DATE_COLUMN!r} for time-based splitting."
        )

    df_sorted = df.sort_values(
        by=[PARSED_SPLIT_DATE_COLUMN, TARGET_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    total_rows = len(df_sorted)
    test_count = int(total_rows * test_size)
    valid_count = int(total_rows * valid_size)
    train_count = total_rows - valid_count - test_count

    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise ValueError("Split sizes produced an empty split. Adjust valid_size/test_size.")

    train_df = df_sorted.iloc[:train_count].copy()
    valid_df = df_sorted.iloc[train_count : train_count + valid_count].copy()
    test_df = df_sorted.iloc[train_count + valid_count :].copy()

    for split_df in (train_df, valid_df, test_df):
        split_df.drop(columns=[SPLIT_DATE_COLUMN, PARSED_SPLIT_DATE_COLUMN], inplace=True)

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def write_jsonl(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        for _, row in df.iterrows():
            record = {
                PROMPT_COLUMN: row[PROMPT_COLUMN],
                COMPLETION_COLUMN: row[COMPLETION_COLUMN],
            }
            handle.write(json.dumps(record) + "\n")


def load_jsonl_prompts(jsonl_path: str | Path) -> pd.DataFrame:
    jsonl_path = Path(jsonl_path)
    records = []
    with jsonl_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            completion = record.get(COMPLETION_COLUMN, "")
            records.append(
                {
                    PROMPT_COLUMN: record.get(PROMPT_COLUMN, ""),
                    COMPLETION_COLUMN: completion,
                    LABEL_COLUMN: extract_default_label(completion),
                }
            )
    return pd.DataFrame(records)
