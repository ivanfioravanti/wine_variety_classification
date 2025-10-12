import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED


PROMPT_COLUMN = "prompt"


def _format_field(value, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    value_str = str(value).strip()
    return value_str if value_str else fallback


def build_prompt(row: pd.Series) -> str:
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


def load_wine_data(split: str = "train") -> pd.DataFrame:
    """Load the wine reviews dataset from Hugging Face and return as DataFrame."""
    dataset = load_dataset("spawn99/wine-reviews", split=split)
    df = dataset.to_pandas()
    df = df.dropna(subset=["variety"]).copy()
    return df


def filter_by_country(df: pd.DataFrame, country: str = COUNTRY) -> pd.DataFrame:
    """Return only rows for ``country``."""
    if country is None:
        return df
    return df[df["country"].fillna("") == country]


def remove_rare_varieties(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Drop varieties that appear fewer than ``min_count`` times."""
    counts = df["variety"].value_counts()
    rare_varieties = counts[counts < min_count].index
    return df[~df["variety"].isin(rare_varieties)]


def sample_rows(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample ``n`` rows from ``df`` with fallback if ``n`` exceeds available rows."""
    if n >= len(df):
        return df.sample(n=len(df), random_state=random_state)
    return df.sample(n=n, random_state=random_state)


def prepare_wine_data(
    split: str = "train",
    country: str = COUNTRY,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
    min_count: int = 5,
):
    """Load, filter and sample the wine dataset.

    Returns the sampled DataFrame and an array of unique varieties after filtering.
    """
    df = load_wine_data(split)
    df = filter_by_country(df, country).reset_index(drop=True)
    df = remove_rare_varieties(df, min_count).reset_index(drop=True)
    df_subset = sample_rows(df, sample_size, random_state).reset_index(drop=True)
    if PROMPT_COLUMN not in df_subset.columns:
        df_subset[PROMPT_COLUMN] = [build_prompt(row) for _, row in df_subset.iterrows()]
    df_subset[PROMPT_COLUMN] = df_subset[PROMPT_COLUMN].astype(str)
    varieties = np.array(df["variety"].unique()).astype("str")
    return df_subset, varieties
