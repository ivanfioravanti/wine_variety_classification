import pandas as pd
import numpy as np
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED


def load_wine_data(csv_path: str = "data/winemag-data-130k-v2.csv") -> pd.DataFrame:
    """Load the wine reviews CSV."""
    return pd.read_csv(csv_path)


def filter_by_country(df: pd.DataFrame, country: str = COUNTRY) -> pd.DataFrame:
    """Return only rows for ``country``."""
    return df[df["country"] == country]


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
    """Sample ``n`` rows from ``df`` with a fixed ``random_state``."""
    return df.sample(n=n, random_state=random_state)


def prepare_wine_data(
    csv_path: str = "data/winemag-data-130k-v2.csv",
    country: str = COUNTRY,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
    min_count: int = 5,
):
    """Load, filter and sample the wine dataset.

    Returns the sampled DataFrame and an array of unique varieties after filtering.
    """
    df = load_wine_data(csv_path)
    df = filter_by_country(df, country)
    df = remove_rare_varieties(df, min_count)
    df_subset = sample_rows(df, sample_size, random_state)
    varieties = np.array(df["variety"].unique()).astype("str")
    return df_subset, varieties
