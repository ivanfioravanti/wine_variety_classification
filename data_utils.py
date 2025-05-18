import pandas as pd
import numpy as np
import json
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED


def load_wine_data(file_path: str = "train/data/test.jsonl") -> pd.DataFrame:
    """Load the wine reviews JSONL file and extract country and variety."""
    raw_df = pd.read_json(file_path, lines=True)
    
    # Extract variety from the 'completion' column (JSON string)
    def extract_variety(completion_str):
        try:
            return json.loads(completion_str)['variety']
        except (json.JSONDecodeError, KeyError, TypeError):
            return np.nan
            
    raw_df['variety'] = raw_df['completion'].apply(extract_variety)
    
    # Extract country from the 'prompt' column using regex
    # Assumes country is mentioned after "region of " and followed by a period.
    country_extract_series = raw_df['prompt'].str.extract(r"region of ([^.]+)\.", expand=False)
    raw_df['country'] = country_extract_series
    
    return raw_df


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
    file_path: str = "train/data/test.jsonl",
    country: str = COUNTRY,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
    min_count: int = 5,
):
    """Load, filter and sample the wine dataset.

    Returns the sampled DataFrame and an array of unique varieties after filtering.
    """
    df = load_wine_data(file_path)
    df = filter_by_country(df, country)
    df = remove_rare_varieties(df, min_count)
    df_subset = sample_rows(df, sample_size, random_state)
    varieties = np.array(df["variety"].unique()).astype("str")
    return df_subset, varieties
