import os
import requests
import time
import json
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import concurrent.futures
from data_utils import prepare_wine_data
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Global variable to store data from JSONL file
jsonl_data = []

# Function to load data from JSONL file
def load_jsonl_data(file_path="./train/data/test.jsonl"):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping malformed JSON line in {file_path}: {line.strip()} - Error: {e}"
                    )
        if not data:
            print(
                f"Warning: No data loaded from {file_path}. File might be empty or all lines were malformed."
            )
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists in the correct location.")
        return []  # Return empty list if file not found
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

# Define default models
DEFAULT_MODELS = ["deepseek/deepseek-chat"]

# Load environment variables from .env file
load_dotenv()

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load and prepare the dataset
df_country_subset, varieties = prepare_wine_data()

# Load data from JSONL file at the beginning
jsonl_data = load_jsonl_data()


def generate_prompt(index):
    """Generates a prompt using the entry at the given index from the loaded JSONL data."""
    if not jsonl_data:
        raise ValueError("JSONL data is not loaded or is empty.")
    if index >= len(jsonl_data):
        raise IndexError(f"Index {index} is out of bounds for JSONL data with length {len(jsonl_data)}.")

    entry = jsonl_data[index]
    if "prompt" not in entry:
        raise KeyError(f"Key 'prompt' not found in JSONL data at index {index}.")

    return entry["prompt"]


response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "grape-variety",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"variety": {"type": "string", "enum": varieties.tolist()}},
            "additionalProperties": False,
            "required": ["variety"],
        },
    },
}


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                },
                data=json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": response_format,
                    }
                ),
                timeout=timeout,
            )
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"].strip()
            return json.loads(content)["variety"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == max_retries - 1:
                print(
                    f"Failed after {max_retries} attempts for model {model}: {str(e)}"
                )
                return "ERROR: Request timeout"
            time.sleep(2**attempt)  # Exponential backoff
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
            return f"ERROR: {str(e)}"


def process_example(index, row, model, df, progress_bar):
    try:
        prompt = generate_prompt(row, varieties)
        result = call_model(model, prompt)
        df.at[index, model + "-variety"] = result

        actual_variety = row["variety"]
        if result == actual_variety:
            tqdm.write(f"✅ Predicted: {result}, Actual: {actual_variety}")
        else:
            tqdm.write(f"❌ Predicted: {result}, Actual: {actual_variety}")

        progress_bar.update(1)
    except Exception as e:
        print(f"Error processing example {index}: {str(e)}")
        df.at[index, model + "-variety"] = f"ERROR: {str(e)}"
        progress_bar.update(1)


def process_dataframe(df, model):
    with tqdm(total=len(df), desc=f"Processing {model}") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {
                executor.submit(
                    process_example, index, row, model, df, progress_bar
                ): index
                for index, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Unexpected error in future: {str(e)}")
    return df


def get_accuracy(model, df):
    return np.mean(df["variety"] == df[model + "-variety"])


def run_provider(models=None):
    """
    Run the provider with specified models or default models.
    Args:
        models: Optional list of model names to use. If None, uses DEFAULT_MODELS.
    Returns:
        DataFrame with results and accuracies for each model.
    """
    models_to_use = models if models is not None else DEFAULT_MODELS
    results = {}

    for model in models_to_use:
        print(f"Processing with {model}...")
        df = process_dataframe(df_country_subset.copy(), model)
        accuracy = get_accuracy(model, df)
        results[model] = {
            "accuracy": accuracy,
            "sample_size": len(df),
            "country": COUNTRY,
        }
        print(f"{model} accuracy: {accuracy * 100:.2f}%")

    return df, results


if __name__ == "__main__":
    run_provider()
