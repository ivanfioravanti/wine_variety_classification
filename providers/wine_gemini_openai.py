from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED
import os
import time
import json
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import concurrent.futures
from data_utils import prepare_wine_data

# Define default models
DEFAULT_MODELS = ["gemini-2.5-pro-preview-05-06"]

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

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
                    print(f"Warning: Skipping malformed JSON line in {file_path}: {line.strip()} - Error: {e}")
        if not data:
            print(f"Warning: No data loaded from {file_path}. File might be empty or all lines were malformed.")
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists in the correct location.")
        return [] # Return empty list if file not found
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load data from JSONL file at the beginning
jsonl_data = load_jsonl_data()

# Load and prepare the dataset
df_country_subset, varieties = prepare_wine_data()


def generate_prompt(index):
    """
    Generates a prompt using the entry at the given index from the loaded JSONL data.
    """
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
        "schema": {
            "type": "object",
            "properties": {"variety": {"type": "string", "enum": varieties.tolist()}},
            "additionalProperties": False,
            "required": ["variety"],
        },
        "strict": True,
    },
}


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
            return json.loads(response.choices[0].message.content.strip())["variety"]
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")


def process_example(df_index, row, jsonl_idx, model, df, progress_bar):
    global progress_index

    try:
        # Generate the prompt using the jsonl_idx
        prompt = generate_prompt(jsonl_idx)

        df.at[df_index, model + "-variety"] = call_model(model, prompt)

        # Update the progress bar
        progress_bar.update(1)

        progress_index += 1
    except Exception as e:
        tqdm.write(f"Error in process_example for df_index {df_index}, jsonl_idx {jsonl_idx}, model {model}: {str(e)}")


def process_dataframe(df, model):
    global progress_index
    progress_index = 1  # Reset progress index

    if not jsonl_data:
        tqdm.write("Warning: JSONL data is not loaded or is empty. No examples will be processed.")
        return df 

    num_examples_to_process = min(len(df), len(jsonl_data))
    if num_examples_to_process == 0:
        tqdm.write("Warning: No examples to process (df is empty, jsonl_data is empty, or jsonl is shorter).")
        return df

    with tqdm(total=num_examples_to_process, desc=f"Processing rows for model {model}") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for i in range(num_examples_to_process):
                df_idx = df.index[i]
                row = df.iloc[i]
                future = executor.submit(process_example, df_idx, row, i, model, df, progress_bar)
                futures[future] = df_idx

            for future in concurrent.futures.as_completed(futures):
                original_df_idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"Error in completed future for df_index {original_df_idx}, model {model}: {str(e)}")
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
    if not jsonl_data:
        print("Exiting: No data loaded from JSONL file. Ensure 'test.jsonl' is present and valid in ./train/data/test.jsonl.")
    else:
        run_provider()
