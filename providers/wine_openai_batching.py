import openai
import time
import json
import os
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from data_utils import prepare_wine_data
from datetime import datetime
from dotenv import load_dotenv
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
DEFAULT_MODELS = ["gpt-4o-mini"]

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        "schema": {
            "type": "object",
            "properties": {"variety": {"type": "string", "enum": varieties.tolist()}},
            "additionalProperties": False,
            "required": ["variety"],
        },
        "strict": True,
    },
}


def create_batch_tasks(df, model):
    """Create batch tasks for wine classification."""
    tasks = []

    for index, row in df.iterrows():
        prompt = generate_prompt(row, varieties)

        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": response_format,
                "metadata": {"distillation": "true"},
            },
        }
        tasks.append(task)

    return tasks


def process_batch_results(df, results, model):
    """Process batch results and update dataframe."""
    for result in results:
        task_id = result["custom_id"]
        index = int(task_id.split("-")[1])
        variety = json.loads(
            result["response"]["body"]["choices"][0]["message"]["content"]
        )["variety"]
        df.at[index, f"{model}-variety"] = variety

        actual_variety = df.at[index, "variety"]
        if variety == actual_variety:
            tqdm.write(f"✅ Predicted: {variety}, Actual: {actual_variety}")
        else:
            tqdm.write(f"❌ Predicted: {variety}, Actual: {actual_variety}")

    return df


def process_dataframe(df, model, batch_job_id=None):
    """Process dataframe using batch API.

    Args:
        df: DataFrame to process
        model: Model name to use
        batch_job_id: Optional batch job ID to resume processing

    Returns:
        Tuple of (processed DataFrame, batch_job_id)
    """
    if batch_job_id:
        print(f"Retrieving existing batch job {batch_job_id}...")
        batch_job = client.batches.retrieve(batch_job_id)
    else:
        print(f"Creating batch tasks for {model}...")
        tasks = create_batch_tasks(df, model)

        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save tasks to JSONL file
        batch_file_path = f"data/batch_tasks_{model}_{timestamp}.jsonl"
        with open(batch_file_path, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")

        # Upload file and create batch job
        print("Uploading batch file...")
        batch_file = client.files.create(
            file=open(batch_file_path, "rb"), purpose="batch"
        )

        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    # Print current job ID for reference
    print(f"Batch Job ID: {batch_job.id}")

    # Wait for job completion
    while True:
        job_status = client.batches.retrieve(batch_job.id)
        if job_status.status == "completed":
            break
        print(f"Waiting for batch job completion... Status: {job_status.status}")
        time.sleep(30)  # Check every 30 seconds

    # Get results
    print("Processing results...")
    result_file_id = job_status.output_file_id
    result_content = client.files.content(result_file_id).content

    # Save results using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_path = f"data/batch_results_{model}_{timestamp}.jsonl"
    with open(result_file_path, "wb") as file:
        file.write(result_content)

    # Process results
    results = []
    with open(result_file_path, "r") as file:
        for line in file:
            results.append(json.loads(line.strip()))

    processed_df = process_batch_results(df, results, model)
    return processed_df, batch_job.id


def get_accuracy(model, df):
    return np.mean(df["variety"] == df[model + "-variety"])


def run_provider(models=None, batch_job_ids=None):
    """
    Run the provider with specified models or default models using batch processing.
    Args:
        models: Optional list of model names to use. If None, uses DEFAULT_MODELS.
        batch_job_ids: Optional dict mapping model names to batch job IDs for resuming jobs.
    Returns:
        Tuple of (DataFrame with results, accuracies for each model, dict of batch job IDs).
    """
    models_to_use = models if models is not None else DEFAULT_MODELS
    batch_job_ids = batch_job_ids or {}
    results = {}
    final_df = None
    job_ids = {}

    for model in models_to_use:
        print(f"Processing with {model}...")
        working_df, job_id = process_dataframe(
            df_country_subset.copy(), model, batch_job_ids.get(model)
        )
        accuracy = get_accuracy(model, working_df)
        results[model] = {
            "accuracy": accuracy,
            "sample_size": len(working_df),
            "country": COUNTRY,
        }
        job_ids[model] = job_id
        print(f"{model} accuracy: {accuracy * 100:.2f}%")
        final_df = working_df if final_df is None else final_df

    return final_df, results, job_ids


if __name__ == "__main__":
    # Example of resuming a job:
    # batch_job_ids = {"gpt-4o-mini": "your-batch-job-id"}
    # final_df, results, job_ids = run_provider(batch_job_ids=batch_job_ids)
    final_df, results, job_ids = run_provider()
