import argparse
import openai
import time
import json
import os
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import concurrent.futures
from data_utils import prepare_wine_data, build_prompt, PROMPT_COLUMN
from dotenv import load_dotenv
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Define default models
DEFAULT_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
MAX_WORKERS = 4

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load and prepare the dataset from Hugging Face
df_country_subset, varieties = prepare_wine_data()
jsonl_data = df_country_subset.copy()


def generate_prompt(index):
    """Generates a prompt using the entry at the given index from the loaded JSONL data."""
    if jsonl_data.empty:
        raise ValueError("Dataset is not loaded or is empty.")
    if index >= len(jsonl_data):
        raise IndexError(
            f"Index {index} is out of bounds for dataset with length {len(jsonl_data)}."
        )

    entry = jsonl_data.iloc[index]
    prompt_value = entry.get(PROMPT_COLUMN)
    if not isinstance(prompt_value, str):
        prompt_value = build_prompt(entry)
        jsonl_data.at[index, PROMPT_COLUMN] = prompt_value

    return prompt_value


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

# Initialize the progress index
metadata_value = "wine-distillation-1000-italy"  # that's a funny metadata tag :-)


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                store=True,
                metadata={
                    "distillation": metadata_value,
                },
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
        except openai.RateLimitError as e:
            retry_after = 1  # Default to 1 second if no explicit time is provided
            if (
                hasattr(e, "response")
                and e.response
                and "retry-after" in e.response.headers
            ):
                retry_after = float(e.response.headers["retry-after"])
            print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)


def process_example(index, row, model, df, progress_bar):
    try:
        # Generate the prompt using the row
        prompt = generate_prompt(index)

        predicted_variety = call_model(model, prompt)
        df.at[index, model + "-variety"] = predicted_variety

        actual_variety = row["variety"]
        if predicted_variety == actual_variety:
            tqdm.write(f"✅ Predicted: {predicted_variety}, Actual: {actual_variety}")
        else:
            tqdm.write(f"❌ Predicted: {predicted_variety}, Actual: {actual_variety}")

        # Update the progress bar
        progress_bar.update(1)
    except Exception as e:
        print(f"Error processing model {model}: {str(e)}")


def process_dataframe(df, model):
    # Create a tqdm progress bar
    with tqdm(total=len(df), desc="Processing rows") as progress_bar:
        # Process each example concurrently using ThreadPoolExecutor with limited workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_example, index, row, model, df, progress_bar
                ): index
                for index, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Wait for each example to be processed
                except Exception as e:
                    print(f"Error processing example: {str(e)}")
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
    models_to_use = (
        list(models) if models is not None and len(models) > 0 else DEFAULT_MODELS
    )
    results = {}
    final_df = None

    for model in models_to_use:
        print(f"Processing with {model}...")
        working_df = process_dataframe(df_country_subset.copy(), model)
        accuracy = get_accuracy(model, working_df)
        results[model] = {
            "accuracy": accuracy,
            "sample_size": len(working_df),
            "country": COUNTRY,
        }
        print(f"{model} accuracy: {accuracy * 100:.2f}%")
        final_df = working_df if final_df is None else final_df

    return final_df, results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run wine variety classification using OpenAI chat models"
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        help="Override the list of models to evaluate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_provider(models=arguments.model)
