import argparse
from anthropic import Anthropic
import time
import json
import os
from tqdm import tqdm
import numpy as np
import concurrent.futures
from data_utils import prepare_wine_data
from dotenv import load_dotenv
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED



# Define default models
DEFAULT_MODELS = [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

# Load environment variables from .env file
load_dotenv()

# Initialize the Anthropic client once with API key and timeout
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=20.0)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Column name for prompts in the Hugging Face dataset
PROMPT_COLUMN = "prompt"

# Load and prepare the dataset from Hugging Face
df_country_subset, varieties = prepare_wine_data()
jsonl_data = df_country_subset.copy()


def generate_prompt(index):
    """Return the prompt string for the given dataset index."""
    if jsonl_data.empty:
        raise ValueError("Dataset is not loaded or is empty.")
    if index >= len(jsonl_data):
        raise IndexError(f"Index {index} is out of bounds for dataset with length {len(jsonl_data)}.")

    entry = jsonl_data.iloc[index]
    prompt_value = entry.get(PROMPT_COLUMN)
    if not isinstance(prompt_value, str):
        raise KeyError(
            f"Column '{PROMPT_COLUMN}' missing or not a string at index {index}."
        )

    return prompt_value


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    while True:
        try:
            response = client.messages.create(
                max_tokens=1024,
                system="""
                You're a sommelier expert and you know everything about wine. 
                You answer precisely with the name of the variety/blend without any additional text,
                using format { "variety" : "answer" }
                """,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                model=model,
            )
            return json.loads(response.content[0].text.strip())["variety"]
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")


def process_example(index, row, model, df, progress_bar):
    global progress_index

    try:
        # Generate the prompt using the index
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

        progress_index += 1
    except Exception as e:
        print(f"Error processing model {model}: {str(e)}")


def process_dataframe(df, model):
    global progress_index
    progress_index = 1  # Reset progress index

    # Create a tqdm progress bar
    with tqdm(total=len(df), desc="Processing rows") as progress_bar:
        # Process each example concurrently using ThreadPoolExecutor with limited workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run wine variety classification with Anthropic models"
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        help="Override the default model list",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_provider(models=arguments.model)
