import argparse
from ollama import chat
from tqdm import tqdm
import numpy as np
from data_utils import prepare_wine_data
from pydantic import BaseModel, Field
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Column name for prompts in the Hugging Face dataset
PROMPT_COLUMN = "prompt"

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load and prepare the dataset from Hugging Face
df_country_subset, varieties = prepare_wine_data()
jsonl_data = df_country_subset.copy()


def generate_prompt(index):
    """Return the prompt string for the given dataset index."""
    if jsonl_data.empty:
        raise ValueError("Dataset is not loaded or is empty.")
    if index >= len(jsonl_data):
        raise IndexError(
            f"Index {index} is out of bounds for dataset with length {len(jsonl_data)}."
        )

    entry = jsonl_data.iloc[index]
    prompt_value = entry.get(PROMPT_COLUMN)
    if not isinstance(prompt_value, str):
        raise KeyError(
            f"Column '{PROMPT_COLUMN}' missing or not a string at index {index}."
        )

    return prompt_value


class WineVariety(BaseModel):
    variety: str = Field(enum=varieties.tolist())


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend.",
            },
            {"role": "user", "content": prompt},
        ],
        format=WineVariety.model_json_schema(),
    )
    wine_variety = WineVariety.model_validate_json(response.message.content)
    return wine_variety.variety


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
        # Process each example sequentially
        for index, row in df.iterrows():
            try:
                process_example(index, row, model, df, progress_bar)
            except Exception as e:
                print(f"Error processing example: {str(e)}")

    return df


def get_accuracy(model, df):
    return np.mean(df["variety"] == df[model + "-variety"])


# Default models to use when running the provider directly
DEFAULT_MODELS = ["llama3.2:latest", "llama3.3:latest"]


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
        description="Run wine variety classification using Ollama"
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        help="Override the default Ollama model list",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_provider(models=arguments.model)
