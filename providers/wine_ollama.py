from ollama import chat
from tqdm import tqdm
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

df = pd.read_csv("data/winemag-data-130k-v2.csv")
df_country = df[df["country"] == COUNTRY]

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Let's also filter out wines that have less than 5 references with their grape variety – even though we'd like to find those
# they're outliers that we don't want to optimize for that would make our enum list be too long
# and they could also add noise for the rest of the dataset on which we'd like to guess, eventually reducing our accuracy.

varieties_less_than_five_list = (
    df_country["variety"]
    .value_counts()[df_country["variety"].value_counts() < 5]
    .index.tolist()
)
df_country = df_country[~df_country["variety"].isin(varieties_less_than_five_list)]

df_country_subset = df_country.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

varieties = np.array(df_country["variety"].unique()).astype("str")


def generate_prompt(row, varieties):
    # Format the varieties list as a comma-separated string
    variety_list = ", ".join(varieties)

    prompt = f"""
    /
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt


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
        # Generate the prompt using the row
        prompt = generate_prompt(row, varieties)

        df.at[index, model + "-variety"] = call_model(model, prompt)

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
DEFAULT_MODELS = [
    "qwen3:8b-fp16",
]

# DEFAULT_MODELS = ["llama3.2:latest", "llama3.3:latest"]


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
