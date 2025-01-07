import os
import requests
import time
import json
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import concurrent.futures
import pandas as pd
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Define default models
DEFAULT_MODELS = ["deepseek/deepseek-chat"]

# Load environment variables from .env file
load_dotenv()

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

df = pd.read_csv("data/winemag-data-130k-v2.csv")
df_country = df[df["country"] == COUNTRY]

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
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt


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
            return response_json["choices"][0]["message"]["content"]
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
