from openai import OpenAI
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Wine variety classification using MLX Server"
)

# Define default models for mlx_omni_server provider
DEFAULT_MODELS = ["mlx-community/Qwen3-0.6B-8bit","mlx-community/Qwen3-0.6B-bf16","mlx-community/Qwen3-1.7B-4bit", "mlx-community/Qwen3-1.7B-8bit","mlx-community/Qwen3-4B-4bit", 
                  "mlx-community/Qwen3-4B-8bit", "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-8B-8bit","mlx-community/Qwen3-14B-4bit","mlx-community/Qwen3-14B-8bit", 
                  "mlx-community/Qwen3-30B-A3B-4bit", "mlx-community/Qwen3-30B-A3B-8bit", "mlx-community/Qwen3-32B-4bit", 
                  "mlx-community/Qwen3-235B-A22B-4bit", "mlx-community/Qwen3-235B-A22B-8bit"]

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

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

# Modified sampling to include random_state
df_country_subset = df_country.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

varieties = np.array(df_country["variety"].unique()).astype("str")


def generate_prompt(row, varieties):
    # Format the varieties list as a comma-separated string
    variety_list = ", ".join(varieties)

    prompt = f"""
    /no_think
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": 
                     """
                        You're a sommelier expert and you know everything about wine. 
                        You answer precisely with the name of the variety/blend without any additional text,
                        using JSON format: { "variety" : "answer" }   
                        don't add anything else, just the answer in the given format.
                        Don't add json in front of the response.
                        Don't forget key variety in the json result.
                        These are wrong format of answers:
                        {"Champagne Blend" : "answer"}
                        {"Grenache"} - missing variety key
                        {"Rosé"} - missing variety key
                        Good ones are:
                        { "variety" : "Chardonnay" }
                        { "variety" : "Bordeaux-style White Blend"}
                    """},            
            {"role": "user", "content": prompt},
        ],
    }

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    
    # Find the JSON part of the response
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in response")
        
    json_str = content[start_idx:end_idx]
    return json.loads(json_str)["variety"]


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

from openai import OpenAI
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Wine variety classification using MLX Server"
)

# Define default models for mlx_omni_server provider
DEFAULT_MODELS = ["ivanfioravanti/Qwen2.5-3B-Instruct-italian-wine-fp16"]

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

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

# Modified sampling to include random_state
df_country_subset = df_country.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

varieties = np.array(df_country["variety"].unique()).astype("str")


def generate_prompt(row, varieties):
    # Format the varieties list as a comma-separated string
    variety_list = ", ".join(varieties)

    prompt = f"""
    /no_think
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": 
                     """
                        You're a sommelier expert and you know everything about wine. 
                        You answer precisely with the name of the variety/blend without any additional text,
                        using JSON format: { "variety" : "answer" }   
                        don't add anything else, just the answer in the given format.
                        Don't add json in front of the response.
                        Don't forget key variety in the json result.
                        These are wrong format of answers:
                        {"Champagne Blend" : "answer"}
                        {"Grenache"} - missing variety key
                        {"Rosé"} - missing variety key
                        Good ones are:
                        { "variety" : "Chardonnay" }
                        { "variety" : "Bordeaux-style White Blend"}
                    """},            
            {"role": "user", "content": prompt},
        ],
    }

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    
    # Find the JSON part of the response
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in response")
        
    json_str = content[start_idx:end_idx]
    return json.loads(json_str)["variety"]


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
