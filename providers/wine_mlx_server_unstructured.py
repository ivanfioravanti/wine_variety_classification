from openai import OpenAI
import json
from tqdm import tqdm
import numpy as np
from data_utils import prepare_wine_data
import argparse
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Wine variety classification using MLX Server"
)

# Define default models for mlx_omni_server provider
DEFAULT_MODELS = ["mlx-community/Qwen3-0.6B-bf16"]

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Load and prepare the dataset
df_country_subset, varieties = prepare_wine_data()

def generate_prompt(row, varieties):
    # Format the varieties list as a comma-separated string
    variety_list = ", ".join(varieties)

    prompt = f"""
    Based on this wine review, guess the grape variety:
    This wine is produced by {row['winery']} in the {row['province']} region of {row['country']}.
    It was grown in {row['region_1']}. It is described as: "{row['description']}".
    The wine has been reviewed by {row['taster_name']} and received {row['points']} points.
    The price is {row['price']}.

    """
    return prompt


# Function to call the API and process the result for a single model (blocking call in this case)
def call_model(model, prompt):
    kwargs = {
        "model": model,
        "messages": [
            # {
            #     "role": "system",
            #     "content": 
            #          """
            #             You're a sommelier expert and you know everything about wine. 
            #             You answer precisely with the name of the variety/blend without any additional text,
            #             using JSON format: { "variety" : "answer" }   
            #             don't add anything else, just the answer in the given format.
            #             Don't add json in front of the response.
            #             Don't forget key variety in the json result.
            #             These are wrong format of answers:
            #             {"Champagne Blend" : "answer"}
            #             {"Grenache"} - missing variety key
            #             {"Rosé"} - missing variety key
            #             Good ones are:
            #             { "variety" : "Chardonnay" }
            #             { "variety" : "Bordeaux-style White Blend"}
            #         """},            
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7, 
        "extra_body" : { "adapters": "/Users/ifioravanti/adapters" }
    }

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    
    # Extract token usage
    usage = response.usage
    total_tokens = usage.total_tokens if usage else 0

    try:
        # Find the JSON part of the response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
            
        json_str = content[start_idx:end_idx]
        # return json.loads(json_str)["variety"]
        return json.loads(json_str)["variety"], total_tokens
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        print(f"Error processing content from model {model}. Raw content received: '{content}'")
        raise e # Re-raise the exception to be handled by the caller


def process_example(index, row, model, df, progress_bar):
    global progress_index
    tokens_processed = 0

    try:
        # Generate the prompt using the row
        prompt = generate_prompt(row, varieties)

        # predicted_variety = call_model(model, prompt)
        predicted_variety, tokens = call_model(model, prompt)
        tokens_processed += tokens
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
        tqdm.write(f"Error processing model {model}: {str(e)}")

    return tokens_processed


def process_dataframe(df, model):
    global progress_index
    progress_index = 1  # Reset progress index
    total_tokens_processed = 0

    # Create a tqdm progress bar
    with tqdm(total=len(df), desc="Processing rows") as progress_bar:
        # Process each example sequentially
        for index, row in df.iterrows():
            try:
                # process_example(index, row, model, df, progress_bar)
                tokens = process_example(index, row, model, df, progress_bar)
                total_tokens_processed += tokens
            except Exception as e:
                print(f"Error processing example: {str(e)}")

    # return df
    return df, total_tokens_processed


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
        # df = process_dataframe(df_country_subset.copy(), model)
        df, total_tokens = process_dataframe(df_country_subset.copy(), model)
        accuracy = get_accuracy(model, df)
        results[model] = {
            "accuracy": accuracy,
            "sample_size": len(df),
            "country": COUNTRY,
            "total_tokens": total_tokens,
        }
        print(f"{model} accuracy: {accuracy * 100:.2f}%")
        print(f"{model} total tokens: {total_tokens}")

    return df, results


if __name__ == "__main__":
    df_results, accuracies = run_provider()
    print("\nFinal Results:")
    for model, data in accuracies.items():
        print(f"  {model}:")
        print(f"    Accuracy: {data['accuracy'] * 100:.2f}%")
        print(f"    Sample Size: {data['sample_size']}")
        print(f"    Country: {data['country']}")
        print(f"    Total Tokens: {data['total_tokens']}")

    # Optionally, save the DataFrame with all results to a CSV file
    # df_results.to_csv("wine_classification_results.csv", index=False)
    # print("\nResults saved to wine_classification_results.csv")


