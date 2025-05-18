import os
import time
import json
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import concurrent.futures
from data_utils import prepare_wine_data
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED

# Define default models
DEFAULT_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

# Load environment variables from .env file
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

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

    Here is a list of possible grape varieties to choose from: {variety_list}.
    
    What is the likely grape variety? Answer only with the grape variety name or blend from the list.
    """
    return prompt


class WineVariety(BaseModel):
    variety: str = Field(enum=varieties.tolist())


# Function to call the API and process the result for a single model
def call_model(model_name, prompt, timeout=5, max_retries=3):
    def _generate():
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction="You're a sommelier expert and you know everything about wine. You answer precisely with the name of the variety/blend.",
                temperature=0,
                response_schema=WineVariety,
            ),
        )
        return json.loads(response.text.strip())["variety"]

    retries = 0
    while retries < max_retries:
        executor = None
        future = None
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_generate)
            variety = future.result(
                timeout=timeout
            )  # This will raise TimeoutError if it takes too long

            # Validate that the variety is in our list
            if variety in varieties:
                return variety
            retries += 1  # Count invalid responses as retries

        except concurrent.futures.TimeoutError:
            print(
                f"Timeout error for model {model_name}: Request took longer than {timeout} seconds (attempt {retries + 1}/{max_retries})"
            )
            retries += 1
            time.sleep(1)  # Add a small delay before retrying
        except Exception as e:
            print(
                f"Error processing model {model_name}: {str(e)} (attempt {retries + 1}/{max_retries})"
            )
            retries += 1
            time.sleep(1)  # Add a small delay before retrying
        finally:
            # Clean up the executor and cancel any pending future
            if future is not None:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=False)

    raise Exception(f"Failed to get valid response after {max_retries} attempts")


def process_example(index, row, model, df, progress_bar):
    global progress_index

    try:
        # Generate the prompt using the row
        prompt = generate_prompt(row, varieties)

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
