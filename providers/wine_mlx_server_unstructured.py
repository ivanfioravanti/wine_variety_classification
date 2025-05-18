from openai import OpenAI
import json
from tqdm import tqdm
import numpy as np
from data_utils import prepare_wine_data
import argparse
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

# Load data from JSONL file at the beginning
jsonl_data = load_jsonl_data()

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
        if not jsonl_data:
            tqdm.write(f"Skipping example {index} due to missing JSONL data.")
            progress_bar.update(1) # Still update progress bar for skipped items
            return tokens_processed
        if index >= len(jsonl_data):
            tqdm.write(f"Skipping example {index}: Index out of bounds for JSONL data (length {len(jsonl_data)}).")
            progress_bar.update(1)
            return tokens_processed

        # Generate the prompt using the index for jsonl_data
        prompt = generate_prompt(index)
        
        entry = jsonl_data[index]
        if "completion" not in entry:
            tqdm.write(f"Skipping example {index}: 'completion' key missing in JSONL data.")
            progress_bar.update(1)
            return tokens_processed
            
        actual_variety_raw = entry["completion"]
        actual_variety = None

        try:
            if isinstance(actual_variety_raw, str):
                # Check if it's a JSON string
                if actual_variety_raw.strip().startswith('{') and actual_variety_raw.strip().endswith('}'):
                    parsed_json = json.loads(actual_variety_raw)
                    if isinstance(parsed_json, dict) and "variety" in parsed_json:
                        actual_variety = parsed_json["variety"]
                    else:
                        # It was JSON, but not the expected format
                        tqdm.write(f"Warning: 'completion' at index {index} is JSON but not in {{'variety': 'value'}} format. Raw: '{actual_variety_raw}'")
                        actual_variety = str(actual_variety_raw) # Use raw string as fallback
                else:
                    # It's a string but not JSON, use it directly
                    actual_variety = actual_variety_raw
            elif isinstance(actual_variety_raw, dict) and "variety" in actual_variety_raw:
                 # If it's already a dictionary (e.g. if json.loads in load_jsonl_data somehow doubly parsed)
                 actual_variety = actual_variety_raw["variety"]
            else:
                # Not a string, not the expected dict, convert to string as a fallback
                tqdm.write(f"Warning: 'completion' at index {index} is not a string or expected dict. Raw: '{actual_variety_raw}'")
                actual_variety = str(actual_variety_raw)

        except json.JSONDecodeError:
            tqdm.write(f"Warning: Could not parse 'completion' JSON string at index {index}: '{actual_variety_raw}'. Using raw value.")
            actual_variety = actual_variety_raw # Fallback to raw value
        except Exception as e:
            tqdm.write(f"Error processing 'completion' at index {index}: {e}. Using raw value: '{actual_variety_raw}'.")
            actual_variety = actual_variety_raw # Fallback to raw value
        
        if actual_variety is None: # Should not happen if logic is correct, but as a safeguard
            tqdm.write(f"Critical Warning: actual_variety became None at index {index}. Using raw: '{actual_variety_raw}'")
            actual_variety = str(actual_variety_raw)

        # Store the actual variety from JSONL for accuracy calculation
        df.at[index, "actual_variety_jsonl"] = actual_variety

        predicted_variety, tokens = call_model(model, prompt)
        tokens_processed += tokens
        df.at[index, model + "-variety"] = predicted_variety
        # actual_variety = row["variety"] # This is now sourced from jsonl_data

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
    
    # Ensure df has the "actual_variety_jsonl" column before starting
    if "actual_variety_jsonl" not in df.columns:
        df["actual_variety_jsonl"] = None


    # Create a tqdm progress bar
    # Iterate only up to the number of available prompts in jsonl_data or df length, whichever is smaller
    num_examples_to_process = len(df)
    if jsonl_data:
        num_examples_to_process = min(len(df), len(jsonl_data))
    else: # If jsonl_data is empty, process 0 examples that depend on it
        num_examples_to_process = 0 
        print("Warning: No JSONL data loaded. No examples will be processed for variety prediction.")

    with tqdm(total=num_examples_to_process, desc=f"Processing rows for model {model}") as progress_bar:
        # Process each example sequentially
        for i in range(num_examples_to_process):
            index = df.index[i] # Get the original DataFrame index
            row = df.iloc[i]    # Get the row data
            try:
                tokens = process_example(index, row, model, df, progress_bar) # Pass df index
                total_tokens_processed += tokens
            except Exception as e:
                # tqdm.write rather than print to avoid messing up progress bar
                tqdm.write(f"Error processing example at original index {index}: {str(e)}")
                progress_bar.update(1) # Ensure progress bar updates even on error within process_example's try-except

    return df, total_tokens_processed


def get_accuracy(model, df):
    if "actual_variety_jsonl" not in df.columns:
        print(f"Warning: 'actual_variety_jsonl' column not found in DataFrame for model {model}. Cannot calculate accuracy.")
        return 0.0
    if (model + "-variety") not in df.columns:
        print(f"Warning: '{model}-variety' column (predictions) not found in DataFrame for model {model}. Cannot calculate accuracy.")
        return 0.0
    
    # Filter out rows where actual_variety_jsonl might be None (e.g. if jsonl was shorter than df)
    relevant_df = df.dropna(subset=["actual_variety_jsonl", model + "-variety"])
    if relevant_df.empty:
        print(f"Warning: No valid data to compare for model {model} after filtering NaNs. Accuracy is 0.")
        return 0.0
        
    return np.mean(relevant_df["actual_variety_jsonl"] == relevant_df[model + "-variety"])


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
    
    processed_df = df_country_subset.copy() # Start with a copy

    for model in models_to_use:
        print(f"Processing with {model}...")
        # df_model_processed is the dataframe with new columns for this model
        df_model_processed, total_tokens = process_dataframe(processed_df, model)
        
        # Update processed_df with the results from this model
        # This ensures that columns from previous models are preserved if needed,
        # and new columns like actual_variety_jsonl and model-specific predictions are added/updated.
        processed_df = df_model_processed 
        
        accuracy = get_accuracy(model, processed_df)
        
        # Determine sample size based on actual processed entries from JSONL data
        # This could be len(jsonl_data) or the number of rows successfully processed up to that limit.
        # Using num_examples_to_process from the last call to process_dataframe might be tricky due to scope.
        # A more robust way: count non-NaN entries in the actual_variety_jsonl column for sample size if it's always filled by process_example.
        # Or, simpler for now, use min(len(df_country_subset), len(jsonl_data)) if jsonl_data loaded.
        current_sample_size = 0
        if jsonl_data:
            current_sample_size = min(len(df_country_subset), len(jsonl_data))
        
        results[model] = {
            "accuracy": accuracy,
            "sample_size": current_sample_size, # Reflects the number of examples attempted based on jsonl
            "country": COUNTRY, # This might need re-evaluation if data source changes fundamentally
            "total_tokens": total_tokens,
        }
        print(f"{model} accuracy: {accuracy * 100:.2f}%")
        print(f"{model} total tokens: {total_tokens}")

    return processed_df, results


if __name__ == "__main__":
    if not jsonl_data:
        print("Exiting: No data loaded from JSONL file. Ensure 'test.jsonl' is present and valid.")
    else:
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


