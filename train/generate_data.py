import os
import pandas as pd
from config import COUNTRY
import json
import random
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Generate training data for wine variety classification"
)
parser.add_argument(
    "--all-countries",
    action="store_true",
    help="Process wines from all countries instead of filtering by country",
)
args = parser.parse_args()

# Load the wine dataset
print("Loading wine dataset...")
df = pd.read_csv("data/winemag-data-130k-v2.csv")

# Apply country filter only if --all-countries is not set
if not args.all_countries:
    print(f"Filtering for wines from {COUNTRY}...")
    df_filtered = df[df["country"] == COUNTRY]
else:
    print("Processing wines from all countries...")
    df_filtered = df

# Filter out wines with less than 5 references of their grape variety
print("Filtering rare varieties...")
varieties_less_than_five_list = (
    df_filtered["variety"]
    .value_counts()[df_filtered["variety"].value_counts() < 5]
    .index.tolist()
)
df_filtered = df_filtered[~df_filtered["variety"].isin(varieties_less_than_five_list)]


def create_wine_sample(row):
    # Handle potential missing values with empty strings to avoid None errors
    winery = row.get("winery", "")
    province = row.get("province", "")
    country = row.get("country", "")
    region_1 = row.get("region_1", "")
    description = row.get("description", "")
    taster_name = row.get("taster_name", "")
    points = row.get("points", "")
    price = row.get("price", "")

    prompt = f"""/no_think\nBased on this wine review, guess the grape variety:
This wine is produced by {winery} in the {province} region of {country}.
It was grown in {region_1}. It is described as: "{description}".
The wine has been reviewed by {taster_name} and received {points} points.
The price is {price}"""

    return {"prompt": prompt, "completion": f"<think>\n\n</think>{{\"variety\": \"{row['variety']}\"}}"}


print("Processing wine data...")
all_data = [create_wine_sample(row) for _, row in df_filtered.iterrows()]
random.shuffle(all_data)

# Calculate split sizes
total_size = len(all_data)
train_size = int(0.8 * total_size)
test_size = int(0.1 * total_size)
valid_size = total_size - train_size - test_size

# Split the data
new_train_data = all_data[:train_size]
new_test_data = all_data[train_size : train_size + test_size]
new_valid_data = all_data[train_size + test_size :]


# Write to JSONL files
def write_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


print("Writing train.jsonl...")
folder_prefix = "./train/data/"
os.makedirs(folder_prefix, exist_ok=True)
write_jsonl(new_train_data, folder_prefix + "train.jsonl")
print("Writing test.jsonl...")
write_jsonl(new_test_data, folder_prefix + "test.jsonl")
print("Writing valid.jsonl...")
write_jsonl(new_valid_data, folder_prefix + "valid.jsonl")

print(
    f"Dataset split and saved: train ({len(new_train_data)}), test ({len(new_test_data)}), valid ({len(new_valid_data)})"
)


# Verify file contents
def count_lines(filename):
    with open(folder_prefix + filename, "r") as f:
        return sum(1 for _ in f)


print("\nVerifying file contents:")
print(f"train.jsonl: {count_lines('train.jsonl')} lines")
print(f"test.jsonl: {count_lines('test.jsonl')} lines")
print(f"valid.jsonl: {count_lines('valid.jsonl')} lines")
