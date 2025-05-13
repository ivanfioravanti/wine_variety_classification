import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import COUNTRY, SAMPLE_SIZE, RANDOM_SEED
import glob
import argparse

# Import all wine analysis modules
from providers import wine_anthropic
from providers import wine_ollama
from providers import wine_openai
from providers import wine_gemini_genai
from providers import wine_openrouter
from providers import wine_lmstudio
from providers import wine_deepseek
from providers import wine_mlx_server_unstructured


def generate_chart(summary_df, timestamp):
    # Sort data by accuracy
    df_sorted = summary_df.sort_values(by="accuracy", ascending=False)

    # Get country from the first row's data
    country = (
        df_sorted["country"].iloc[0] if "country" in df_sorted.columns else "Unknown"
    )

    # Create the plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df_sorted.index, df_sorted["accuracy"] * 100, color="skyblue")

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.title(
        f"{country} Wines variety classification from reviews accuracy (Best to Worst)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the chart
    plt.savefig(f"results/accuracy_chart_{timestamp}.png")
    plt.close()


def generate_chart_from_summary(summary_path=None):
    """Generate a chart from an existing summary CSV file.
    If no path is provided, uses the most recent summary file in results folder."""
    if summary_path is None:
        # Find the most recent summary file
        summary_files = glob.glob("results/summary_*.csv")
        if not summary_files:
            print("No summary files found in results directory")
            return
        summary_path = max(summary_files, key=os.path.getctime)

    # Extract timestamp from filename
    timestamp = summary_path.split("summary_")[1].replace(".csv", "")

    # Read the summary CSV
    summary_df = pd.read_csv(summary_path, index_col="model")

    # Generate the chart
    generate_chart(summary_df, timestamp)
    print(f"Chart generated: results/accuracy_chart_{timestamp}.png")


def run_all_models():
    # Create a timestamp for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dictionary to store all results
    all_results = {}

    # List of all modules and their models
    module_models = [
        # (
        #     wine_anthropic,
        #     [
        #         "claude-3-5-haiku-20241022",
        #         "claude-3-5-sonnet-20241022",
        #         "claude-3-opus-20240229",
        #     ],
        # ),
        # (
        #     wine_ollama,
        #     [
        #         # "qwen3:235b",
        #         # "qwen3:30b",
        #         "qwen3:32b-q8_0",
        #         "qwen3:14b-q8_0",
        #         "qwen3:8b-q8_0",
        #         "qwen3:4b-q8_0",
        #         "qwen3:0.6b-q8_0",
        #         "qwen3:1.7b-q8_0"
        #     ],
        # ),
        # (wine_openai, ["gpt-4o", "gpt-4o-mini"]),
        # (
        #     wine_gemini_genai,
        #     ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
        # ),
        # (wine_openrouter, ["deepseek/deepseek-chat"]),
        # (wine_lmstudio, ["Llama-3.2-3B-Instruct-4bit"]),
        # (wine_deepseek, ["deepseek-chat"]),
        # (wine_mlx_server_unstructured, ["mlx-community/Qwen3-0.6B-8bit","mlx-community/Qwen3-0.6B-bf16","mlx-community/Qwen3-1.7B-4bit", "mlx-community/Qwen3-1.7B-8bit",
        #           "mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-4B-8bit", "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-8B-8bit",
        #           "mlx-community/Qwen3-14B-4bit", "mlx-community/Qwen3-14B-8bit", "mlx-community/Qwen3-30B-A3B-4bit", "mlx-community/Qwen3-30B-A3B-8bit", 
        #           "mlx-community/Qwen3-32B-4bit", "mlx-community/Qwen3-32B-8bit", "mlx-community/Qwen3-235B-A22B-4bit", "mlx-community/Qwen3-235B-A22B-8bit"]),        
        (wine_mlx_server_unstructured, ["mlx-community/Qwen3-8B-4bit-DWQ"]),
        # (wine_mlx_server_unstructured, ["mlx-community/gemma-3-27b-it-qat-4bit"]),
    ]

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Run each module's models
    for module, models in module_models:
        print(f"\nRunning {module.__name__}...")
        try:
            # Run the provider with specified models
            df, provider_results = module.run_provider(models)

            # Add results to all_results
            all_results.update(provider_results)

            # Save the detailed results for this module
            df.to_csv(f"results/{module.__name__}_{timestamp}.csv", index=False)

        except Exception as e:
            print(f"Error running {module.__name__}: {str(e)}")

    # Create a summary DataFrame
    summary_df = pd.DataFrame.from_dict(all_results, orient="index")
    summary_df.index.name = "model"
    summary_df = summary_df.sort_values("accuracy", ascending=False)

    # Save the summary
    summary_df.to_csv(f"results/summary_{timestamp}.csv")

    # Generate and save the chart
    generate_chart(summary_df, timestamp)

    # Print the final summary with accuracy in percentage format
    print("\nFinal Summary:")
    display_df = summary_df.copy()
    display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x*100:.2f}%")
    print(display_df.to_string())

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run wine variety prediction tests across different LLM providers"
    )
    parser.add_argument(
        "--generate-chart",
        action="store_true",
        help="Generate chart from most recent results without running new tests",
    )
    parser.add_argument(
        "--no-provider-csv",
        action="store_true",
        help="Disable saving individual provider results to CSV files",
    )
    parser.add_argument(
        "--summary",
        help="Specific summary file name to generate chart from (e.g., summary_20250105_095642.csv)",
    )
    args = parser.parse_args()

    if args.generate_chart:
        summary_path = f"results/{args.summary}" if args.summary else None
        generate_chart_from_summary(summary_path)
    else:
        run_all_models()
