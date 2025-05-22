import pandas as pd
import numpy as np
import os
import argparse

def analyze_processed_ids(file_path):
    """
    Analyzes the processed_ids.txt file to extract and print various statistics.

    Args:
        file_path (str): The path to the processed_ids.txt file.
    """
    # Verify if the file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} not found. Please check the path.")
        return

    print(f"File {file_path} found. Starting analysis...")

    column_names = [
        'dish_id', 'status',
        'final_confidence', 'nutritional_score', 'ingredient_score',
        'msre_nutritional', 'jaccard_ingredient', 'nb_ingredient'
    ]

    try:
        # Read the file, ignoring lines starting with '#' (comments/header)
        df = pd.read_csv(file_path, comment='#', delim_whitespace=True, names=column_names, header=None)
        print("File loaded successfully.")

        # Filter for successfully processed dishes
        successful_df = df[df['status'] == 'successful'].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        if successful_df.empty:
            print("No successfully processed dishes found in the file.")
            return
            
        print(f"{len(successful_df)} successfully processed dishes found.")

        # Convert metric columns to numeric. Errors will become NaN.
        metric_columns = ['final_confidence', 'nutritional_score', 'ingredient_score', 'msre_nutritional', 'jaccard_ingredient', 'nb_ingredient']
        for col in metric_columns:
            successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')

        # Remove rows where conversion failed for metrics (NaN)
        original_count = len(successful_df)
        successful_df.dropna(subset=metric_columns, inplace=True)
        cleaned_count = len(successful_df)
        if original_count > cleaned_count:
            print(f"{original_count - cleaned_count} rows were removed due to non-numeric values in metric columns.")
        
        if successful_df.empty:
            print("No dishes with valid metrics after cleaning.")
            return

        print(f"{len(successful_df)} dishes with valid metrics after cleaning.")

        # --- Means and Medians of Metrics ---
        print("\nStatistics for successfully processed dishes:\n")

        mean_stats = successful_df[metric_columns].mean()
        median_stats = successful_df[metric_columns].median()

        print("Means:")
        for metric, value in mean_stats.items():
            print(f"- {metric}: {value:.4f}")

        print("\nMedians:")
        for metric, value in median_stats.items():
            print(f"- {metric}: {value:.4f}")

        # --- Filtered Statistics ---
        print("\nStatistics for filtered dishes (final_confidence >= 0.65 AND nb_ingredient > 1):\n")
        
        filtered_stats_df = successful_df[
            (successful_df['final_confidence'] >= 0.65) &
            (successful_df['nb_ingredient'] > 1)
        ]

        count_filtered_dishes = len(filtered_stats_df)
        print(f"- Number of matching dishes: {count_filtered_dishes}")

        if count_filtered_dishes > 0:
            print("\n  Average metrics for these filtered dishes:")
            filtered_mean_stats = filtered_stats_df[metric_columns].mean()
            for metric, value in filtered_mean_stats.items():
                print(f"  - {metric}: {value:.4f}")
        else:
            print("  No dishes match the filter criteria for detailed average metrics.")

    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found during read attempt.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: File {file_path} is empty or contains no columns to parse after ignoring comments.")
    except Exception as e:
        print(f"An error occurred while reading or processing the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the processed_ids.txt file and display statistics.")
    
    # Default path. 
    # If the script is in src/analysis/ and processed_ids.txt is in data/processed/
    # (i.e., ../../data/processed/processed_ids.txt from src/analysis/)
    default_file_path = "../../data/processed/processed_ids.txt" 
    
    # To use the specific path you mentioned, uncomment and modify the following line:
    # default_file_path = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"

    parser.add_argument(
        "--file_path", 
        type=str, 
        default=default_file_path,
        help="Path to the processed_ids.txt file"
    )
    args = parser.parse_args()
    
    # Determine which path to use (argument, default, or a user-specified fixed path)
    # By default, we use the user-specified absolute path if no other logic is uncommented.
    path_to_analyze = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"

    # Option 1: Use the path passed as an argument or the relative default path
    # If you uncomment this section, comment out the "Option 2" section
    # path_to_analyze = args.file_path 
    # print(f"Analyzing file (via argument or relative default): {path_to_analyze}")

    # Option 2: Use the hardcoded absolute path (the one you provided)
    # If you use this section, ensure argparse lines are either commented out,
    # or you understand that path_to_analyze will be overwritten here.
    # path_to_analyze = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"
    # print(f"Analyzing file (hardcoded absolute path): {path_to_analyze}")

    # Final execution line with the determined path
    print(f"Analyzing file: {path_to_analyze}") # Keep one print for clarity on which path is used.
    analyze_processed_ids(path_to_analyze) 