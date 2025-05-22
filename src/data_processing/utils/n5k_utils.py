"""
Utilities for parsing Nutrition5k dataset files, primarily for determining dish splits
and reading original N5K metadata.
"""
import os
import csv
import logging
from . import config
from .common_utils import read_text_file_lines

def get_dish_ids_and_splits():
    """
    Loads all dish IDs and determines their train/test split based on specified files.
    The order of processing is: 
    1. Dish IDs from `depth_train_ids.txt` (in file order).
    2. Dish IDs from `depth_test_ids.txt` (in file order, if not already in train).
    3. Remaining dish IDs from `dish_ids_all.txt` (alphabetical order, assigned to 'train' by default).

    Returns:
        list: A list of tuples (dish_id, split_name), ordered as described above.
              Example: [('dish_123', 'train'), ('dish_456', 'test'), ...]
    """
    all_dish_ids_path = os.path.join(config.N5K_SPLITS_DIR, "..", "dish_ids_all.txt") # Path to all dish IDs.
    train_ids_path = os.path.join(config.N5K_SPLITS_DIR, "depth_train_ids.txt") # Train IDs file.
    test_ids_path = os.path.join(config.N5K_SPLITS_DIR, "depth_test_ids.txt")   # Test IDs file.

    all_ids_set = set(read_text_file_lines(all_dish_ids_path))
    # Read train and test IDs, preserving their order from the files.
    train_ids_ordered = read_text_file_lines(train_ids_path)
    test_ids_ordered = read_text_file_lines(test_ids_path)

    ordered_dish_processing_list = []
    processed_ids_set = set() # To track IDs already added to the list, ensuring uniqueness.

    # Add train IDs in their original file order.
    for dish_id in train_ids_ordered:
        if dish_id in all_ids_set and dish_id not in processed_ids_set:
            ordered_dish_processing_list.append((dish_id, "train"))
            processed_ids_set.add(dish_id)
        elif dish_id not in all_ids_set:
            logging.warning(f"Dish ID {dish_id} from train splits file not found in all_ids_set. Skipping.")
        # If dish_id was already processed (e.g., duplicate in train_ids_ordered), it's implicitly skipped.

    # Add test IDs in their original file order, if not already processed as train.
    for dish_id in test_ids_ordered:
        if dish_id in all_ids_set and dish_id not in processed_ids_set:
            ordered_dish_processing_list.append((dish_id, "test"))
            processed_ids_set.add(dish_id)
        elif dish_id in processed_ids_set: # Already processed (likely as train).
             logging.warning(f"Dish ID {dish_id} from test splits file was already processed (e.g., in train). Kept with its initial split assignment.")
        elif dish_id not in all_ids_set:
            logging.warning(f"Dish ID {dish_id} from test splits file not found in all_ids_set. Skipping.")
        # If dish_id was already processed (e.g., duplicate in test_ids_ordered), it's implicitly skipped.

    # Add any remaining IDs from all_ids_set that weren't in specific train/test files.
    # These are sorted alphabetically for a deterministic fallback order.
    remaining_ids = sorted(list(all_ids_set - processed_ids_set))
    for dish_id in remaining_ids:
        logging.warning(f"Dish ID {dish_id} from all_ids_set not found in specific train/test depth splits. Assigning to 'train' and processing last.")
        ordered_dish_processing_list.append((dish_id, "train"))
        # processed_ids_set.add(dish_id) # Not strictly necessary here as remaining_ids is from a set difference.
            
    if not ordered_dish_processing_list and all_ids_set: # If all_ids_set had content but nothing matched splits.
        logging.error(f"No processable dish IDs after checking splits, though all_ids_set was not empty. Check split files: {train_ids_path}, {test_ids_path}")
    elif not all_ids_set:
        logging.error(f"No dish IDs found in all_ids_set (from {all_dish_ids_path}).")
        
    return ordered_dish_processing_list

def parse_dish_metadata_csv(dish_id):
    """
    Parses Nutrition5k's dish_metadata_cafeX.csv files to retrieve nutritional information for a specific dish.
    It checks both cafe1 and cafe2 metadata files.

    Args:
        dish_id (str): The ID of the dish (e.g., "dish_1556572657").

    Returns:
        dict or None: A dictionary containing dish totals and a list of ingredients if found and parsed successfully,
                      otherwise None. The dictionary structure includes keys like `dish_total_calories_kcal`,
                      `dish_total_mass_g`, and an `ingredients` list, where each ingredient is a dict
                      with `ingredient_id`, `ingredient_name`, `weight_g`, and nutritional values.
    """
    # Nutrition5k has two main metadata files; try both.
    metadata_files = [
        os.path.join(config.N5K_METADATA_DIR, "dish_metadata_cafe1.csv"),
        os.path.join(config.N5K_METADATA_DIR, "dish_metadata_cafe2.csv")
    ]

    dish_data = None
    for file_path in metadata_files:
        if not os.path.exists(file_path):
            logging.debug(f"Metadata file not found, skipping: {file_path}") # Debug as this is expected for one file.
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Skip header row
                if not header:
                    logging.warning(f"Metadata file is empty or has no header: {file_path}")
                    continue

                for row in reader:
                    if not row or row[0] != dish_id:
                        continue
                    
                    # Found the dish, parse its data.
                    dish_data = {
                        "dish_total_calories_kcal": float(row[1]),
                        "dish_total_mass_g": float(row[2]),
                        "dish_total_fat_g": float(row[3]),
                        "dish_total_carbs_g": float(row[4]),
                        "dish_total_protein_g": float(row[5]),
                        "ingredients": []
                    }
                    
                    num_fixed_cols = 6 # dish_id, total_kcal, total_mass, total_fat, total_carb, total_protein
                    # Each ingredient block has 7 fields.
                    if (len(row) - num_fixed_cols) % 7 != 0:
                        logging.error(f"Malformed ingredient data for dish {dish_id} in {file_path}. Row length: {len(row)}. Non-standard ingredient field count.")
                        # Continue to parse what might be possible, or return None if strict parsing is needed.

                    num_ingredients = (len(row) - num_fixed_cols) // 7
                    for i in range(num_ingredients):
                        base_idx = num_fixed_cols + (i * 7)
                        try:
                            ingredient_info = {
                                "ingredient_id": row[base_idx].strip(),
                                "ingredient_name": row[base_idx + 1].strip(),
                                "weight_g": float(row[base_idx + 2]),
                                "calories_kcal": float(row[base_idx + 3]),
                                "fat_g": float(row[base_idx + 4]),
                                "carbs_g": float(row[base_idx + 5]),
                                "protein_g": float(row[base_idx + 6])
                            }
                            dish_data["ingredients"].append(ingredient_info)
                        except IndexError:
                            logging.error(f"IndexError while parsing ingredients for dish {dish_id} in {file_path}. Row: {row}, base_idx: {base_idx}. Incomplete ingredient data.")
                            break # Stop processing ingredients for this row due to missing fields.
                        except ValueError as ve:
                            logging.error(f"ValueError for ingredient {i} of dish {dish_id} in {file_path}: {ve}. Data: {row[base_idx:base_idx+7]}")
                            continue # Skip this problematic ingredient, try next one.
                    return dish_data # Found and parsed the dish in this file.
        except Exception as e:
            logging.error(f"Critical error parsing metadata file {file_path} for dish {dish_id}: {e}", exc_info=True)
            return None # Stop if a critical error occurs with a file.
    
    if dish_data is None:
        logging.warning(f"Dish ID {dish_id} not found in any metadata CSVs ({metadata_files}).")
    return dish_data

# Example usage for standalone testing of this module.
if __name__ == '__main__':
    # Configure basic logging for the test.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    print("Testing get_dish_ids_and_splits...")
    # To test this properly, ensure dummy split files exist or N5K_SPLITS_DIR in config.py points to valid ones.
    splits = get_dish_ids_and_splits()
    if splits:
        print(f"Found {len(splits)} dishes. First 5: {splits[:5]}")
        print(f"Last 5: {splits[-5:]}")
    else:
        print("get_dish_ids_and_splits returned empty or None.")

    print("\nTesting parse_dish_metadata_csv...")
    # Example test; replace with a valid dish_id from your N5K metadata files.
    # Ensure N5K_METADATA_DIR in config.py points to valid CSV files.
    test_dish_id_example = "dish_1563207364" # Replace with a real ID from your data for testing.
    # metadata = parse_dish_metadata_csv(test_dish_id_example)
    # if metadata:
    #     import json # For pretty printing
    #     print(f"Successfully parsed metadata for {test_dish_id_example}:")
    #     print(json.dumps(metadata, indent=2))
    # else:
    #     print(f"Could not parse metadata for {test_dish_id_example}.")
    print(f"(parse_dish_metadata_csv test commented out, requires actual data/dish ID: {test_dish_id_example})")
    pass
