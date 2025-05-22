"""
Utilities for parsing Nutrition5k dataset files.
"""
import os
import csv
import logging
from . import config
from .common_utils import read_text_file_lines

def get_dish_ids_and_splits():
    """
    Loads all dish IDs and determines their train/test split.
    The order of processing will be: all train_ids in order, then all test_ids in order.
    Dishes in all_ids but not in specific train/test files will be processed last, assigned to 'train'.

    Returns:
        list: A list of tuples, where each tuple is (dish_id, split_name),
              ordered by train split first, then test split, then fallbacks.
    """
    all_dish_ids_path = os.path.join(config.N5K_SPLITS_DIR, "..", "dish_ids_all.txt") # one level up from splits
    train_ids_path = os.path.join(config.N5K_SPLITS_DIR, "depth_train_ids.txt")
    test_ids_path = os.path.join(config.N5K_SPLITS_DIR, "depth_test_ids.txt")

    all_ids_set = set(read_text_file_lines(all_dish_ids_path))
    # Read train and test IDs, preserving their order from the files
    train_ids_ordered = read_text_file_lines(train_ids_path)
    test_ids_ordered = read_text_file_lines(test_ids_path)

    ordered_dish_processing_list = []
    processed_ids_set = set() # To keep track of IDs already added to the list

    # Add train IDs in their original order
    for dish_id in train_ids_ordered:
        if dish_id in all_ids_set and dish_id not in processed_ids_set:
            ordered_dish_processing_list.append((dish_id, "train"))
            processed_ids_set.add(dish_id)
        elif dish_id not in all_ids_set:
            logging.warning(f"Dish ID {dish_id} from depth_train_ids.txt not found in dish_ids_all.txt. Skipping.")
        # If dish_id was already processed (e.g. duplicate in train_ids_ordered), it's skipped here.

    # Add test IDs in their original order
    for dish_id in test_ids_ordered:
        if dish_id in all_ids_set and dish_id not in processed_ids_set:
            ordered_dish_processing_list.append((dish_id, "test"))
            processed_ids_set.add(dish_id)
        elif dish_id in train_ids_ordered: # Already processed as train
             logging.warning(f"Dish ID {dish_id} from depth_test_ids.txt was already listed in depth_train_ids.txt. Kept as 'train'.")
        elif dish_id not in all_ids_set:
            logging.warning(f"Dish ID {dish_id} from depth_test_ids.txt not found in dish_ids_all.txt. Skipping.")
        # If dish_id was already processed (e.g. duplicate in test_ids_ordered), it's skipped here.

    # Add any remaining IDs from all_ids_set that weren't in train/test specific files
    # These will be sorted alphabetically for deterministic fallback order
    remaining_ids = sorted(list(all_ids_set - processed_ids_set))
    for dish_id in remaining_ids:
        logging.warning(f"Dish ID {dish_id} from dish_ids_all.txt not found in specific train/test depth splits. Assigning to 'train' and processing last.")
        ordered_dish_processing_list.append((dish_id, "train"))
        processed_ids_set.add(dish_id) # Should already be covered, but for safety
            
    if not ordered_dish_processing_list and all_ids_set: # If all_ids had content but nothing matched splits
        logging.error(f"No processable dish IDs after checking splits, though dish_ids_all.txt was not empty. Check split files: {train_ids_path}, {test_ids_path}")
    elif not all_ids_set:
        logging.error(f"No dish IDs found in dish_ids_all.txt. Check path: {all_dish_ids_path}")
        
    return ordered_dish_processing_list

def parse_dish_metadata_csv(dish_id):
    """
    Parses the dish_metadata_cafeX.csv file to retrieve nutritional information for a specific dish.
    Nutrition5k has two main metadata files, so we check both.

    Args:
        dish_id (str): The ID of the dish (e.g., "dish_1556572657").

    Returns:
        dict: A dictionary containing dish totals and a list of ingredients, 
              or None if the dish_id is not found or an error occurs.
              Format:
              {
                  "dish_total_calories_kcal": float,
                  "dish_total_fat_g": float,
                  "dish_total_carbs_g": float,
                  "dish_total_protein_g": float,
                  "dish_total_mass_g": float, # Added for completeness
                  "ingredients": [
                      {
                          "ingredient_id": str,
                          "ingredient_name": str,
                          "weight_g": float,
                          "calories_kcal": float,
                          "fat_g": float,
                          "carbs_g": float,
                          "protein_g": float
                      }, ...
                  ]
              }
    """
    # Determine which metadata file to search (cafe1 or cafe2)
    # This might require a lookup or trying both, as Nutrition5k doesn't explicitly map dish_id to cafe file directly.
    # For now, we try both.
    metadata_files = [
        os.path.join(config.N5K_METADATA_DIR, "dish_metadata_cafe1.csv"),
        os.path.join(config.N5K_METADATA_DIR, "dish_metadata_cafe2.csv")
    ]

    dish_data = None
    for file_path in metadata_files:
        if not os.path.exists(file_path):
            logging.warning(f"Metadata file not found: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0] != dish_id:
                        continue
                    
                    # Found the dish
                    dish_data = {
                        "dish_total_calories_kcal": float(row[1]),
                        "dish_total_mass_g": float(row[2]),
                        "dish_total_fat_g": float(row[3]),
                        "dish_total_carbs_g": float(row[4]),
                        "dish_total_protein_g": float(row[5]),
                        "ingredients": []
                    }
                    
                    num_fixed_cols = 6
                    # Each ingredient has 7 fields: id, name, grams, calories, fat, carb, protein
                    if (len(row) - num_fixed_cols) % 7 != 0:
                        logging.error(f"Malformed ingredient data for dish {dish_id} in {file_path}. Row length: {len(row)}. Expected (len - {num_fixed_cols}) % 7 == 0.")
                        # Try to parse what we can if it's just trailing empty cells
                        # return None # Or handle error more gracefully

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
                            logging.error(f"IndexError while parsing ingredients for dish {dish_id} in {file_path}. Expected more fields. Row: {row}, base_idx: {base_idx}")
                            break # Stop processing ingredients for this row
                        except ValueError as ve:
                            logging.error(f"ValueError while parsing ingredient {i} for dish {dish_id} in {file_path}: {ve}. Row segment: {row[base_idx:base_idx+7]}")
                            continue # Skip this ingredient
                    return dish_data # Found and parsed
        except Exception as e:
            logging.error(f"Error parsing metadata file {file_path} for dish {dish_id}: {e}")
            return None # Critical error during parsing of this file
    
    if dish_data is None:
        logging.warning(f"Dish ID {dish_id} not found in any metadata CSVs.")
    return dish_data

# Example usage (for testing this module independently):
if __name__ == '__main__':
    # Test get_dish_ids_and_splits
    # Ensure you have dummy files in expected locations if N5K_ROOT is not set for a real dataset
    # For example, create dummy versions of dish_ids_all.txt, rgb_train_ids.txt, rgb_test_ids.txt
    # in appropriate subdirectories of where this script might think N5K_ROOT is.
    # This will likely fail if config.py cannot resolve N5K_SPLITS_DIR correctly without the full project structure.
    
    print("Testing get_dish_ids_and_splits...")
    splits = get_dish_ids_and_splits()
    print(f"Found {len(splits)} dishes. First 5: {splits[:5]}")

    # Test parse_dish_metadata_csv
    # You would need actual or dummy dish_metadata_cafe1/2.csv files for this to work.
    # And a valid dish_id from those files.
    # Example: test_dish_id = "dish_1563207364" 
    # metadata = parse_dish_metadata_csv(test_dish_id)
    # if metadata:
    #     print(f"\nSuccessfully parsed metadata for {test_dish_id}:")
    #     print(json.dumps(metadata, indent=2))
    # else:
    #     print(f"\nCould not parse metadata for {test_dish_id}.")
    pass
