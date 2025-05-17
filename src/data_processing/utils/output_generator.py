"""
Generates the final metadata.json file for each processed dish.
"""
import os
import logging
from . import config
from .common_utils import save_json
from .n5k_utils import parse_dish_metadata_csv

def generate_final_metadata_json(dish_id, output_dir):
    """
    Generates the metadata.json file for a given dish ID using parsed Nutrition5k data.

    Args:
        dish_id (str): The ID of the dish.
        output_dir (str): The directory where metadata.json will be saved 
                          (e.g., data/processed/split/dish_id/).

    Returns:
        bool: True if metadata.json was successfully generated, False otherwise.
    """
    n5k_dish_info = parse_dish_metadata_csv(dish_id)

    if not n5k_dish_info:
        logging.error(f"Could not retrieve Nutrition5k metadata for dish {dish_id}. Skipping metadata.json generation.")
        return False

    # Construct the metadata dictionary in the desired format
    # User format:
    # {
    #   "dish_total_calories_kcal": ..., "dish_total_fat_g": ..., 
    #   "dish_total_carbs_g": ..., "dish_total_protein_g": ...,
    #   "ingredients": [
    #     {"ingredient_name": ..., "weight_g": ..., "calories_kcal": ..., 
    #      "fat_g": ..., "carbs_g": ..., "protein_g": ...}, ...
    #   ]
    # }
    # Our n5k_dish_info format from n5k_utils.parse_dish_metadata_csv is already very close.

    output_metadata = {
        "dish_total_calories_kcal": n5k_dish_info.get("dish_total_calories_kcal"),
        "dish_total_fat_g": n5k_dish_info.get("dish_total_fat_g"),
        "dish_total_carbs_g": n5k_dish_info.get("dish_total_carbs_g"),
        "dish_total_protein_g": n5k_dish_info.get("dish_total_protein_g"),
        "ingredients": n5k_dish_info.get("ingredients", [])
    }

    # Optional: Add dish_total_mass_g if desired, though not in user's final spec
    # output_metadata["dish_total_mass_g"] = n5k_dish_info.get("dish_total_mass_g")

    output_json_path = os.path.join(output_dir, "metadata.json")
    try:
        save_json(output_metadata, output_json_path)
        logging.info(f"Successfully generated metadata.json for {dish_id} at {output_json_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save metadata.json for {dish_id}: {e}")
        return False

# Example usage (for testing this module independently):
if __name__ == '__main__':
    # This requires n5k_utils.parse_dish_metadata_csv to work, which needs N5K data.
    # test_dish_id = "dish_1563207364" # Example dish ID
    # temp_output_dir = os.path.join(config.PROJECT_ROOT, "data", "scripts", "utils", "test_output", test_dish_id)
    # common_utils.create_dir_if_not_exists(temp_output_dir)
    # success = generate_final_metadata_json(test_dish_id, temp_output_dir)
    # if success:
    #     print(f"Test metadata.json generated in {temp_output_dir}")
    # else:
    #     print(f"Failed to generate test metadata.json for {test_dish_id}")
    pass
