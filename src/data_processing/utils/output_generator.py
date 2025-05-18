"""
Generates the final metadata.json file for each processed dish.
Also handles saving of ground truth N5K metadata to an intermediate location.
"""
import os
import logging
from . import config
from .common_utils import save_json, create_dir_if_not_exists
from .n5k_utils import parse_dish_metadata_csv


def save_n5k_ground_truth_metadata(dish_id, split_name):
    """
    Saves the original Nutrition5k metadata for a dish to an intermediate 
    directory for later verification or comparison.
    The saved file will be named <dish_id>.json.

    Args:
        dish_id (str): The ID of the dish.
        split_name (str): The split name ('train' or 'test').

    Returns:
        dict: The parsed original N5K metadata, or None if parsing failed.
    """
    original_n5k_dish_info = parse_dish_metadata_csv(dish_id)

    if not original_n5k_dish_info:
        logging.error(f"Could not retrieve original Nutrition5k metadata for dish {dish_id}. Skipping ground truth save.")
        return None

    # Construct the path for the intermediate ground truth metadata
    # INTERMEDIATE_FOODSAM_OUTPUT_DIR / <split_name> / INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME / <dish_id> / <dish_id>.json
    gt_metadata_dish_dir = os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, 
                                        split_name, 
                                        config.INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME, 
                                        dish_id)
    create_dir_if_not_exists(gt_metadata_dish_dir)
    output_gt_json_path = os.path.join(gt_metadata_dish_dir, f"{dish_id}.json")

    # The format from parse_dish_metadata_csv is already good for saving as JSON.
    # It includes dish totals and the ingredients list with all their N5K attributes.
    try:
        save_json(original_n5k_dish_info, output_gt_json_path)
        logging.info(f"Successfully saved original N5K metadata (ground truth) for {dish_id} to {output_gt_json_path}")
        return original_n5k_dish_info
    except Exception as e:
        logging.error(f"Failed to save original N5K metadata for {dish_id}: {e}")
        return None


# The old `generate_final_metadata_json` is now superseded by the alignment logic 
# in alignment_utils.py for the *final* metadata.json. 
# This file now primarily focuses on handling the *original* N5K metadata.

# Example usage (for testing this module independently):
if __name__ == '__main__':
    # This requires n5k_utils.parse_dish_metadata_csv to work, which needs N5K data.
    # And config.py to be set up.
    # test_dish_id = "dish_1563207364" # Example dish ID
    # test_split = "train"
    
    # # Setup dummy intermediate directory structure if it doesn't exist
    # dummy_intermediate_root = os.path.join(config.PROJECT_ROOT, "data", "tmp_intermediate_outputs_test")
    # config.INTERMEDIATE_FOODSAM_OUTPUT_DIR = dummy_intermediate_root # Override for test
    # create_dir_if_not_exists(os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, test_split, config.INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME, test_dish_id))

    # parsed_gt_meta = save_n5k_ground_truth_metadata(test_dish_id, test_split)
    # if parsed_gt_meta:
    #     print(f"Test original N5K metadata saved and parsed for {test_dish_id}")
    # else:
    #     print(f"Failed to save/parse test original N5K metadata for {test_dish_id}")
    pass
