"""
Utilities for handling and saving original Nutrition5k (N5K) metadata.
This module focuses on saving the ground truth N5K metadata for a dish to an intermediate location,
which can be used for verification, comparison, or as input to alignment processes.
"""
import os
import logging
from . import config
from .common_utils import save_json, create_dir_if_not_exists
from .n5k_utils import parse_dish_metadata_csv


def save_n5k_ground_truth_metadata(dish_id, split_name):
    """
    Parses and saves the original Nutrition5k metadata for a given dish to a structured
    intermediate directory. The saved file is named `<dish_id>.json`.

    This function is crucial for providing the ground truth against which aligned or
    predicted metadata can be compared, and for supplying the N5K ingredient details
    needed during the alignment process.

    Args:
        dish_id (str): The ID of the dish (e.g., "dish_1556572657").
        split_name (str): The data split to which the dish belongs (e.g., 'train', 'test').
                         Used to organize the intermediate output path.

    Returns:
        dict or None: The parsed original N5K metadata dictionary if successful, 
                      otherwise None if parsing or saving fails.
    """
    original_n5k_dish_info = parse_dish_metadata_csv(dish_id)

    if not original_n5k_dish_info:
        logging.error(f"Could not retrieve original Nutrition5k metadata for dish {dish_id}. Ground truth save skipped.")
        return None

    # Define the path for saving the intermediate ground truth metadata.
    # Path structure: INTERMEDIATE_FOODSAM_OUTPUT_DIR / <split_name> / <gt_metadata_dir_name> / <dish_id> / <dish_id>.json
    gt_metadata_dish_dir = os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, 
                                        split_name, 
                                        config.INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME, 
                                        dish_id)
    create_dir_if_not_exists(gt_metadata_dish_dir)
    output_gt_json_path = os.path.join(gt_metadata_dish_dir, f"{dish_id}.json")

    # The dictionary format from parse_dish_metadata_csv is suitable for direct JSON saving.
    try:
        save_json(original_n5k_dish_info, output_gt_json_path)
        logging.info(f"Successfully saved original N5K metadata (ground truth) for {dish_id} to {output_gt_json_path}.")
        return original_n5k_dish_info
    except Exception as e:
        logging.error(f"Failed to save original N5K metadata for {dish_id} to {output_gt_json_path}: {e}", exc_info=True)
        return None


# Note: The generation of the *final* processed metadata.json (which includes alignment and scores)
# is handled by `generate_aligned_n5k_metadata_with_scores` in `alignment_utils.py`.
# This file (`output_generator.py`) is now primarily for managing the *original* N5K metadata.

# Example usage for standalone testing of this module.
if __name__ == '__main__':
    # Basic configuration for logging during tests.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # To run this test, ensure that:
    # 1. `config.py` is correctly set up (especially N5K_ROOT and INTERMEDIATE_FOODSAM_OUTPUT_DIR).
    # 2. `n5k_utils.parse_dish_metadata_csv` can access actual N5K metadata CSV files.
    # 3. You provide a valid `test_dish_id` that exists in your N5K dataset.

    test_dish_id = "dish_1563207364"  # Replace with a valid dish ID from your dataset.
    test_split = "train"

    # Optional: Override INTERMEDIATE_FOODSAM_OUTPUT_DIR for testing to avoid writing to main data folders.
    # dummy_intermediate_root = os.path.join(config.PROJECT_ROOT, "data", "tmp_test_intermediate_outputs")
    # original_intermediate_dir = config.INTERMEDIATE_FOODSAM_OUTPUT_DIR
    # config.INTERMEDIATE_FOODSAM_OUTPUT_DIR = dummy_intermediate_root
    # print(f"Overriding intermediate output dir for test: {dummy_intermediate_root}")

    print(f"\nAttempting to save and parse ground truth metadata for dish: {test_dish_id}, split: {test_split}")
    parsed_gt_meta = save_n5k_ground_truth_metadata(test_dish_id, test_split)

    if parsed_gt_meta:
        print(f"Successfully saved and parsed original N5K metadata for {test_dish_id}.")
        # print(f"Preview: {str(parsed_gt_meta)[:200]}...") # Uncomment for a brief preview
    else:
        print(f"Failed to save or parse original N5K metadata for {test_dish_id}.")
    
    # # Restore original config if overridden for test
    # if 'original_intermediate_dir' in locals():
    #     config.INTERMEDIATE_FOODSAM_OUTPUT_DIR = original_intermediate_dir
    #     # Consider cleaning up dummy_intermediate_root if created.
    print("\nTest finished. Check logs and the intermediate directory for results.")
    pass
