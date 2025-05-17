"""
Main script to process the Nutrition5k dataset and generate a 4M-compatible dataset.

Pipeline for each dish:
1. Identify dish and its split (train/test).
2. Create output directory: data/processed/<split>/<dish_id>/.
3. Copy rgb.png and depth_color.png from Nutrition5k.
4. Generate FoodSAM related modalities directly (semseg.png, masks.npy, sam_mask_label.txt):
   - Outputs to a temporary dish-specific directory (e.g., data/tmp_foodsam_outputs/<dish_id>/).
   - Key outputs: enhanced_mask.png, masks.npy, sam_mask_label.txt.
5. Retrieve directly generated FoodSAM outputs:
   - Copy enhanced_mask.png to <output_dir>/semseg.png.
   - Copy masks.npy and sam_mask_label.txt to <output_dir>/ for intermediate processing (_temp_masks.npy, _temp_sam_metadata.csv).
6. Generate sam_instance.npy:
   - Convert _temp_masks.npy (raw boolean masks) to polygon format.
   - Save as <output_dir>/sam_instance.npy.
7. Generate bounding_box.json:
   - Use _temp_masks.npy and _temp_sam_metadata.csv (which was originally sam_mask_label.txt).
   - Save as <output_dir>/bounding_box.json.
8. Generate metadata.json:
   - Parse Nutrition5k's dish_metadata_cafeX.csv for the dish.
   - Save as <output_dir>/metadata.json.
9. Clean up temporary files for the dish (e.g. _temp_ files in output_dir and the temp FoodSAM output dir for this dish).
"""
import os
import argparse
import logging
import shutil
from tqdm import tqdm

# Ensure utils can be imported if script is run from src/data_processing/
if __name__ == "__main__" and __package__ is None:
    import sys
    # Correctly add the script's directory (src/data_processing/) to sys.path.
    # This allows `from utils import ...` to work, as 'utils' is a subdirectory.
    # os.path.dirname(__file__) will be src/data_processing/
    script_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(script_dir)

from utils import config
from utils.common_utils import create_dir_if_not_exists, copy_file
from utils.n5k_utils import get_dish_ids_and_splits
from utils.foodsam_handler import (
    generate_direct_foodsam_outputs,
    retrieve_direct_foodsam_outputs,
    convert_sam_masks_to_polygons,
    generate_bounding_boxes_json
)
from utils.output_generator import generate_final_metadata_json

def process_single_dish(dish_id, split_name, dish_output_base_dir):
    """
    Processes a single dish to generate all required modalities.

    Args:
        dish_id (str): The ID of the dish.
        split_name (str): The split name ('train' or 'test').
        dish_output_base_dir (str): The base output directory for this dish (e.g., data/processed/train/dish_XXXX)

    Returns:
        bool: True if processing was successful for this dish, False otherwise.
    """
    logging.info(f"Processing dish: {dish_id} for split: {split_name}")
    create_dir_if_not_exists(dish_output_base_dir)

    # --- 1. Paths for Nutrition5k source files ---
    n5k_dish_image_dir = os.path.join(config.N5K_IMAGERY_DIR, dish_id)
    src_rgb_path = os.path.join(n5k_dish_image_dir, "rgb.png")
    src_depth_path = os.path.join(n5k_dish_image_dir, "depth_color.png")

    if not os.path.exists(src_rgb_path):
        logging.error(f"RGB image not found for dish {dish_id} at {src_rgb_path}")
        return False
    if not os.path.exists(src_depth_path):
        logging.warning(f"Depth image not found for dish {dish_id} at {src_depth_path}. Proceeding without it.")

    # --- 2. Copy basic image files ---    
    copy_file(src_rgb_path, os.path.join(dish_output_base_dir, "rgb.png"))
    if os.path.exists(src_depth_path):
        copy_file(src_depth_path, os.path.join(dish_output_base_dir, "depth_color.png"))

    # --- 3. Generate FoodSAM related modalities directly ---    
    temp_foodsam_dish_outputs_dir = os.path.join(config.TEMP_FOODSAM_OUTPUT_DIR, dish_id)
    create_dir_if_not_exists(temp_foodsam_dish_outputs_dir)

    direct_foodsam_gen_paths = generate_direct_foodsam_outputs(dish_id, src_rgb_path, temp_foodsam_dish_outputs_dir)
    
    if not direct_foodsam_gen_paths:
        logging.error(f"Direct FoodSAM modalitity generation failed for dish {dish_id}. Skipping further processing.")
        if os.path.exists(temp_foodsam_dish_outputs_dir):
             shutil.rmtree(temp_foodsam_dish_outputs_dir)
        return False

    # --- 4. Retrieve FoodSAM outputs (semseg.png, and intermediate _temp_masks.npy, _temp_sam_metadata.csv) ---
    retrieved_paths = retrieve_direct_foodsam_outputs(dish_id, temp_foodsam_dish_outputs_dir, dish_output_base_dir)
    if not retrieved_paths:
        logging.error(f"Failed to retrieve FoodSAM outputs for dish {dish_id}.")
        if os.path.exists(temp_foodsam_dish_outputs_dir):
             shutil.rmtree(temp_foodsam_dish_outputs_dir)
        return False
    
    raw_masks_intermediate_path = retrieved_paths.get("raw_masks_intermediate_path")
    sam_metadata_intermediate_path = retrieved_paths.get("sam_metadata_intermediate_path")

    # --- 5. Generate sam_instance.npy ---    
    if raw_masks_intermediate_path and os.path.exists(raw_masks_intermediate_path):
        sam_instance_output_path = os.path.join(dish_output_base_dir, "sam_instance.npy")
        sam_conversion_success = convert_sam_masks_to_polygons(raw_masks_intermediate_path, sam_instance_output_path)
        if not sam_conversion_success:
            logging.error(f"Failed to convert SAM masks to polygons for dish {dish_id}.")
    else:
        logging.warning(f"Skipping SAM instance generation for {dish_id} as raw masks path is missing or invalid.")

    # --- 6. Generate bounding_box.json ---    
    condition_raw_masks_ok = raw_masks_intermediate_path and os.path.exists(raw_masks_intermediate_path)
    condition_sam_meta_ok = sam_metadata_intermediate_path and os.path.exists(sam_metadata_intermediate_path)

    if condition_raw_masks_ok and condition_sam_meta_ok:
        bbox_output_path = os.path.join(dish_output_base_dir, "bounding_box.json")
        bbox_gen_success = generate_bounding_boxes_json(raw_masks_intermediate_path, 
                                                        sam_metadata_intermediate_path, 
                                                        bbox_output_path)
        if not bbox_gen_success:
            logging.error(f"Failed to generate bounding_box.json for dish {dish_id}.")
    else:
        logging.warning(f"Skipping bounding box generation for {dish_id} due to missing intermediate files (raw_masks_ok: {condition_raw_masks_ok}, sam_meta_ok: {condition_sam_meta_ok}).")

    # --- 7. Generate metadata.json ---    
    metadata_gen_success = generate_final_metadata_json(dish_id, dish_output_base_dir)
    if not metadata_gen_success:
        logging.error(f"Failed to generate metadata.json for dish {dish_id}.")

    # --- 8. Clean up intermediate files from dish_output_base_dir and the temp FoodSAM output dir for this dish ---
    logging.debug(f"Cleaning up intermediate files for {dish_id}...")
    if raw_masks_intermediate_path and os.path.exists(raw_masks_intermediate_path):
        try: os.remove(raw_masks_intermediate_path) 
        except OSError as e: logging.warning(f"Could not remove temp file {raw_masks_intermediate_path}: {e}")
    if sam_metadata_intermediate_path and os.path.exists(sam_metadata_intermediate_path):
        try: os.remove(sam_metadata_intermediate_path) 
        except OSError as e: logging.warning(f"Could not remove temp file {sam_metadata_intermediate_path}: {e}")
    
    if os.path.exists(temp_foodsam_dish_outputs_dir):
        try: shutil.rmtree(temp_foodsam_dish_outputs_dir) 
        except OSError as e: logging.warning(f"Could not remove temp FoodSAM dir {temp_foodsam_dish_outputs_dir}: {e}")

    logging.info(f"Successfully processed dish: {dish_id}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process Nutrition5k dataset for 4M model compatibility.")
    parser.add_argument("--n5k_root", type=str, default=None, 
                        help=f"Path to the root of the Nutrition5k dataset (default from config: {config.N5K_ROOT}).")
    parser.add_argument("--output_root", type=str, default=None, 
                        help=f"Path to save the processed dataset (default from config: {config.PROCESSED_DATA_DIR}).")
    parser.add_argument("--foodsam_dir", type=str, default=None,
                        help=f"Path to the FoodSAM library directory (default from config: {config.FOODSAM_DIR}). This is where FoodSAM tools and checkpoints are expected.")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help=f"Path for temporary FoodSAM outputs per dish (default from config: {config.TEMP_FOODSAM_OUTPUT_DIR}).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of dishes to process (for testing).")

    args = parser.parse_args()

    if args.n5k_root: config.N5K_ROOT = args.n5k_root
    if args.output_root: config.PROCESSED_DATA_DIR = args.output_root
    if args.foodsam_dir: config.FOODSAM_DIR = args.foodsam_dir
    if args.temp_dir: config.TEMP_FOODSAM_OUTPUT_DIR = args.temp_dir
    
    config.SAM_CHECKPOINT = os.path.join(config.FOODSAM_DIR, "ckpts", "sam_vit_h_4b8939.pth")

    create_dir_if_not_exists(config.PROCESSED_DATA_DIR)
    create_dir_if_not_exists(config.TEMP_FOODSAM_OUTPUT_DIR)
    for split in ["train", "test"]:
        create_dir_if_not_exists(os.path.join(config.PROCESSED_DATA_DIR, split))

    logging.info(f"Starting dataset processing.")
    logging.info(f"Nutrition5k Root: {config.N5K_ROOT}")
    logging.info(f"Processed Data Output: {config.PROCESSED_DATA_DIR}")
    logging.info(f"FoodSAM Directory (for tools/ckpts): {config.FOODSAM_DIR}")
    logging.info(f"Temporary FoodSAM Output per dish: {config.TEMP_FOODSAM_OUTPUT_DIR}")

    dish_id_to_split = get_dish_ids_and_splits()
    if not dish_id_to_split:
        logging.error("No dish IDs found or splits could not be determined. Exiting.")
        return

    dish_ids_to_process = list(dish_id_to_split.keys())
    if args.limit:
        dish_ids_to_process = dish_ids_to_process[:args.limit]
        logging.info(f"Processing a limit of {args.limit} dishes.")

    processed_count = 0
    failed_count = 0
    for dish_id in tqdm(dish_ids_to_process, desc="Processing Dishes"):
        split_name = dish_id_to_split[dish_id]
        dish_final_output_dir = os.path.join(config.PROCESSED_DATA_DIR, split_name, dish_id)
        
        success = process_single_dish(dish_id, split_name, dish_final_output_dir)
        if success:
            processed_count += 1
        else:
            failed_count += 1
            logging.error(f"Failed to process dish {dish_id}.")
    
    logging.info(f"Dataset processing complete.")
    logging.info(f"Successfully processed dishes: {processed_count}")
    logging.info(f"Failed dishes: {failed_count}")

if __name__ == "__main__":
    main() 