"""
Main script to process the Nutrition5k dataset and generate a 4M-compatible dataset.
Aligned with N5K categories using FoodSAM intermediate outputs.

Pipeline for each dish:
1. Copy rgb.png and depth_color.png from Nutrition5k to final processed dir.
2. Generate FoodSAM intermediate outputs (raw SAM masks, raw FoodSAM-103 semantic prediction) 
   and save to INTERMEDIATE_FOODSAM_OUTPUT_DIR.
3. Save original N5K metadata as metadata_ground_truth.json in INTERMEDIATE_FOODSAM_OUTPUT_DIR.
4. Extract FoodSAM instances (raw mask + FoodSAM-103 category) from intermediate outputs.
5. Generate N5K-aligned metadata.json (with ingredient alignment and confidence scores) and save to final processed dir.
   This involves mapping FoodSAM detections to N5K ingredients from original dish metadata.
6. Generate N5K-aligned semseg.png (using N5K category IDs) and save to final processed dir.
7. Generate N5K-aligned bounding_box.json (using N5K category IDs) and save to final processed dir.
8. Copy raw SAM masks (masks.npy) to be the final sam_instance.npy in processed dir.
"""
import os
import argparse
import logging
import shutil
from tqdm import tqdm
import cv2 # For imwrite in semseg
import numpy as np # For image shape

# Ensure utils can be imported if script is run from src/data_processing/
if __name__ == "__main__" and __package__ is None:
    import sys
    script_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(script_dir)
    # If utils is one level up from script_dir (i.e. in src/data_processing/)
    # sys.path.append(os.path.dirname(script_dir))

from utils import config
from utils.common_utils import (create_dir_if_not_exists, copy_file, 
                                save_json, load_json, load_category_info)
from utils.n5k_utils import get_dish_ids_and_splits # parse_dish_metadata_csv is used by output_generator
from utils.foodsam_handler import generate_direct_foodsam_outputs
from utils.output_generator import save_n5k_ground_truth_metadata
from utils.alignment_utils import (
    extract_foodsam_instances_with_masks,
    generate_aligned_n5k_metadata_with_scores,
    generate_n5k_semseg_from_sam,
    generate_n5k_bboxes_from_sam
)

# Define modality names as constants for clarity and consistency (final output modalities)
MODALITY_RGB = "rgb"
MODALITY_DEPTH = "depth_color"
MODALITY_SEMSEG_N5K = "semseg"       # Now N5K aligned
MODALITY_SAM_INSTANCE = "sam_instance" # Raw SAM masks
MODALITY_BBOX_N5K = "bounding_box"   # Now N5K aligned
MODALITY_METADATA_N5K = "metadata"   # Now N5K aligned, with scores

# Loaded global resources
LOADED_RESOURCES = {}

def process_single_dish(dish_id, split_name, processed_data_root_dir, loaded_resources):
    """
    Processes a single dish to generate all N5K-aligned modalities.

    Args:
        dish_id (str): The ID of the dish.
        split_name (str): The split name ('train' or 'test').
        processed_data_root_dir (str): The base output directory for final processed data.
        loaded_resources (dict): Pre-loaded mappings and category info.

    Returns:
        bool: True if processing was successful for this dish, False otherwise.
    """
    logging.info(f"Processing dish: {dish_id} for split: {split_name}")

    # --- Retrieve loaded resources ---
    foodsam103_id_to_name = loaded_resources['foodsam103_id_to_name']
    foodsam_name_to_n5k_names = loaded_resources['foodsam_to_n5k_map']
    n5k_id_to_name = loaded_resources['n5k_id_to_name']
    n5k_name_to_id = loaded_resources['n5k_name_to_id']

    # --- 1. Paths for Nutrition5k source files & Create Final Output Dirs ---
    n5k_dish_image_dir = os.path.join(config.N5K_IMAGERY_DIR, dish_id)
    src_rgb_path = os.path.join(n5k_dish_image_dir, "rgb.png")
    src_depth_path = os.path.join(n5k_dish_image_dir, "depth_color.png")

    if not os.path.exists(src_rgb_path):
        logging.error(f"RGB image not found for dish {dish_id} at {src_rgb_path}")
        return False
    # Depth image is optional, handled by copy_file if it exists

    # Create final output directories for each modality for this dish
    # Structure: <processed_data_root_dir>/<split_name>/<modality_name>/<dish_id>/
    final_modality_dirs = {}
    for mod_name in [MODALITY_RGB, MODALITY_DEPTH, MODALITY_SEMSEG_N5K, 
                     MODALITY_SAM_INSTANCE, MODALITY_BBOX_N5K, MODALITY_METADATA_N5K]:
        mod_dir = os.path.join(processed_data_root_dir, split_name, mod_name, dish_id)
        create_dir_if_not_exists(mod_dir)
        final_modality_dirs[mod_name] = mod_dir

    # --- 1.1 Copy base images (RGB, Depth) to final location ---
    final_rgb_path = os.path.join(final_modality_dirs[MODALITY_RGB], f"{dish_id}.png")
    copy_file(src_rgb_path, final_rgb_path)
    if os.path.exists(src_depth_path):
        final_depth_path = os.path.join(final_modality_dirs[MODALITY_DEPTH], f"{dish_id}.png")
        copy_file(src_depth_path, final_depth_path)
    
    # Get image shape from RGB for semseg generation later
    try:
        rgb_image_for_shape = cv2.imread(src_rgb_path)
        if rgb_image_for_shape is None: raise ValueError("Failed to load RGB for shape")
        image_h, image_w = rgb_image_for_shape.shape[:2]
    except Exception as e:
        logging.error(f"[{dish_id}] Could not read RGB image {src_rgb_path} to get shape: {e}")
        return False

    # --- 2. Generate FoodSAM intermediate outputs (raw SAM masks, raw FoodSAM-103 semantic) ---
    # These are saved into config.INTERMEDIATE_FOODSAM_OUTPUT_DIR
    intermediate_foodsam_paths = generate_direct_foodsam_outputs(dish_id, split_name, src_rgb_path)
    if not intermediate_foodsam_paths or \
       not intermediate_foodsam_paths.get("masks_npy_path") or \
       not intermediate_foodsam_paths.get("raw_semantic_pred_path"):
        logging.error(f"FoodSAM intermediate modality generation failed for dish {dish_id}. Skipping.")
        return False
    raw_sam_masks_path = intermediate_foodsam_paths["masks_npy_path"]
    raw_foodsam103_semseg_path = intermediate_foodsam_paths["raw_semantic_pred_path"]

    # --- 3. Save original N5K metadata as ground truth in intermediate dir ---
    original_n5k_metadata = save_n5k_ground_truth_metadata(dish_id, split_name)
    if not original_n5k_metadata:
        logging.error(f"Failed to save or parse original N5K metadata for dish {dish_id}. Skipping.")
        return False

    # --- 4. Extract FoodSAM instances (raw mask + FoodSAM-103 category) ---
    detected_foodsam_instances = extract_foodsam_instances_with_masks(
        raw_sam_masks_path, raw_foodsam103_semseg_path, foodsam103_id_to_name
    )
    if not detected_foodsam_instances: # Can be empty if no relevant instances found, but not a failure
        logging.warning(f"[{dish_id}] No FoodSAM instances extracted or extraction failed. Proceeding with potentially empty alignment.")
        # Depending on strictness, could return False here

    # --- 5. Generate N5K-aligned metadata.json (with scores) ---
    aligned_metadata_content, instance_to_n5k_assignment = \
        generate_aligned_n5k_metadata_with_scores(
            dish_id, original_n5k_metadata, detected_foodsam_instances,
            foodsam_name_to_n5k_names, n5k_name_to_id, n5k_id_to_name
        )
    if not aligned_metadata_content or instance_to_n5k_assignment is None: # instance_to_n5k_assignment can be {} but not None
        logging.error(f"Failed to generate N5K-aligned metadata for dish {dish_id}. Skipping.")
        return False
    final_metadata_path = os.path.join(final_modality_dirs[MODALITY_METADATA_N5K], f"{dish_id}.json")
    save_json(aligned_metadata_content, final_metadata_path)

    # --- 6. Generate N5K-aligned semseg.png ---
    n5k_semseg_image = generate_n5k_semseg_from_sam(
        raw_sam_masks_path, instance_to_n5k_assignment, (image_h, image_w), n5k_id_to_name
    )
    if n5k_semseg_image is not None:
        final_semseg_path = os.path.join(final_modality_dirs[MODALITY_SEMSEG_N5K], f"{dish_id}.png")
        try:
            cv2.imwrite(final_semseg_path, n5k_semseg_image)
            logging.info(f"Saved N5K semseg to {final_semseg_path}")
        except Exception as e_imwrite:
            logging.error(f"Failed to save N5K semseg image {final_semseg_path}: {e_imwrite}")
            # Not returning false, as other modalities might be okay.
    else:
        logging.warning(f"Failed to generate N5K semseg image for {dish_id}.")

    # --- 7. Generate N5K-aligned bounding_box.json ---
    n5k_bbox_content = generate_n5k_bboxes_from_sam(
        dish_id, raw_sam_masks_path, instance_to_n5k_assignment, n5k_id_to_name
    )
    if n5k_bbox_content:
        final_bbox_path = os.path.join(final_modality_dirs[MODALITY_BBOX_N5K], f"{dish_id}.json")
        save_json(n5k_bbox_content, final_bbox_path)
    else:
        logging.warning(f"Failed to generate N5K bounding boxes for {dish_id}.")

    # --- 8. Copy raw SAM masks (masks.npy) to be final sam_instance.npy ---
    # This is the direct output from SAM, representing instance segmentations without class labels here.
    final_sam_instance_path = os.path.join(final_modality_dirs[MODALITY_SAM_INSTANCE], f"{dish_id}.npy")
    copy_file(raw_sam_masks_path, final_sam_instance_path)

    logging.info(f"Successfully processed dish: {dish_id}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process Nutrition5k dataset for 4M model compatibility, aligning to N5K categories.")
    parser.add_argument("--n5k_root", type=str, default=None, 
                        help=f"Path to the root of the Nutrition5k dataset (default from config: {config.N5K_ROOT}).")
    parser.add_argument("--output_root", type=str, default=None, 
                        help=f"Path to save the processed dataset (default from config: {config.PROCESSED_DATA_DIR}).")
    parser.add_argument("--foodsam_dir", type=str, default=None,
                        help=f"Path to the FoodSAM library directory (default from config: {config.FOODSAM_DIR}).")
    parser.add_argument("--intermediate_output_root", type=str, default=None, 
                        help=f"Path to save intermediate FoodSAM outputs (default from config: {config.INTERMEDIATE_FOODSAM_OUTPUT_DIR}).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of dishes to process (for testing).")

    args = parser.parse_args()

    if args.n5k_root: config.N5K_ROOT = args.n5k_root
    if args.output_root: config.PROCESSED_DATA_DIR = args.output_root
    if args.foodsam_dir: config.FOODSAM_DIR = args.foodsam_dir
    if args.intermediate_output_root: config.INTERMEDIATE_FOODSAM_OUTPUT_DIR = args.intermediate_output_root
    
    # Ensure FoodSAM checkpoints are correctly pointed to based on the (potentially overridden) FOODSAM_DIR
    config.SAM_CHECKPOINT = os.path.join(config.FOODSAM_DIR, "ckpts", "sam_vit_h_4b8939.pth")
    # SEMANTIC_CHECKPOINT_FILENAME and SEMANTIC_CONFIG_FILENAME are relative to FOODSAM_DIR in foodsam_handler
    
    # --- Pre-load all necessary mapping files and category info --- 
    global LOADED_RESOURCES
    try:
        LOADED_RESOURCES['foodsam103_id_to_name'], LOADED_RESOURCES['foodsam103_name_to_id'] = \
            load_category_info(config.FOODSAM103_CATEGORY_TXT)
        LOADED_RESOURCES['n5k_id_to_name'], LOADED_RESOURCES['n5k_name_to_id'] = \
            load_category_info(config.N5K_CATEGORY_TXT)
        LOADED_RESOURCES['foodsam_to_n5k_map'] = load_json(config.FOODSAM_TO_N5K_MAPPING_JSON)

        if not LOADED_RESOURCES['foodsam103_id_to_name'] or \
           not LOADED_RESOURCES['n5k_id_to_name'] or \
           LOADED_RESOURCES['foodsam_to_n5k_map'] is None: # load_json returns None on error
            raise ValueError("One or more essential mapping/category files failed to load.")
        logging.info("Successfully pre-loaded all category and mapping files.")
    except Exception as e:
        logging.error(f"Fatal error: Could not load essential mapping/category files: {e}", exc_info=True)
        return

    create_dir_if_not_exists(config.PROCESSED_DATA_DIR)
    create_dir_if_not_exists(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR)
    for split in ["train", "test"]:
        create_dir_if_not_exists(os.path.join(config.PROCESSED_DATA_DIR, split))
        create_dir_if_not_exists(os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, split))
        # Create intermediate ground truth dir as well
        create_dir_if_not_exists(os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, split, config.INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME))


    logging.info(f"Starting dataset processing with N5K alignment.")
    logging.info(f"Nutrition5k Root: {config.N5K_ROOT}")
    logging.info(f"Processed Data Output (Final): {config.PROCESSED_DATA_DIR}")
    logging.info(f"FoodSAM Directory (for tools/ckpts): {config.FOODSAM_DIR}")
    logging.info(f"Intermediate Outputs (FoodSAM raw, N5K GT): {config.INTERMEDIATE_FOODSAM_OUTPUT_DIR}")

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
        
        success = process_single_dish(dish_id, split_name, config.PROCESSED_DATA_DIR, LOADED_RESOURCES)
        if success:
            processed_count += 1
        else:
            failed_count += 1
            logging.error(f"Failed to process dish {dish_id}.") # process_single_dish should log specifics
    
    logging.info(f"Dataset processing complete.")
    logging.info(f"Successfully processed dishes: {processed_count}")
    logging.info(f"Failed dishes: {failed_count}")

if __name__ == "__main__":
    main() 