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
                                save_json, load_json, load_category_info,
                                parse_foodsam_mask_labels)
from utils.n5k_utils import get_dish_ids_and_splits # parse_dish_metadata_csv is used by output_generator
from utils.foodsam_handler import generate_direct_foodsam_outputs
from utils.output_generator import save_n5k_ground_truth_metadata
from utils.alignment_utils import (
    generate_aligned_n5k_metadata_with_scores,
    generate_n5k_semseg_from_foodsam_pred,
    generate_n5k_bboxes_from_sam_and_semseg
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
    foodsam_id_to_n5k_map = loaded_resources['foodsam_id_to_n5k_map']
    n5k_id_to_name = loaded_resources['n5k_id_to_name']
    n5k_string_id_to_int_id_map = loaded_resources['n5k_string_id_to_int_id_map']
    n5k_int_id_to_name_map = loaded_resources['n5k_int_id_to_name_map']

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
       not intermediate_foodsam_paths.get("raw_semantic_pred_path") or \
       not intermediate_foodsam_paths.get("sam_mask_label_path"):
        logging.error(f"FoodSAM intermediate modality generation failed for dish {dish_id}. Skipping.")
        return False
    raw_sam_masks_path = intermediate_foodsam_paths["masks_npy_path"]
    raw_foodsam103_semseg_path = intermediate_foodsam_paths["raw_semantic_pred_path"]
    foodsam_mask_labels_path = intermediate_foodsam_paths["sam_mask_label_path"]

    # --- 3. Save original N5K metadata as ground truth in intermediate dir ---
    original_n5k_metadata = save_n5k_ground_truth_metadata(dish_id, split_name)
    if not original_n5k_metadata:
        logging.error(f"Failed to save or parse original N5K metadata for dish {dish_id}. Skipping.")
        return False

    # --- 4. Parse FoodSAM's detected category observations (from sam_mask_labels.txt) ---
    # This replaces extract_foodsam_instances_with_masks for the purpose of metadata generation.
    raw_foodsam_observations = parse_foodsam_mask_labels(foodsam_mask_labels_path)
    if not raw_foodsam_observations:
        logging.warning(f"[{dish_id}] No FoodSAM category observations parsed from {foodsam_mask_labels_path}. Proceeding with empty observations.")
        # Metadata generation will likely result in 0 ingredients, which is handled.

    # Assign a unique observation_index to each, as the same FoodSAM category can be listed multiple times.
    foodsam_category_observations_with_indices = [
        {**obs, 'observation_index': i} for i, obs in enumerate(raw_foodsam_observations)
    ]

    # --- 5. Generate N5K-aligned metadata.json (with scores) and FoodSAM->N5K map ---
    aligned_metadata_content, foodsam_cat_id_to_final_n5k_int_id_map = \
        generate_aligned_n5k_metadata_with_scores(
            dish_id=dish_id,
            original_n5k_metadata=original_n5k_metadata, 
            foodsam_category_observations=foodsam_category_observations_with_indices,
            foodsam_id_to_n5k_map=foodsam_id_to_n5k_map,
            n5k_string_id_to_int_id_map=n5k_string_id_to_int_id_map,
            n5k_id_to_name_map=n5k_id_to_name
        )

    if not aligned_metadata_content or foodsam_cat_id_to_final_n5k_int_id_map is None: # Map can be {} but not None
        logging.error(f"Failed to generate N5K-aligned metadata or FoodSAM-N5K map for dish {dish_id}. Skipping.")
        return False
    final_metadata_path = os.path.join(final_modality_dirs[MODALITY_METADATA_N5K], f"{dish_id}.json")
    save_json(aligned_metadata_content, final_metadata_path)

    # --- 6. Generate N5K-aligned semseg.png (by recoloring FoodSAM's raw semantic prediction) ---
    n5k_semseg_image = generate_n5k_semseg_from_foodsam_pred(
        raw_foodsam103_semseg_path=raw_foodsam103_semseg_path,
        foodsam_cat_id_to_final_n5k_int_id_map=foodsam_cat_id_to_final_n5k_int_id_map,
        n5k_int_id_to_name_map=n5k_int_id_to_name_map
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
        # Create a blank image as a placeholder if generation fails, to maintain dataset structure
        # This helps avoid issues with downstream tools expecting a file.
        final_semseg_path = os.path.join(final_modality_dirs[MODALITY_SEMSEG_N5K], f"{dish_id}.png")
        try:
            blank_semseg = np.zeros((image_h, image_w), dtype=np.uint8)
            cv2.imwrite(final_semseg_path, blank_semseg)
            logging.info(f"Saved BLANK N5K semseg placeholder to {final_semseg_path}")
        except Exception as e_imwrite_blank:
            logging.error(f"Failed to save BLANK N5K semseg placeholder {final_semseg_path}: {e_imwrite_blank}")

    # --- 7. Generate N5K-aligned bounding_box.json (using SAM masks for geometry, new N5K semseg for labels) ---
    # The n5k_semseg_image here is the one generated in step 6.
    n5k_bbox_content = generate_n5k_bboxes_from_sam_and_semseg(
        dish_id=dish_id, 
        raw_sam_masks_npy_path=raw_sam_masks_path, 
        n5k_semseg_image=n5k_semseg_image,
        n5k_int_id_to_name_map=n5k_int_id_to_name_map
    )
    if n5k_bbox_content and n5k_bbox_content.get("instances") is not None:
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
    parser.add_argument("--n5k_root", type=str, default=None, \
                        help=f"Path to the root of the Nutrition5k dataset (default from config: {config.N5K_ROOT}).")
    parser.add_argument("--output_root", type=str, default=None, \
                        help=f"Path to save the processed dataset (default from config: {config.PROCESSED_DATA_DIR}).")
    parser.add_argument("--foodsam_dir", type=str, default=None,\
                        help=f"Path to the FoodSAM library directory (default from config: {config.FOODSAM_DIR}).")
    parser.add_argument("--intermediate_output_root", type=str, default=None, \
                        help=f"Path to save intermediate FoodSAM outputs (default from config: {config.INTERMEDIATE_FOODSAM_OUTPUT_DIR}).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of dishes to process (for testing).")
    parser.add_argument("--starting_from", type=str, default=None, help="Dish ID to start processing from (skips dishes before it in the ordered list).")

    args = parser.parse_args()

    if args.n5k_root: config.N5K_ROOT = args.n5k_root
    if args.output_root: config.PROCESSED_DATA_DIR = args.output_root
    if args.foodsam_dir: config.FOODSAM_DIR = args.foodsam_dir
    if args.intermediate_output_root: config.INTERMEDIATE_FOODSAM_OUTPUT_DIR = args.intermediate_output_root
    
    # Ensure FoodSAM checkpoints are correctly pointed to based on the (potentially overridden) FOODSAM_DIR
    config.SAM_CHECKPOINT = os.path.join(config.FOODSAM_DIR, "ckpts", "sam_vit_h_4b8939.pth")
    # SEMANTIC_CHECKPOINT_FILENAME and SEMANTIC_CONFIG_FILENAME are relative to FOODSAM_DIR in foodsam_handler
    
    # --- Setup Logging ---
    log_file_path = os.path.join(config.PROCESSED_DATA_DIR, "processing_log.txt")
    create_dir_if_not_exists(os.path.dirname(log_file_path))
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
                            logging.StreamHandler()
                        ])

    # --- Prepare processed_ids.txt tracker ---
    processed_ids_log_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_ids.txt")
    # Create/clear the file at the beginning of the run
    with open(processed_ids_log_path, 'w', encoding='utf-8') as f_processed_log:
        f_processed_log.write("# dish_id status\n") # Header
    logging.info(f"Tracking processed dish IDs in: {processed_ids_log_path}")

    # --- Pre-load all necessary mapping files and category info --- 
    global LOADED_RESOURCES
    try:
        LOADED_RESOURCES['foodsam103_id_to_name'], LOADED_RESOURCES['foodsam103_name_to_id'] = \
            load_category_info(config.FOODSAM103_CATEGORY_TXT)
        LOADED_RESOURCES['n5k_id_to_name'], LOADED_RESOURCES['n5k_name_to_id'] = \
            load_category_info(config.N5K_CATEGORY_TXT)
        LOADED_RESOURCES['foodsam_id_to_n5k_map'] = load_json(config.FOODSAM_TO_N5K_MAPPING_JSON)

        # Create N5K string ID to int ID map, and int ID to name map
        n5k_string_id_to_int_id = {}
        n5k_int_id_to_name = {}
        if LOADED_RESOURCES['n5k_id_to_name']:
            for str_id, name in LOADED_RESOURCES['n5k_id_to_name'].items():
                try:
                    # N5K string IDs are like "ingr_0000000001", "dish_0000000001"
                    # Or sometimes just integer strings from mapping files "1", "25"
                    if isinstance(str_id, str) and str_id.startswith("ingr_"):
                        int_id = int(str_id.split('_')[-1])
                        n5k_string_id_to_int_id[str_id] = int_id
                        n5k_int_id_to_name[int_id] = name
                    elif isinstance(str_id, (int, str)): # Handle cases where str_id might be '25' from mapping
                        int_id = int(str_id)
                        # We need a canonical string ID to map back if needed, or ensure n5k_id_to_name also has int keys
                        # For n5k_id_to_name (loaded from N5K_CATEGORY_TXT), keys are already the string IDs like "ingr_..."
                        # So, n5k_int_id_to_name is the primary map for int ID -> name.
                        n5k_int_id_to_name[int_id] = name
                        # Create a string_id_to_int_id mapping if the original key was just an int string
                        if isinstance(str_id, str) and str_id.isdigit():
                            n5k_string_id_to_int_id[f"ingr_{int(str_id):010d}"] = int_id # Store a formatted one too
                            n5k_string_id_to_int_id[str_id] = int_id # And the plain one
                        elif isinstance(str_id, int):
                            n5k_string_id_to_int_id[f"ingr_{str_id:010d}"] = int_id

                except ValueError:
                    logging.warning(f"Could not parse int ID from N5K string ID '{str_id}' while building maps.")
        LOADED_RESOURCES['n5k_string_id_to_int_id_map'] = n5k_string_id_to_int_id
        LOADED_RESOURCES['n5k_int_id_to_name_map'] = n5k_int_id_to_name

        if not LOADED_RESOURCES['foodsam103_id_to_name'] or \
           not LOADED_RESOURCES['n5k_id_to_name'] or \
           LOADED_RESOURCES['foodsam_id_to_n5k_map'] is None or \
           not LOADED_RESOURCES['n5k_string_id_to_int_id_map'] or \
           not LOADED_RESOURCES['n5k_int_id_to_name_map']:
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

    # --- Get all dish IDs and their splits, now in a defined order ---
    ordered_dish_processing_list = get_dish_ids_and_splits()

    if not ordered_dish_processing_list:
        logging.error("No dishes found to process. Exiting.")
        return

    # Apply limit if specified
    if args.limit is not None:
        logging.info(f"Limiting processing to the first {args.limit} dishes from the ordered list.")
        ordered_dish_processing_list = ordered_dish_processing_list[:args.limit]

    # Apply starting_from if specified
    if args.starting_from is not None:
        start_dish_id = args.starting_from
        try:
            # Find the index of the starting_from dish_id
            # The list contains tuples (dish_id, split_name)
            start_index = next(i for i, (d_id, _) in enumerate(ordered_dish_processing_list) if d_id == start_dish_id)
            logging.info(f"Starting processing from dish_id: {start_dish_id} at index {start_index}.")
            ordered_dish_processing_list = ordered_dish_processing_list[start_index:]
        except StopIteration: # Raised if next() doesn't find the item
            logging.warning(f"Dish ID '{start_dish_id}' specified with --starting_from not found in the current processing list. Processing will start from the beginning of the current list.")

    logging.info(f"Starting processing for {len(ordered_dish_processing_list)} dishes.")

    # --- Process each dish ---
    for dish_id, split_name in tqdm(ordered_dish_processing_list, desc="Processing Dishes"):
        success = process_single_dish(dish_id, split_name, config.PROCESSED_DATA_DIR, LOADED_RESOURCES)
        status_message = "successful" if success else "failed"
        with open(processed_ids_log_path, 'a', encoding='utf-8') as f_processed_log:
            f_processed_log.write(f"{dish_id} {status_message}\n")

    logging.info("Finished processing all dishes.")

if __name__ == "__main__":
    main() 