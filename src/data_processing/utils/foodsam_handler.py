"""
Handler for running FoodSAM components, processing their outputs (masks, semantic info),
and preparing them for the main dataset generation pipeline.
This involves generating raw SAM masks, FoodSAM-103 semantic predictions, and associated label files.
"""
import os
import sys
import numpy as np
import cv2 # OpenCV for contour finding and bounding box calculation
import logging
import shutil
import csv # For reading sam_mask_label.txt
import torch

from . import config
from .common_utils import create_dir_if_not_exists, copy_file, save_json

# Helper to add FoodSAM to sys.path to allow importing its tools.
# This is necessary because FoodSAM tools are often designed to be run from its own directory.
def _add_foodsam_to_path():
    # config.FOODSAM_DIR points to the root of the cloned FoodSAM repository.
    # Adding this to sys.path allows importing packages like `segment_anything`,
    # `mmseg`, and the inner `FoodSAM` package directly.
    foodsam_clone_root = config.FOODSAM_DIR
    if foodsam_clone_root not in sys.path:
        sys.path.insert(0, foodsam_clone_root)

_add_foodsam_to_path()

# Now import FoodSAM tools. This might raise an error if FoodSAM is not set up correctly.
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmcv.parallel import MMDataParallel # For semantic model if run on GPU
    from mmcv.image import tensor2imgs
    import mmcv # For imread, imwrite if needed by save_result from semantic_predict
    from FoodSAM.FoodSAM_tools.predict_semantic_mask import save_result as semantic_save_result # Renamed to avoid conflict
    from FoodSAM.FoodSAM_tools.enhance_semantic_masks import calculate_single_image_masks_label, enhance_masks as enhance_masks_tool_func, visualization_save
except ImportError as e:
    logging.error(f"Failed to import FoodSAM tools. Ensure FoodSAM is correctly placed in {config.FOODSAM_DIR} and all dependencies are installed. Error: {e}")
    # Depending on how critical these are, you might raise e or have fallbacks.
    raise

# Global cache for SAM and Semantic Segmentation models to avoid reloading for each image.
_SAM_MODEL_CACHE = None
_SEMANTIC_MODEL_CACHE = None

# Names for intermediate subdirectories within the main intermediate output directory.
INTERMEDIATE_MASKS_NPY = "masks_npy" # For raw SAM masks (masks.npy)
INTERMEDIATE_SEMANTIC_PRED_RAW = "semantic_pred_raw" # For raw FoodSAM-103 semantic_pred.png
INTERMEDIATE_SAM_LABELS = "sam_mask_labels" # For sam_mask_label.txt (FoodSAM-103 categories for SAM masks)
INTERMEDIATE_ENHANCED_MASK = "enhanced_semantic_mask" # For enhanced_mask.png (FoodSAM-103 based, potentially colored)

def _get_sam_model():
    """Loads or retrieves the SAM model from cache."""
    global _SAM_MODEL_CACHE
    if _SAM_MODEL_CACHE is None:
        logging.info(f"Loading SAM model: {config.MODEL_TYPE_SAM} from {config.SAM_CHECKPOINT}")
        model = sam_model_registry[config.MODEL_TYPE_SAM](checkpoint=config.SAM_CHECKPOINT)
        _SAM_MODEL_CACHE = model.to(device=config.FOODSAM_DEVICE)
        _SAM_MODEL_CACHE.eval()
    return _SAM_MODEL_CACHE

def _get_semantic_model():
    """Loads or retrieves the Semantic Segmentation model from cache."""
    global _SEMANTIC_MODEL_CACHE
    if _SEMANTIC_MODEL_CACHE is None:
        # Ensure FoodSAM_DIR is in path for relative config/checkpoint paths.
        cfg_path = os.path.join(config.FOODSAM_DIR, config.SEMANTIC_CONFIG_FILENAME)
        ckpt_path = os.path.join(config.FOODSAM_DIR, config.SEMANTIC_CHECKPOINT_FILENAME)
        logging.info(f"Loading Semantic Segmentation model from config: {cfg_path} and checkpoint: {ckpt_path}")
        model = init_segmentor(cfg_path, ckpt_path, device=config.FOODSAM_DEVICE)
        if config.FOODSAM_DEVICE != 'cpu': # MMDataParallel is for GPU.
             model = MMDataParallel(model, device_ids=[0]) # Assuming single GPU.
        model.eval()
        _SEMANTIC_MODEL_CACHE = model
    return _SEMANTIC_MODEL_CACHE

def generate_direct_foodsam_outputs(dish_id, split_name, rgb_image_path):
    """
    Generates core FoodSAM outputs: raw SAM masks, raw FoodSAM-103 semantic prediction,
    SAM mask labels (FoodSAM-103 categories for each SAM mask), and an enhanced semantic mask.
    Saves these to a structured intermediate directory.

    Args:
        dish_id (str): The ID of the dish.
        split_name (str): The split name ('train' or 'test').
        rgb_image_path (str): Path to the input RGB image.

    Returns:
        dict: A dictionary containing paths to the key generated intermediate files if successful,
              otherwise None. Keys include: "enhanced_mask_path", "masks_npy_path", 
              "sam_mask_label_path", "raw_semantic_pred_path".
    """
    logging.info(f"Starting FoodSAM processing for {dish_id} (split: {split_name}) -> intermediate structured output.")

    # Base output directories for this dish's intermediate files.
    base_intermediate_split_dir = os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, split_name)

    # Create dish-specific directories for each intermediate modality.
    masks_npy_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_MASKS_NPY, dish_id)
    pred_mask_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_SEMANTIC_PRED_RAW, dish_id)
    sam_labels_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_SAM_LABELS, dish_id)
    enhanced_mask_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_ENHANCED_MASK, dish_id)

    for dir_path in [masks_npy_dish_dir, pred_mask_dish_dir, sam_labels_dish_dir, enhanced_mask_dish_dir]:
        create_dir_if_not_exists(dir_path)

    # Define full paths for the output files.
    masks_npy_path = os.path.join(masks_npy_dish_dir, f"{dish_id}.npy")
    pred_mask_png_path = os.path.join(pred_mask_dish_dir, f"{dish_id}.png") # Raw semantic prediction
    sam_mask_label_path = os.path.join(sam_labels_dish_dir, f"{dish_id}.txt")
    enhanced_mask_path = os.path.join(enhanced_mask_dish_dir, f"{dish_id}.png")

    color_list_full_path = os.path.join(config.FOODSAM_DIR, config.FOODSAM_COLOR_LIST_PATH_FILENAME)
    category_txt_full_path = os.path.join(config.FOODSAM_DIR, config.FOODSAM_CATEGORY_TXT_FILENAME)

    try:
        # --- A. Initialize SAM and Semantic Segmentation Models ---
        sam_model = _get_sam_model()
        semantic_model = _get_semantic_model()

        # --- B. Generate Raw SAM Masks (masks.npy) ---
        logging.info(f"[{dish_id}] Generating raw SAM masks...")
        image = cv2.imread(rgb_image_path)
        if image is None:
            logging.error(f"Could not load image {rgb_image_path}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # AMG settings for SAM. These could be exposed via config.py if needed.
        mask_generator = SamAutomaticMaskGenerator(sam_model, points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100)
        sam_masks_data = mask_generator.generate(image_rgb)
        
        if not sam_masks_data:
            logging.warning(f"[{dish_id}] SAM did not generate any masks for {rgb_image_path}")
            np.save(masks_npy_path, np.array([]).astype(bool).reshape(0, image_rgb.shape[0], image_rgb.shape[1]))
            with open(sam_mask_label_path, 'w') as f_sml:
                f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        else:
            # Save all generated masks. Sorting/filtering (e.g., by top_k) can be done later if needed.
            stacked_masks = np.array([m['segmentation'] for m in sam_masks_data], dtype=bool)
            np.save(masks_npy_path, stacked_masks)
            logging.info(f"[{dish_id}] Saved {len(stacked_masks)} raw SAM masks to {masks_npy_path}")

        # --- C. Generate Semantic Prediction (pred_mask.png with FoodSAM-103 class IDs) ---
        logging.info(f"[{dish_id}] Generating semantic prediction...")
        with torch.no_grad(): # Ensure no gradients for inference.
            model_for_inference = semantic_model
            if isinstance(semantic_model, MMDataParallel):
                model_for_inference = semantic_model.module
            semantic_result = inference_segmentor(model_for_inference, rgb_image_path)

        if not semantic_result or len(semantic_result) == 0:
            logging.error(f"[{dish_id}] Semantic inference failed or returned empty result for {rgb_image_path}.")
            return None

        try:
            # Save the raw semantic mask (FoodSAM-103 class IDs).
            # semantic_result[0] is the segmentation map from inference_segmentor.
            raw_semantic_mask = semantic_result[0].astype(np.uint8)
            mmcv.imwrite(raw_semantic_mask, pred_mask_png_path)
            logging.info(f"[{dish_id}] Saved raw semantic prediction to {pred_mask_png_path}")
        except Exception as e_save:
            logging.error(f"[{dish_id}] Failed to save semantic prediction {pred_mask_png_path}: {e_save}", exc_info=True)
            return None

        # Ensure the file was actually created.
        if not os.path.exists(pred_mask_png_path):
            logging.error(f"[{dish_id}] {pred_mask_png_path} was not created after mmcv.imwrite call.")
            return None

        # --- D. Generate SAM Mask Labels (sam_mask_label.txt) ---
        # This uses the generated masks.npy and pred_mask.png to assign a FoodSAM-103 category
        # to each SAM mask based on majority voting within the mask area.
        logging.info(f"[{dish_id}] Assigning categories to SAM masks...")
        if not os.path.exists(masks_npy_path) or np.load(masks_npy_path).shape[0] == 0:
            logging.warning(f"[{dish_id}] masks.npy is empty or missing. Skipping SAM mask label generation.")
            with open(sam_mask_label_path, 'w') as f_sml:
                f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        else:
            # calculate_single_image_masks_label from FoodSAM_tools assigns a FoodSAM-103 category
            # to each raw SAM mask based on the raw semantic prediction.
            category_list_for_tool = []
            with open(category_txt_full_path, 'r') as f_cat:
                category_lines = f_cat.readlines()
                category_list_for_tool = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]

            calculate_single_image_masks_label(mask_file=masks_npy_path, 
                                             pred_mask_file=pred_mask_png_path,
                                             category_list=category_list_for_tool, 
                                             sam_mask_label_file_name=f"{dish_id}.txt",
                                             sam_mask_label_file_dir=sam_labels_dish_dir)
            
            if os.path.exists(sam_mask_label_path):
                logging.info(f"[{dish_id}] Saved SAM mask labels to {sam_mask_label_path}")
            else:
                logging.error(f"[{dish_id}] Failed to generate {sam_mask_label_path}.")
                with open(sam_mask_label_path, 'w') as f_sml:
                    f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        
        # --- E. Generate Enhanced Semantic Mask (enhanced_mask.png) ---
        # This step uses the FoodSAM_tools.enhance_masks logic to refine the semantic mask.
        logging.info(f"[{dish_id}] Generating enhanced semantic mask...")
        
        _pred_mask_img = cv2.imread(pred_mask_png_path)[:,:,2] # Assumes BGR, takes B channel (should be grayscale)
        _sam_masks_arr = np.load(masks_npy_path)
        _enhanced_mask_img = _pred_mask_img.copy()

        if os.path.exists(sam_mask_label_path):
            with open(sam_mask_label_path, 'r') as f_sml_enhance:
                category_info_lines = f_sml_enhance.readlines()[1:]
            
            # Sort and filter by top_k based on mask area ratio, as in enhance_masks_tool_func.
            valid_category_info = []
            for line in category_info_lines:
                parts = line.strip().split(',')
                if len(parts) == 5:
                    try: valid_category_info.append({'line': line, 'mask_area_ratio': float(parts[4])}) # parts[4] is mask_count_ratio
                    except ValueError: continue      
            
            category_info_lines_sorted = sorted(valid_category_info, key=lambda x: x['mask_area_ratio'], reverse=True)
            category_info_lines_top_k = [item['line'] for item in category_info_lines_sorted[:config.ENHANCE_MASKS_TOP_K]]

            if _sam_masks_arr.shape[0] > 0: # Check if there are any SAM masks
                for info_line in category_info_lines_top_k:
                    parts = info_line.strip().split(',')                    
                    idx, label_id_str, _, count_ratio_str, area_ratio_str = parts[0], parts[1], parts[2], parts[3], parts[4]
                    try:
                        idx = int(idx)
                        label_id = int(label_id_str)
                        count_ratio = float(count_ratio_str) # Ratio of pixels of this class within SAM mask
                        area_ratio = float(area_ratio_str) # Ratio of SAM mask area to total image area
                    except ValueError as ve:
                        logging.warning(f"[{dish_id}] Skipping malformed line in sam_mask_label.txt for enhance: {info_line} - {ve}")
                        continue

                    # Apply thresholds as in FoodSAM's enhance_masks tool.
                    if area_ratio * (image_rgb.shape[0] * image_rgb.shape[1]) < config.ENHANCE_MASKS_AREA_THR:
                        continue
                    if count_ratio < config.ENHANCE_MASKS_RATIO_THR:
                        continue
                    if idx >= _sam_masks_arr.shape[0]:
                        logging.warning(f"[{dish_id}] Mask index {idx} out of bounds for _sam_masks_arr during enhance. Max idx: {_sam_masks_arr.shape[0]-1}")
                        continue

                    sam_mask_slice = _sam_masks_arr[idx].astype(bool)
                    _enhanced_mask_img[sam_mask_slice] = label_id # Recolor based on assigned label

        # Save the enhanced mask (this will be a grayscale image with FoodSAM-103 IDs)
        cv2.imwrite(enhanced_mask_path, _enhanced_mask_img)
        logging.info(f"[{dish_id}] Saved enhanced semantic mask to {enhanced_mask_path}")

        return {
            "enhanced_mask_path": enhanced_mask_path,
            "masks_npy_path": masks_npy_path,
            "sam_mask_label_path": sam_mask_label_path,
            "raw_semantic_pred_path": pred_mask_png_path
        }

    except Exception as e:
        logging.error(f"[{dish_id}] Error during FoodSAM direct output generation: {e}", exc_info=True)
        return None


def convert_sam_masks_to_polygons(raw_masks_npy_path, output_polygon_npy_path):
    """
    Converts raw SAM boolean masks to polygon instances.
    Each instance is a dictionary with 'points' (a list of [x,y] coordinates)
    and 'original_mask_index'. This format might be used for specific tokenizers.

    Args:
        raw_masks_npy_path (str): Path to the .npy file containing raw SAM masks ((N, H, W) boolean array).
        output_polygon_npy_path (str): Path to save the .npy file of polygon instances.

    Returns:
        bool: True if conversion and saving were successful, False otherwise.
    """
    polygon_instances = []
    try:
        binary_masks = np.load(raw_masks_npy_path)
        if binary_masks.ndim == 0 or binary_masks.size == 0 : # Handle empty masks_npy
             logging.warning(f"Raw SAM masks file is empty or invalid: {raw_masks_npy_path}. Saving empty polygons.")
             np.save(output_polygon_npy_path, np.array([], dtype=object))
             return True
        if binary_masks.ndim != 3 or binary_masks.dtype != bool:
            logging.error(f"Invalid format for raw SAM masks: {raw_masks_npy_path}. Expected 3D bool array. Got {binary_masks.shape}, {binary_masks.dtype}")
            return False

        for i in range(binary_masks.shape[0]):
            mask_slice = binary_masks[i].astype(np.uint8) # findContours expects uint8
            contours, hierarchy = cv2.findContours(mask_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours or hierarchy is None:
                continue

            # Iterate through contours, keeping only outer ones (hierarchy[0][contour_idx][3] == -1)
            for contour_idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= config.MIN_CONTOUR_AREA_SAM_INSTANCE: # Filter by min area
                    is_outer = hierarchy[0][contour_idx][3] == -1 
                    if is_outer:
                        # .squeeze(axis=1) removes the redundant middle dimension from contour points (N,1,2) -> (N,2)
                        polygon_points_list = contour.squeeze(axis=1).tolist() 
                        polygon_instances.append({"points": polygon_points_list, "original_mask_index": i})
        
        np.save(output_polygon_npy_path, np.array(polygon_instances, dtype=object))
        logging.info(f"Converted SAM masks to polygons: {output_polygon_npy_path} with {len(polygon_instances)} instances.")
        return True
    except FileNotFoundError:
        logging.error(f"Mask file not found for polygon conversion: {raw_masks_npy_path}")
        return False
    except Exception as e:
        logging.error(f"Error converting SAM masks to polygons ({raw_masks_npy_path}): {e}", exc_info=True)
        return False

def generate_bounding_boxes_json(masks_npy_path, sam_mask_label_path, output_bbox_json_path):
    """
    Generates a bounding_box.json file with FoodSAM-103 category information.
    Bounding boxes are derived from SAM masks. Labels and scores come from sam_mask_label.txt.
    This output is FoodSAM-103 based, not N5K-aligned.

    Args:
        masks_npy_path (str): Path to the .npy file with raw SAM masks ((N, H, W) boolean array).
        sam_mask_label_path (str): Path to the .txt file with FoodSAM-103 labels for SAM masks.
        output_bbox_json_path (str): Path to save the output bounding_box.json.

    Returns:
        bool: True if successful, False otherwise.
    """
    bounding_boxes_output_list = [] # Renamed for clarity
    try:
        masks_data = np.load(masks_npy_path)
        if masks_data.ndim == 0 or masks_data.size == 0:
            logging.warning(f"masks.npy is empty for bounding box generation: {masks_npy_path}. Saving empty JSON.")
            save_json({"dish_id": os.path.basename(output_bbox_json_path).replace(".json", ""), "instances": []}, output_bbox_json_path)
            return True
        if masks_data.ndim != 3 or masks_data.dtype != bool:
            logging.error(f"Invalid masks_npy file for bbox: {masks_npy_path}. Got {masks_data.shape}, {masks_data.dtype}")
            return False

        mask_idx_to_meta = {} # For quick lookup of FoodSAM-103 category info per SAM mask index
        if not os.path.exists(sam_mask_label_path):
            logging.warning(f"sam_mask_label.txt not found: {sam_mask_label_path}. Bounding boxes will lack class names/scores.")
        else:
            with open(sam_mask_label_path, 'r') as f_labels:
                reader = csv.DictReader(f_labels)
                for row in reader:
                    try:
                        idx = int(row['id'])
                        # Store relevant info for this mask index
                        mask_idx_to_meta[idx] = {
                            'foodsam_category_id': int(row['category_id']),
                            'foodsam_category_name': row['category_name'],
                            'score': float(row.get('mask_count_ratio', config.DEFAULT_BBOX_SCORE)) # Using mask_count_ratio as a score
                        }
                    except (ValueError, KeyError) as e_parse:
                        logging.warning(f"Skipping malformed or incomplete line in {sam_mask_label_path}: {row} - {e_parse}")
                        continue

        for i in range(masks_data.shape[0]): # Iterate through each mask in masks.npy
            mask_slice = masks_data[i]
            if not np.any(mask_slice): # Skip empty masks
                continue
            
            # Default category if not found in sam_mask_label.txt or if file is missing
            category_name = "unknown_foodsam_category"
            foodsam_category_id_val = -1 # Default unknown ID
            score_val = config.DEFAULT_BBOX_SCORE

            meta_entry = mask_idx_to_meta.get(i)
            if meta_entry:
                category_name = meta_entry.get('foodsam_category_name', category_name)
                foodsam_category_id_val = meta_entry.get('foodsam_category_id', foodsam_category_id_val)
                score_val = meta_entry.get('score', score_val)
            
            # Skip if the assigned FoodSAM category is background (ID 0)
            if foodsam_category_id_val == 0:
                logging.debug(f"Skipping bbox for SAM mask {i} as its FoodSAM category is background (ID 0).")
                continue
            
            contours, _ = cv2.findContours(mask_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            # Standard COCO format: [x_min, y_min, width, height]
            bbox_coco_format = [int(x), int(y), int(w), int(h)]
            
            bounding_boxes_output_list.append({
                "bbox_coco": bbox_coco_format,
                "foodsam_category_id": foodsam_category_id_val,
                "foodsam_category_name": category_name,
                "score": score_val
            })

        output_structure = {
            "dish_id": os.path.basename(output_bbox_json_path).replace(".json", ""),
            "instances": bounding_boxes_output_list
        }
        save_json(output_structure, output_bbox_json_path)
        logging.info(f"Generated FoodSAM-103 based bounding_box.json: {output_bbox_json_path} with {len(bounding_boxes_output_list)} boxes.")
        return True
    except FileNotFoundError:
        logging.error(f"Masks file not found for bbox generation: {masks_npy_path}")
        return False
    except Exception as e:
        logging.error(f"Error generating FoodSAM-103 bounding_box.json from {masks_npy_path} and {sam_mask_label_path}: {e}", exc_info=True)
        return False
