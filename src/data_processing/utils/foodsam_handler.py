"""
Handler for running FoodSAM components directly, processing its outputs (masks, semantic info),
and generating SAM instances and bounding boxes efficiently.
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

# Helper to add FoodSAM to path to import its tools
def _add_foodsam_to_path():
    # config.FOODSAM_DIR points to the root of the cloned FoodSAM repository
    # e.g., PROJECT_ROOT/external_libs/FoodSAM
    # Adding this to sys.path allows importing packages like `segment_anything`,
    # `mmseg`, and the inner `FoodSAM` package directly from the clone.
    foodsam_clone_root = config.FOODSAM_DIR
    if foodsam_clone_root not in sys.path:
        sys.path.insert(0, foodsam_clone_root)

_add_foodsam_to_path()

# Now import FoodSAM tools
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

# Global cache for models to avoid reloading them for each image
_SAM_MODEL_CACHE = None
_SEMANTIC_MODEL_CACHE = None

# Intermediate modality folder names (within INTERMEDIATE_FOODSAM_OUTPUT_DIR/<split_name>/)
INTERMEDIATE_MASKS_NPY = "masks_npy" # For masks.npy
INTERMEDIATE_SEMANTIC_PRED_RAW = "semantic_pred_raw" # For raw semantic_pred.png (FoodSeg103 class IDs)
INTERMEDIATE_SAM_LABELS = "sam_mask_labels" # For sam_mask_label.txt (FoodSeg103 based)
INTERMEDIATE_ENHANCED_MASK = "enhanced_semantic_mask" # For enhanced_mask.png (FoodSeg103 based, colored or class IDs)

def _get_sam_model():
    global _SAM_MODEL_CACHE
    if _SAM_MODEL_CACHE is None:
        logging.info(f"Loading SAM model: {config.MODEL_TYPE_SAM} from {config.SAM_CHECKPOINT}")
        model = sam_model_registry[config.MODEL_TYPE_SAM](checkpoint=config.SAM_CHECKPOINT)
        _SAM_MODEL_CACHE = model.to(device=config.FOODSAM_DEVICE)
        _SAM_MODEL_CACHE.eval()
    return _SAM_MODEL_CACHE

def _get_semantic_model():
    global _SEMANTIC_MODEL_CACHE
    if _SEMANTIC_MODEL_CACHE is None:
        # semantic_predict tool expects config paths relative to its execution or a known root.
        # We ensure FOODSAM_DIR is in path, so it should find them if paths in config are relative to FOODSAM_DIR root.
        cfg_path = os.path.join(config.FOODSAM_DIR, config.SEMANTIC_CONFIG_FILENAME)
        ckpt_path = os.path.join(config.FOODSAM_DIR, config.SEMANTIC_CHECKPOINT_FILENAME)
        logging.info(f"Loading Semantic Segmentation model from config: {cfg_path} and checkpoint: {ckpt_path}")
        # device arg in init_segmentor handles moving model to device
        model = init_segmentor(cfg_path, ckpt_path, device=config.FOODSAM_DEVICE) 
        if config.FOODSAM_DEVICE != 'cpu': # MMDataParallel is for GPU
             model = MMDataParallel(model, device_ids=[0]) # Assuming single GPU for simplicity
        model.eval()
        _SEMANTIC_MODEL_CACHE = model
    return _SEMANTIC_MODEL_CACHE

def generate_direct_foodsam_outputs(dish_id, split_name, rgb_image_path):
    """
    Generates core FoodSAM outputs (raw SAM masks, semantic prediction, labels, enhanced mask)
    and saves them to a structured intermediate directory.

    Args:
        dish_id (str): The ID of the dish.
        split_name (str): The split name ('train' or 'test').
        rgb_image_path (str): Path to the input RGB image.

    Returns:
        dict: Paths to key generated intermediate files if successful, else None.
              Keys: "enhanced_mask_path", "masks_npy_path", "sam_mask_label_path", "raw_semantic_pred_path"
    """
    logging.info(f"Starting FoodSAM processing for {dish_id} (split: {split_name}) -> intermediate structured output.")

    # --- Base output directories for this dish's intermediate files ---
    base_intermediate_split_dir = os.path.join(config.INTERMEDIATE_FOODSAM_OUTPUT_DIR, split_name)

    # Create dish-specific directories for each intermediate modality
    masks_npy_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_MASKS_NPY, dish_id)
    pred_mask_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_SEMANTIC_PRED_RAW, dish_id)
    sam_labels_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_SAM_LABELS, dish_id)
    enhanced_mask_dish_dir = os.path.join(base_intermediate_split_dir, INTERMEDIATE_ENHANCED_MASK, dish_id)

    for dir_path in [masks_npy_dish_dir, pred_mask_dish_dir, sam_labels_dish_dir, enhanced_mask_dish_dir]:
        create_dir_if_not_exists(dir_path)

    # --- Output file paths (now structured) ---
    masks_npy_path = os.path.join(masks_npy_dish_dir, f"{dish_id}.npy")
    pred_mask_png_path = os.path.join(pred_mask_dish_dir, f"{dish_id}.png") # Raw semantic prediction
    sam_mask_label_path = os.path.join(sam_labels_dish_dir, f"{dish_id}.txt")
    enhanced_mask_path = os.path.join(enhanced_mask_dish_dir, f"{dish_id}.png")

    color_list_full_path = os.path.join(config.FOODSAM_DIR, config.FOODSAM_COLOR_LIST_PATH_FILENAME)
    category_txt_full_path = os.path.join(config.FOODSAM_DIR, config.FOODSAM_CATEGORY_TXT_FILENAME)

    try:
        # --- A. Initialize Models ---
        sam_model = _get_sam_model()
        semantic_model = _get_semantic_model()

        # --- B. Generate Raw SAM Masks (masks.npy) ---
        logging.info(f"[{dish_id}] Generating raw SAM masks...")
        image = cv2.imread(rgb_image_path)
        if image is None:
            logging.error(f"Could not load image {rgb_image_path}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TODO: AMG settings from panoptic.py (get_amg_kwargs) should be exposed via config if needed
        mask_generator = SamAutomaticMaskGenerator(sam_model, points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100)
        sam_masks_data = mask_generator.generate(image_rgb)
        
        if not sam_masks_data:
            logging.warning(f"[{dish_id}] SAM did not generate any masks for {rgb_image_path}")
            np.save(masks_npy_path, np.array([]).astype(bool).reshape(0, image_rgb.shape[0], image_rgb.shape[1]))
            with open(sam_mask_label_path, 'w') as f_sml:
                f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        else:
            # Sort by predicted_iou (or area as in panoptic's write_masks_to_folder)
            # top_k for SAM masks is applied in enhance_masks tool. Here we save all from generator.
            stacked_masks = np.array([m['segmentation'] for m in sam_masks_data], dtype=bool)
            np.save(masks_npy_path, stacked_masks)
            logging.info(f"[{dish_id}] Saved {len(stacked_masks)} raw SAM masks to {masks_npy_path}")

        # --- C. Generate Semantic Prediction (pred_mask.png) ---
        logging.info(f"[{dish_id}] Generating semantic prediction...")
        with torch.no_grad(): # Ensure no gradients for inference
            model_for_inference = semantic_model
            if isinstance(semantic_model, MMDataParallel):
                model_for_inference = semantic_model.module
            semantic_result = inference_segmentor(model_for_inference, rgb_image_path)

        if not semantic_result or len(semantic_result) == 0:
            logging.error(f"[{dish_id}] Semantic inference failed or returned empty result for {rgb_image_path}.")
            return None

        try:
            # Directly save the raw semantic mask (class IDs)
            # semantic_result[0] is the segmentation map
            raw_semantic_mask = semantic_result[0].astype(np.uint8)
            # pred_mask_png_path is defined earlier in the function as:
            # os.path.join(temp_dish_output_dir, "pred_mask.png")
            mmcv.imwrite(raw_semantic_mask, pred_mask_png_path)
            logging.info(f"[{dish_id}] Saved raw semantic prediction to {pred_mask_png_path}")
        except Exception as e_save:
            logging.error(f"[{dish_id}] Failed to save semantic prediction {pred_mask_png_path}: {e_save}", exc_info=True)
            return None

        # Ensure the file was actually created
        if not os.path.exists(pred_mask_png_path):
            logging.error(f"[{dish_id}] {pred_mask_png_path} was not created after mmcv.imwrite call.")
            return None

        # --- D. Generate SAM Mask Labels (sam_mask_label.txt) ---
        # This uses masks.npy and pred_mask.png
        logging.info(f"[{dish_id}] Assigning categories to SAM masks...")
        if not os.path.exists(masks_npy_path) or np.load(masks_npy_path).shape[0] == 0:
            logging.warning(f"[{dish_id}] masks.npy is empty or missing. Skipping SAM mask label generation.")
            with open(sam_mask_label_path, 'w') as f_sml:
                f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        else:
            # calculate_single_image_masks_label expects masks_path_name, pred_mask_file, category_list, sam_mask_label_file_name, sam_mask_label_file_dir
            # It operates on a single image's data. We pass direct file paths.
            # It reads category_txt to get category_list.
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
        logging.info(f"[{dish_id}] Generating enhanced semantic mask...")
        # enhance_masks_tool_func expects: data_folder, category_txt, color_list_path, num_class, area_thr, ratio_thr, top_k ...
        # Its internal loop `for img_folder in os.listdir(data_folder):` is problematic.
        # We need to call its core logic for a single image, or adapt it.
        # For now, let's replicate the core logic for a single image.
        
        _pred_mask_img = cv2.imread(pred_mask_png_path)[:,:,2]
        _sam_masks_arr = np.load(masks_npy_path)
        _enhanced_mask_img = _pred_mask_img.copy()

        if os.path.exists(sam_mask_label_path):
            with open(sam_mask_label_path, 'r') as f_sml_enhance:
                category_info_lines = f_sml_enhance.readlines()[1:]
            
            # Sort and filter by top_k (as in enhance_masks_tool_func)
            # Note: score for sorting is 'mask_count_ratio' (area of mask / total image area)
            valid_category_info = []
            for line in category_info_lines:
                parts = line.strip().split(',')
                if len(parts) == 5:
                    try: valid_category_info.append({'line': line, 'mask_area_ratio': float(parts[4])}) 
                    except ValueError: continue      
            
            category_info_lines_sorted = sorted(valid_category_info, key=lambda x: x['mask_area_ratio'], reverse=True)
            category_info_lines_top_k = [item['line'] for item in category_info_lines_sorted[:config.ENHANCE_MASKS_TOP_K]]

            if _sam_masks_arr.shape[0] > 0: # Check if there are any SAM masks
                for info_line in category_info_lines_top_k:
                    parts = info_line.strip().split(',')                    
                    idx, label_id, count_ratio_str, area_ratio_str = parts[0], parts[1], parts[3], parts[4]
                    try:
                        idx = int(idx)
                        label_id = int(label_id)
                        count_ratio = float(count_ratio_str)
                        area_ratio = float(area_ratio_str) # This is mask_count_ratio from sam_mask_label.txt
                    except ValueError as ve:
                        logging.warning(f"[{dish_id}] Skipping malformed line in sam_mask_label.txt: {info_line} - {ve}")
                        continue

                    if area_ratio * (image_rgb.shape[0] * image_rgb.shape[1]) < config.ENHANCE_MASKS_AREA_THR: # area_ratio is already a ratio of total image area
                        continue
                    if count_ratio < config.ENHANCE_MASKS_RATIO_THR:
                        continue
                    if idx >= _sam_masks_arr.shape[0]:
                        logging.warning(f"[{dish_id}] Mask index {idx} out of bounds for _sam_masks_arr.")
                        continue

                    sam_mask_slice = _sam_masks_arr[idx].astype(bool)
                    _enhanced_mask_img[sam_mask_slice] = label_id
        
        cv2.imwrite(enhanced_mask_path, _enhanced_mask_img)
        logging.info(f"[{dish_id}] Saved enhanced semantic mask to {enhanced_mask_path}")

        return {
            "enhanced_mask_path": enhanced_mask_path,
            "masks_npy_path": masks_npy_path,
            "sam_mask_label_path": sam_mask_label_path,
            "raw_semantic_pred_path": pred_mask_png_path
        }

    except Exception as e:
        logging.error(f"[{dish_id}] Failed during direct FoodSAM processing: {e}", exc_info=True)
        return None

def convert_sam_masks_to_polygons(raw_masks_npy_path, output_polygon_npy_path):
    """ (Identical to previous version, kept for completeness) """
    try:
        binary_masks = np.load(raw_masks_npy_path)
        if binary_masks.ndim == 0 or binary_masks.size == 0 : # Handle empty masks_npy from no SAM output
             logging.warning(f"Raw SAM masks file is empty or invalid: {raw_masks_npy_path}. Saving empty polygons.")
             np.save(output_polygon_npy_path, np.array([], dtype=object))
             return True
        if binary_masks.ndim != 3 or binary_masks.dtype != bool:
            logging.error(f"Invalid format for raw SAM masks: {raw_masks_npy_path}. Expected 3D bool array. Got {binary_masks.shape}, {binary_masks.dtype}")
            return False

        polygon_instances = []
        for i in range(binary_masks.shape[0]):
            mask_slice = binary_masks[i].astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours or hierarchy is None: # hierarchy can be None if no contours
                continue

            for contour_idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= config.MIN_CONTOUR_AREA_SAM_INSTANCE:
                    is_outer = hierarchy[0][contour_idx][3] == -1 
                    if is_outer:
                        polygon_points = contour.squeeze(axis=1).tolist() 
                        polygon_instances.append({"points": polygon_points, "original_mask_index": i})
        
        np.save(output_polygon_npy_path, np.array(polygon_instances, dtype=object))
        logging.info(f"Converted SAM masks to polygons: {output_polygon_npy_path} with {len(polygon_instances)} instances.")
        return True
    except Exception as e:
        logging.error(f"Error converting SAM masks to polygons ({raw_masks_npy_path}): {e}", exc_info=True)
        return False

def generate_bounding_boxes_json(masks_npy_path, sam_mask_label_path, output_bbox_json_path):
    """ (Adjusted to use masks_npy_path directly for bounding box calculation from boolean masks) """
    try:
        # masks_data is (N, H, W) boolean array
        masks_data = np.load(masks_npy_path)
        if masks_data.ndim == 0 or masks_data.size == 0:
            logging.warning(f"masks.npy is empty for bounding box generation: {masks_npy_path}. Saving empty JSON.")
            save_json([], output_bbox_json_path)
            return True
        if masks_data.ndim != 3 or masks_data.dtype != bool:
            logging.error(f"Invalid masks_npy file for bbox: {masks_npy_path}. Got {masks_data.shape}, {masks_data.dtype}")
            return False

        metadata_entries = []
        if not os.path.exists(sam_mask_label_path):
            logging.warning(f"sam_mask_label.txt not found: {sam_mask_label_path}. Bounding boxes will lack class names/scores.")
        else:
            with open(sam_mask_label_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata_entries.append(row)

        bounding_boxes_4m = []
        # Create a map from mask_idx to its category info for quick lookup
        mask_idx_to_meta = {int(entry['id']): entry for entry in metadata_entries if 'id' in entry}

        for i in range(masks_data.shape[0]): # Iterate through each mask in masks.npy
            mask_slice = masks_data[i]
            if not np.any(mask_slice): # Skip empty masks
                continue

            category_name = "unknown"
            score = config.DEFAULT_BBOX_SCORE

            meta_entry = mask_idx_to_meta.get(i)
            if meta_entry:
                category_name = meta_entry.get('category_name', category_name)
                try:
                    score = float(meta_entry.get('mask_count_ratio', score)) # Using mask_count_ratio as score
                except ValueError: pass
            
            if category_name.lower() == "background":
                continue
            
            contours, _ = cv2.findContours(mask_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            
            bounding_boxes_4m.append({
                "boxes": [xmin, ymin, xmax, ymax],
                "class_name": category_name,
                "score": score
            })

        save_json(bounding_boxes_4m, output_bbox_json_path)
        logging.info(f"Generated bounding_box.json: {output_bbox_json_path} with {len(bounding_boxes_4m)} boxes.")
        return True
    except Exception as e:
        logging.error(f"Error generating bounding_box.json from {masks_npy_path} and {sam_mask_label_path}: {e}", exc_info=True)
        return False
