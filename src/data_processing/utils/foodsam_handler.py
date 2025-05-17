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

def generate_direct_foodsam_outputs(dish_id, rgb_image_path, temp_dish_output_dir):
    """
    Generates core FoodSAM outputs (raw SAM masks, semantic prediction, labels, enhanced mask)
    directly using FoodSAM tools to avoid unnecessary intermediate files.

    Args:
        dish_id (str): The ID of the dish.
        rgb_image_path (str): Path to the input RGB image.
        temp_dish_output_dir (str): Directory to save these direct outputs.

    Returns:
        dict: Paths to key generated files if successful, else None.
              Keys: "enhanced_mask_path", "masks_npy_path", "sam_mask_label_path"
    """
    create_dir_if_not_exists(temp_dish_output_dir)
    logging.info(f"Starting direct FoodSAM processing for {dish_id}...")

    # --- Output file paths ---
    masks_npy_path = os.path.join(temp_dish_output_dir, "masks.npy")
    # Temp for sam_metadata from SAM, not the final sam_mask_label.txt
    # sam_initial_metadata_path = os.path.join(temp_dish_output_dir, "_sam_initial_metadata.csv") 
    pred_mask_png_path = os.path.join(temp_dish_output_dir, "pred_mask.png")
    # The calculate_single_image_masks_label saves to a subdir by default, we want it directly.
    sam_mask_label_dir = os.path.join(temp_dish_output_dir, "sam_mask_label_data") # a sub-folder for calculate_single_image_masks_label
    create_dir_if_not_exists(sam_mask_label_dir)
    sam_mask_label_filename = "sam_mask_label.txt"
    sam_mask_label_path_final_location = os.path.join(temp_dish_output_dir, sam_mask_label_filename) # What we want ultimately
    
    enhanced_mask_path = os.path.join(temp_dish_output_dir, "enhanced_mask.png")
    # Path to color list, relative to FOODSAM_DIR
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
            # Create empty files to allow pipeline to continue if this is acceptable
            np.save(masks_npy_path, np.array([]).astype(bool).reshape(0, image_rgb.shape[0], image_rgb.shape[1]))
            # No SAM masks, so sam_mask_label.txt will be empty or indicate no labels
            with open(sam_mask_label_path_final_location, 'w') as f_sml:
                f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
            # pred_mask and enhanced_mask might still be generated if semantic model runs
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
        # logging.info(f"[{dish_id}] Saved semantic prediction to {pred_mask_png_path}") # This log is now covered above

        # --- D. Generate SAM Mask Labels (sam_mask_label.txt) ---
        # This uses masks.npy and pred_mask.png
        logging.info(f"[{dish_id}] Assigning categories to SAM masks...")
        if not os.path.exists(masks_npy_path) or np.load(masks_npy_path).shape[0] == 0:
            logging.warning(f"[{dish_id}] masks.npy is empty or missing. Skipping SAM mask label generation.")
            # Create an empty sam_mask_label.txt if masks.npy was empty
            with open(sam_mask_label_path_final_location, 'w') as f_sml:
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
                                             sam_mask_label_file_name=sam_mask_label_filename, 
                                             sam_mask_label_file_dir=sam_mask_label_dir) # this will save into temp_dish_output_dir/sam_mask_label_data/sam_mask_label.txt
            
            # Move the generated file to the desired flat location
            generated_sml_path = os.path.join(sam_mask_label_dir, sam_mask_label_filename)
            if os.path.exists(generated_sml_path):
                shutil.move(generated_sml_path, sam_mask_label_path_final_location)
                shutil.rmtree(sam_mask_label_dir) # Clean up the temp subdir
                logging.info(f"[{dish_id}] Saved SAM mask labels to {sam_mask_label_path_final_location}")
            else:
                logging.error(f"[{dish_id}] Failed to generate {generated_sml_path}. It might be empty if no SAM masks.")
                 # Create an empty one if it failed and masks.npy wasn't empty initially
                with open(sam_mask_label_path_final_location, 'w') as f_sml:
                    f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n")
        
        # --- E. Generate Enhanced Semantic Mask (enhanced_mask.png) ---
        logging.info(f"[{dish_id}] Generating enhanced semantic mask...")
        # enhance_masks_tool_func expects: data_folder, category_txt, color_list_path, num_class, area_thr, ratio_thr, top_k ...
        # Its internal loop `for img_folder in os.listdir(data_folder):` is problematic.
        # We need to call its core logic for a single image, or adapt it.
        # For now, let's replicate the core logic for a single image.
        
        # Copy input.jpg as enhance_masks_tool_func expects it for visualization_save, though we want to bypass that.
        shutil.copyfile(rgb_image_path, os.path.join(temp_dish_output_dir, "input.jpg"))

        _pred_mask_img = cv2.imread(pred_mask_png_path)[:,:,2] # Usually red channel from FoodSAM format
        _sam_masks_arr = np.load(masks_npy_path)
        _enhanced_mask_img = _pred_mask_img.copy()

        if os.path.exists(sam_mask_label_path_final_location):
            with open(sam_mask_label_path_final_location, 'r') as f_sml_enhance:
                category_info_lines = f_sml_enhance.readlines()[1:] # Skip header
            
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

        # Clean up the copied input.jpg
        if os.path.exists(os.path.join(temp_dish_output_dir, "input.jpg")):
            os.remove(os.path.join(temp_dish_output_dir, "input.jpg"))

        return {
            "enhanced_mask_path": enhanced_mask_path,
            "masks_npy_path": masks_npy_path,
            "sam_mask_label_path": sam_mask_label_path_final_location
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

def retrieve_direct_foodsam_outputs(dish_id, foodsam_temp_dir, target_dir):
    """
    Copies the directly generated FoodSAM outputs from the temporary directory
    to the final processed dish directory.
    """
    source_files_map = {
        "enhanced_mask.png": "semseg.png", # Source filename in temp_dir : Target filename in final_dir
        "masks.npy": "_temp_masks.npy",    # Will be used for sam_instance.npy and then deleted by main
        "sam_mask_label.txt": "_temp_sam_metadata.csv" # Will be used for bounding_box.json then deleted
    }

    retrieved_file_paths = {}

    for src_name, target_name_in_final_dir_or_temp in source_files_map.items():
        src_path = os.path.join(foodsam_temp_dir, src_name)
        target_path = os.path.join(target_dir, target_name_in_final_dir_or_temp)
        
        if not os.path.exists(src_path):
            # masks.npy or sam_mask_label.txt could be empty if no SAM masks were found, which is handled inside their generation.
            # enhanced_mask.png should always exist if semantic segmentation ran.
            if src_name == "enhanced_mask.png": 
                 logging.error(f"Required FoodSAM output file missing: {src_path} for dish {dish_id}")
                 return None
            else:
                 logging.warning(f"Optional FoodSAM output file missing: {src_path} for dish {dish_id}. Proceeding.")
                 # Create empty files if they are critical for downstream and missing
                 if src_name == "masks.npy" and not os.path.exists(target_path):
                     np.save(target_path, np.array([]).astype(bool).reshape(0,1,1)) # save empty valid npy
                 elif src_name == "sam_mask_label.txt" and not os.path.exists(target_path):
                     with open(target_path, 'w') as f_sml:
                         f_sml.write("id,category_id,category_name,category_count_ratio,mask_count_ratio\n") 
        else:
            copy_file(src_path, target_path)
        
        # Store paths for return, using keys expected by main.py
        if target_name_in_final_dir_or_temp == "semseg.png":
            retrieved_file_paths["semseg_final_path"] = target_path
        elif target_name_in_final_dir_or_temp == "_temp_masks.npy":
            retrieved_file_paths["raw_masks_intermediate_path"] = target_path
        elif target_name_in_final_dir_or_temp == "_temp_sam_metadata.csv": # Note: main.py expects this key
            retrieved_file_paths["sam_metadata_intermediate_path"] = target_path
    
    # Ensure all critical keys are present even if files were missing and dummy created
    if "semseg_final_path" not in retrieved_file_paths and os.path.exists(os.path.join(target_dir, "semseg.png")):
        retrieved_file_paths["semseg_final_path"] = os.path.join(target_dir, "semseg.png")
    if "raw_masks_intermediate_path" not in retrieved_file_paths and os.path.exists(os.path.join(target_dir, "_temp_masks.npy")):
        retrieved_file_paths["raw_masks_intermediate_path"] = os.path.join(target_dir, "_temp_masks.npy")
    if "sam_metadata_intermediate_path" not in retrieved_file_paths and os.path.exists(os.path.join(target_dir, "_temp_sam_metadata.csv")):
        retrieved_file_paths["sam_metadata_intermediate_path"] = os.path.join(target_dir, "_temp_sam_metadata.csv")

    if "semseg_final_path" not in retrieved_file_paths: # Critical check
        logging.error(f"Failed to retrieve semseg.png for {dish_id}")
        return None
        
    logging.info(f"Retrieved direct FoodSAM outputs for {dish_id} to {target_dir} (and its _temp_ files)")
    return retrieved_file_paths
