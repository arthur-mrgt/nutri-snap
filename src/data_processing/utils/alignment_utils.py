"""
Utilities for aligning FoodSAM detections with Nutrition5k categories and generating
N5K-aligned modalities.
"""
import os
import numpy as np
import cv2 # OpenCV for contour finding, bounding box calculation, image reading/writing
import logging
import math
from . import config # For constants if needed (e.g. scoring parameters)
from .common_utils import load_json, save_json

# --- Constants for Scoring (as per user prompt) ---
SCORE_K_MSRE = 1.0
SCORE_W_NUTR = 0.8
SCORE_W_INGR = 0.2
# Small value to prevent division by zero in relative error calculation,
# or to assign a large penalty if original is 0 and merged is not.
# Using a fixed large RE if X_orig=0 and X_merged!=0 might be more stable than division by epsilon.
# For now, let's use RE=1 in that specific case, as user suggested.
RE_X_ORIG_ZERO_MERGED_NON_ZERO_PENALTY = 1.0 


def extract_foodsam_instances_with_masks(raw_sam_masks_npy_path, raw_foodsam103_semantic_pred_path, foodsam103_id_to_name_map):
    """
    Extracts FoodSAM instances, their raw masks, and their majority FoodSAM-103 category.

    Args:
        raw_sam_masks_npy_path (str): Path to masks.npy (raw SAM masks, (N, H, W) boolean array).
        raw_foodsam103_semantic_pred_path (str): Path to semantic_pred_food103.png (H, W, class IDs).
        foodsam103_id_to_name_map (dict): Mapping from FoodSAM-103 category ID to name.

    Returns:
        list: List of dictionaries, e.g.,
              [{'original_mask_index': int, 
                'foodsam_category_id': int, 
                'foodsam_category_name': str, 
                'instance_mask': np.array((H,W), dtype=bool)}, ...]
              Returns empty list on error or if no masks.
    """
    detected_instances = []
    try:
        sam_masks = np.load(raw_sam_masks_npy_path)
        if sam_masks.ndim != 3 or sam_masks.shape[0] == 0:
            logging.warning(f"SAM masks at {raw_sam_masks_npy_path} are not valid or empty. Shape: {sam_masks.shape}")
            return []

        # Load semantic prediction (grayscale, where pixel value is class ID)
        semantic_pred_img = cv2.imread(raw_foodsam103_semantic_pred_path, cv2.IMREAD_GRAYSCALE)
        if semantic_pred_img is None:
            logging.error(f"Could not load FoodSAM-103 semantic prediction image: {raw_foodsam103_semantic_pred_path}")
            return []

        if sam_masks.shape[1:] != semantic_pred_img.shape:
            logging.error(f"Shape mismatch: SAM masks {sam_masks.shape[1:]} vs Semantic pred {semantic_pred_img.shape}")
            # Attempt resize if config allows, or fail
            # For now, fail:
            return []

        for i in range(sam_masks.shape[0]):
            instance_mask = sam_masks[i] # (H, W) boolean array
            if not np.any(instance_mask): # Skip empty masks
                continue

            # Get FoodSAM-103 class IDs from semantic image where SAM mask is true
            pixels_in_mask = semantic_pred_img[instance_mask]
            
            if pixels_in_mask.size == 0:
                majority_foodsam_cat_id = -1 # or some background/unknown ID
            else:
                unique_ids, counts = np.unique(pixels_in_mask, return_counts=True)
                majority_foodsam_cat_id = unique_ids[np.argmax(counts)]
            
            foodsam_category_name = foodsam103_id_to_name_map.get(int(majority_foodsam_cat_id), "unknown_foodsam_id")

            detected_instances.append({
                'original_mask_index': i,
                'foodsam_category_id': int(majority_foodsam_cat_id),
                'foodsam_category_name': foodsam_category_name,
                'instance_mask': instance_mask 
            })
        logging.info(f"Extracted {len(detected_instances)} FoodSAM instances from {os.path.basename(raw_sam_masks_npy_path)}")

    except FileNotFoundError:
        logging.error(f"File not found during FoodSAM instance extraction. SAM masks: {raw_sam_masks_npy_path} or Sem pred: {raw_foodsam103_semantic_pred_path}")
        return []
    except Exception as e:
        logging.error(f"Error extracting FoodSAM instances: {e}", exc_info=True)
        return []
    return detected_instances


def generate_aligned_n5k_metadata_with_scores(dish_id, original_n5k_metadata, sam_mask_labels, 
                                           foodsam_name_to_n5k_names_map, 
                                           n5k_name_to_id_map, n5k_id_to_name_map):
    """
    Generates N5K-aligned metadata.json content, including nutritional info and confidence scores.
    Follows the exact mapping logic specified:
    1. Get unique FoodSAM categories from sam_mask_labels
    2. For each FoodSAM category, find matching N5K ingredient using the mapping file
    3. Calculate confidence scores
    4. Generate new semseg and bounding boxes using N5K categories

    Args:
        dish_id (str): The ID of the dish.
        original_n5k_metadata (dict): Parsed metadata from Nutrition5k for the dish.
        sam_mask_labels: Either a path to sam_mask_labels.csv or a list of dictionaries containing the data.
        foodsam_name_to_n5k_names_map (dict): Mapping FoodSAM-103 category ID to list of N5K categories.
        n5k_name_to_id_map (dict): Mapping N5K category name to N5K ID.
        n5k_id_to_name_map (dict): Mapping N5K ID to N5K name.

    Returns:
        tuple: (aligned_metadata_content_dict, instance_to_final_n5k_assignment_dict)
    """
    if not original_n5k_metadata or not original_n5k_metadata.get('ingredients'):
        logging.warning(f"[{dish_id}] Original N5K metadata is empty or has no ingredients. Cannot align.")
        return None, None

    # Helper to quickly look up original N5K ingredients by their N5K ID
    original_n5k_ingredients_by_id = {str(ing['ingredient_id']): ing for ing in original_n5k_metadata['ingredients']}
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_by_id.keys())}")

    # --- Step 1: Process detected FoodSAM instances (sam_mask_labels) ---
    try:
        import pandas as pd
        # sam_mask_labels is detected_foodsam_instances (list of dicts from extract_foodsam_instances_with_masks)
        # Expected keys in dicts: 'original_mask_index', 'foodsam_category_id', 'foodsam_category_name', 'instance_mask'

        if isinstance(sam_mask_labels, str):
            logging.info(f"[{dish_id}] Processing sam_mask_labels as a CSV file path: {sam_mask_labels}")
            sam_labels_df = pd.read_csv(sam_mask_labels)
            # Ensure 'category_id' and 'original_mask_index' (if CSV path is used and requires it) exist.
            # This path is not the source of the current error.
            # For robustness, check for 'category_id':
            if 'category_id' not in sam_labels_df.columns:
                 logging.error(f"[{dish_id}] CSV {sam_mask_labels} does not contain 'category_id' column.")
                 # Return or create an empty DF with expected columns if that's handled gracefully.
                 # For now, let's assume CSVs are correctly formatted or this path isn't used.
                 # Consider `return None, None` for critical missing columns.

        elif isinstance(sam_mask_labels, list): # This is the path taken for the reported error
            if not sam_mask_labels: # Empty list of detections
                logging.warning(f"[{dish_id}] sam_mask_labels (detected_foodsam_instances) is empty.")
                # Create an empty DataFrame with expected columns for downstream consistency.
                sam_labels_df = pd.DataFrame(columns=['category_id', 'original_mask_index'])
            else:
                # Check structure of the first element for required keys
                first_item = sam_mask_labels[0]
                if not isinstance(first_item, dict) or \
                   'foodsam_category_id' not in first_item or \
                   'original_mask_index' not in first_item:
                    logging.error(f"[{dish_id}] sam_mask_labels list elements have unexpected structure. "
                                  f"First item: {first_item}. Required keys: 'foodsam_category_id', 'original_mask_index'.")
                    return None, None # Critical: input data structure is wrong
                
                sam_labels_df = pd.DataFrame(sam_mask_labels)
                # Rename 'foodsam_category_id' to 'category_id' for consistency with subsequent code.
                # 'original_mask_index' column is already present with the correct name from extract_foodsam_instances_with_masks.
                if 'foodsam_category_id' in sam_labels_df.columns:
                    sam_labels_df.rename(columns={'foodsam_category_id': 'category_id'}, inplace=True)
                else:
                    # This should not happen if extract_foodsam_instances_with_masks provides the 'foodsam_category_id' key.
                    logging.error(f"[{dish_id}] 'foodsam_category_id' not found in DataFrame columns from sam_mask_labels list. Columns: {sam_labels_df.columns}")
                    return None, None # Critical: input data is missing an expected column
        else: # Not a string path, not a list
            logging.error(f"[{dish_id}] Invalid sam_mask_labels format: type {type(sam_mask_labels)}. Expected str or list.")
            return None, None

        # Now, sam_labels_df should be correctly populated with 'category_id' and 'original_mask_index' columns,
        # or be an empty DataFrame (with these columns) if sam_mask_labels was an empty list.
        
        # Get unique category_ids, excluding background (0)
        if 'category_id' in sam_labels_df.columns and not sam_labels_df.empty: # Check if column exists and df not empty
            unique_foodsam_categories = sam_labels_df[sam_labels_df['category_id'] != 0]['category_id'].unique()
        else: # Column missing or DataFrame is empty (e.g., from empty sam_mask_labels)
            unique_foodsam_categories = np.array([]) 
        
        logging.debug(f"[{dish_id}] Unique FoodSAM category IDs for processing: {unique_foodsam_categories}")

    except Exception as e:
        # Catch any other unexpected error during DataFrame setup.
        logging.error(f"[{dish_id}] Error during initial processing of sam_mask_labels into DataFrame: {e}", exc_info=True)
        return None, None

    # --- Step 2: Map FoodSAM categories to N5K ingredients ---
    merged_ingredients_list_for_json = []
    instance_to_final_n5k_assignment = {}  # Will store mapping for semseg/bbox generation
    
    # Initialize accumulators for dish totals
    total_summed_weight_for_dish = 0.0
    total_summed_calories_for_dish = 0.0
    total_summed_fat_for_dish = 0.0
    total_summed_carbs_for_dish = 0.0
    total_summed_protein_for_dish = 0.0

    # Process each unique FoodSAM category
    for foodsam_cat_id in unique_foodsam_categories:
        foodsam_cat_id_str = str(foodsam_cat_id)
        
        # Get mapping entry for this FoodSAM category
        mapping_entry = foodsam_name_to_n5k_names_map.get(foodsam_cat_id_str)
        if not mapping_entry:
            logging.warning(f"[{dish_id}] No mapping found for FoodSAM category ID {foodsam_cat_id_str}")
            continue

        # Get the list of potential N5K categories for this FoodSAM category
        mapped_n5k_categories = mapping_entry.get('mapped_n5k_categories', [])
        if not mapped_n5k_categories:
            logging.warning(f"[{dish_id}] No N5K categories mapped for FoodSAM category ID {foodsam_cat_id_str}")
            continue

        # Try to find a matching N5K ingredient in the dish
        matched_n5k_ingredient = None
        for n5k_candidate in mapped_n5k_categories:
            n5k_id = str(n5k_candidate.get('n5k_id'))
            # Convert to ground truth format (e.g., "448" -> "ingr_0000000448")
            n5k_id_formatted = f"ingr_{int(n5k_id):08d}"
            
            if n5k_id_formatted in original_n5k_ingredients_by_id:
                matched_n5k_ingredient = original_n5k_ingredients_by_id[n5k_id_formatted]
                logging.debug(f"[{dish_id}] Matched FoodSAM category {foodsam_cat_id_str} to N5K ingredient {n5k_id_formatted}")
                break

        if matched_n5k_ingredient:
            # Add to merged ingredients list
            merged_ingredient = {
                'id': matched_n5k_ingredient['ingredient_id'],
                'name': matched_n5k_ingredient['ingredient_name'],
                'weight_g': matched_n5k_ingredient['weight_g'],
                'calories_kcal': matched_n5k_ingredient['calories_kcal'],
                'fat_g': matched_n5k_ingredient['fat_g'],
                'carbs_g': matched_n5k_ingredient['carbs_g'],
                'protein_g': matched_n5k_ingredient['protein_g']
            }
            merged_ingredients_list_for_json.append(merged_ingredient)

            # Update dish totals
            total_summed_weight_for_dish += matched_n5k_ingredient['weight_g']
            total_summed_calories_for_dish += matched_n5k_ingredient['calories_kcal']
            total_summed_fat_for_dish += matched_n5k_ingredient['fat_g']
            total_summed_carbs_for_dish += matched_n5k_ingredient['carbs_g']
            total_summed_protein_for_dish += matched_n5k_ingredient['protein_g']

            # Store mapping for semseg/bbox generation
            # Get all instances (identified by 'original_mask_index') for this FoodSAM category
            if 'original_mask_index' not in sam_labels_df.columns:
                logging.error(f"[{dish_id}] Critical: 'original_mask_index' column is missing from sam_labels_df. Cannot map instances for semseg/bbox.")
                # This implies a problem in the DataFrame setup logic earlier.
                # Continue without this part or return error, depending on desired strictness.
            else:
                relevant_instances_df = sam_labels_df[sam_labels_df['category_id'] == foodsam_cat_id]
                for original_idx in relevant_instances_df['original_mask_index']:
                    # Ensure original_idx is an integer, suitable as a dictionary key.
                    instance_to_final_n5k_assignment[int(original_idx)] = {
                        'id': matched_n5k_ingredient['ingredient_id'], # Should be N5K integer ID
                        'name': matched_n5k_ingredient['ingredient_name']
                    }

    # --- Step 3: Calculate confidence scores ---
    # Calculate merged totals
    merged_totals = {
        'weight_g': round(total_summed_weight_for_dish, 2),
        'calories_kcal': round(total_summed_calories_for_dish, 1),
        'fat_g': round(total_summed_fat_for_dish, 2),
        'carbs_g': round(total_summed_carbs_for_dish, 2),
        'protein_g': round(total_summed_protein_for_dish, 2)
    }

    # Original totals
    orig_M = original_n5k_metadata.get('dish_total_mass_g', 0.0)
    orig_K = original_n5k_metadata.get('dish_total_calories_kcal', 0.0)
    orig_L = original_n5k_metadata.get('dish_total_fat_g', 0.0)
    orig_G = original_n5k_metadata.get('dish_total_carbs_g', 0.0)
    orig_P = original_n5k_metadata.get('dish_total_protein_g', 0.0)

    # Calculate nutritional score
    nutrients_comparison = [
        (merged_totals['weight_g'], orig_M),
        (merged_totals['calories_kcal'], orig_K),
        (merged_totals['fat_g'], orig_L),
        (merged_totals['carbs_g'], orig_G),
        (merged_totals['protein_g'], orig_P)
    ]
    
    squared_relative_errors = []
    for merged_val, orig_val in nutrients_comparison:
        if orig_val != 0:
            re = (merged_val - orig_val) / orig_val
        else:
            if merged_val == 0:
                re = 0.0
            else:
                re = RE_X_ORIG_ZERO_MERGED_NON_ZERO_PENALTY
        squared_relative_errors.append(re**2)
    
    msre_nutritional = sum(squared_relative_errors) / len(squared_relative_errors) if squared_relative_errors else 0.0
    score_nutr = math.exp(-SCORE_K_MSRE * msre_nutritional)

    # Calculate ingredient score (Jaccard Index)
    set_ingr_merged = {ing['id'] for ing in merged_ingredients_list_for_json}
    set_ingr_orig = {ing['ingredient_id'] for ing in original_n5k_metadata['ingredients']}
    
    intersection_len = len(set_ingr_merged.intersection(set_ingr_orig))
    union_len = len(set_ingr_merged.union(set_ingr_orig))
    jaccard_ingredient = intersection_len / union_len if union_len > 0 else 0.0
    score_ingr = jaccard_ingredient

    # Calculate final confidence score
    final_confidence = (SCORE_W_NUTR * score_nutr) + (SCORE_W_INGR * score_ingr)

    # --- Step 4: Assemble final metadata content ---
    aligned_metadata_content = {
        "dish_id": dish_id,
        "ingredients": merged_ingredients_list_for_json,
        "total_nutritional_values": merged_totals,
        "confidence_scores": {
            "final_confidence": round(final_confidence, 4),
            "nutritional_score": round(score_nutr, 4),
            "ingredient_score": round(score_ingr, 4),
            "msre_nutritional": round(msre_nutritional, 4),
            "jaccard_ingredient": round(jaccard_ingredient, 4)
        }
    }

    logging.info(f"[{dish_id}] Generated aligned N5K metadata with scores. Ingredients count: {len(merged_ingredients_list_for_json)}. Final confidence: {final_confidence:.4f}")
    return aligned_metadata_content, instance_to_final_n5k_assignment


def generate_n5k_semseg_from_sam(raw_sam_masks_npy_path, instance_to_final_n5k_assignment, image_shape, n5k_id_to_name_map=None):
    """
    Generates an N5K category-based semantic segmentation image from raw SAM masks.
    Phase 3 for semseg.

    Args:
        raw_sam_masks_npy_path (str): Path to masks.npy (raw SAM masks).
        instance_to_final_n5k_assignment (dict): Maps original_mask_index to {'id': n5k_id, 'name': n5k_name} or None.
        image_shape (tuple): (height, width) of the output image.
        n5k_id_to_name_map (dict, optional): For logging/debugging.

    Returns:
        np.array: Semantic segmentation image (H, W) with N5K category IDs, or None on error.
                  IDs are N5K integer IDs.
    """
    try:
        sam_masks = np.load(raw_sam_masks_npy_path)
        if sam_masks.ndim != 3 or sam_masks.shape[0] == 0:
            logging.warning(f"SAM masks at {raw_sam_masks_npy_path} are not valid or empty for N5K semseg. Shape: {sam_masks.shape}")
            return np.zeros(image_shape, dtype=np.uint8) # Return empty mask

        # Output semantic mask (N5K category IDs)
        # Ensure dtype can hold all N5K IDs (e.g., uint16 if IDs > 255)
        # Assuming N5K IDs fit in uint8 for now (0-555 is fine for uint16, might be an issue for uint8 if background=0 is already max)
        # Nutrition5k IDs go up to 555. Default background is 0.
        # Check max ID or use a dtype that can accommodate.
        # For simplicity, let's assume IDs are mapped to something that fits uint8 or the visualization handles it.
        # Or, more robustly, find max n5k_id from instance_to_final_n5k_assignment or category map.
        # For now, let's assume N5K IDs passed are directly usable.
        max_id = 0
        for assignment in instance_to_final_n5k_assignment.values():
            if assignment and isinstance(assignment.get('id'), int) and assignment.get('id') > max_id:
                max_id = assignment.get('id')
        
        dtype_semseg = np.uint16 if max_id > 255 else np.uint8
        n5k_semantic_img = np.zeros(image_shape, dtype=dtype_semseg)


        # Iterate from last mask to first so earlier masks (typically larger or background)
        # can be overwritten by later, more specific masks if they overlap.
        # Or, consider mask areas / confidence if available.
        # For now, simple overwrite based on SAM mask order.
        num_masks = sam_masks.shape[0]
        for i in range(num_masks): # Iterate through each SAM mask by its original index
            instance_mask_data = sam_masks[i]
            
            n5k_assignment = instance_to_final_n5k_assignment.get(i)
            if n5k_assignment and n5k_assignment.get('id') is not None:
                n5k_id = n5k_assignment['id']
                # Ensure n5k_id is an integer if it comes from string parsing somewhere
                # try:
                #     n5k_id_int = int(n5k_id)
                #     n5k_semantic_img[instance_mask_data] = n5k_id_int
                #     if n5k_id_to_name_map and i % 50 == 0: # Log occasionally
                #          logging.debug(f"Mask {i} mapped to N5K ID {n5k_id_int} ({n5k_id_to_name_map.get(n5k_id_int, 'Unknown N5K ID')})")
                # except ValueError:
                #     logging.warning(f"Could not convert N5K ID '{n5k_id}' to int for mask index {i}.")
                if isinstance(n5k_id, int):
                    n5k_semantic_img[instance_mask_data] = n5k_id
                    if n5k_id_to_name_map and i % 20 == 0 and n5k_id != 0 : # Log non-background assignments occasionally
                         logging.debug(f"Mask {i} (partially) painted with N5K ID {n5k_id} ({n5k_id_to_name_map.get(n5k_id, 'Unknown N5K ID')})")
                else:
                    logging.debug(f"Mask {i} mapped to N5K ID '{n5k_id}' (type: {type(n5k_id)}), which is not an integer. Skipping for semseg coloring (remains background).")

            # Else: SAM mask not mapped to an N5K ingredient, pixels remain background (0)
        
        # Check if the image is all zeros (black)
        if np.all(n5k_semantic_img == 0):
            logging.warning(f"Generated N5K semantic segmentation for {os.path.basename(raw_sam_masks_npy_path)} is completely black (all background).")
        else:
            logging.info(f"Generated N5K semantic segmentation image from {os.path.basename(raw_sam_masks_npy_path)}")
        return n5k_semantic_img

    except FileNotFoundError:
        logging.error(f"SAM masks file not found for N5K semseg: {raw_sam_masks_npy_path}")
    except Exception as e:
        logging.error(f"Error generating N5K semantic segmentation: {e}", exc_info=True)
    return None


def generate_n5k_bboxes_from_sam(dish_id, raw_sam_masks_npy_path, instance_to_final_n5k_assignment, n5k_id_to_name_map=None):
    """
    Generates bounding_box.json content with N5K categories.
    Phase 3 for bounding_box.

    Args:
        dish_id (str): The ID of the dish.
        raw_sam_masks_npy_path (str): Path to masks.npy (raw SAM masks).
        instance_to_final_n5k_assignment (dict): Maps original_mask_index to {'id': n5k_id, 'name': n5k_name} or None.
        n5k_id_to_name_map (dict, optional): For richer logging if needed.

    Returns:
        dict: Content for bounding_box.json, or None on error.
              Format: {"dish_id": ..., "bounding_boxes": [...]}
    """
    bounding_boxes_list = []
    try:
        sam_masks = np.load(raw_sam_masks_npy_path)
        if sam_masks.ndim != 3 or sam_masks.shape[0] == 0:
            logging.warning(f"SAM masks at {raw_sam_masks_npy_path} are not valid or empty for N5K bbox. Shape: {sam_masks.shape}")
            return {"dish_id": dish_id, "bounding_boxes": []}


        num_masks = sam_masks.shape[0]
        for i in range(num_masks):
            instance_mask_data = sam_masks[i]
            if not np.any(instance_mask_data): # Skip empty masks
                continue
            
            n5k_assignment = instance_to_final_n5k_assignment.get(i)
            if n5k_assignment and n5k_assignment.get('id') is not None:
                n5k_id_val = n5k_assignment['id']
                n5k_name_val = n5k_assignment['name']

                # Calculate bounding box from the boolean mask
                # Ensure mask is uint8 for findContours
                mask_u8 = instance_mask_data.astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # If multiple contours, take the bounding box of all points (union)
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)

                bounding_boxes_list.append({
                    "label": n5k_name_val,
                    "id": n5k_id_val, # N5K ID (can be string or int based on source)
                    "box2d": [xmin, ymin, xmax, ymax]
                })
        
        logging.info(f"Generated {len(bounding_boxes_list)} N5K bounding boxes from {os.path.basename(raw_sam_masks_npy_path)}")
        return {"dish_id": dish_id, "bounding_boxes": bounding_boxes_list}

    except FileNotFoundError:
        logging.error(f"SAM masks file not found for N5K bbox: {raw_sam_masks_npy_path}")
    except Exception as e:
        logging.error(f"Error generating N5K bounding boxes: {e}", exc_info=True)
    return None 