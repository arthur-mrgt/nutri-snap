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


def generate_aligned_n5k_metadata_with_scores(dish_id, original_n5k_metadata, detected_foodsam_instances, 
                                           foodsam_name_to_n5k_names_map, 
                                           n5k_name_to_id_map, n5k_id_to_name_map):
    """
    Generates N5K-aligned metadata.json content, including nutritional info and confidence scores.
    Phase 2 & 4 of the plan.

    Args:
        dish_id (str): The ID of the dish.
        original_n5k_metadata (dict): Parsed metadata from Nutrition5k for the dish (includes ingredient IDs and names).
        detected_foodsam_instances (list): Output from extract_foodsam_instances_with_masks.
        foodsam_name_to_n5k_names_map (dict): Mapping FoodSAM-103 category name to list of N5K category names.
        n5k_name_to_id_map (dict): Mapping N5K category name to N5K ID.
        n5k_id_to_name_map (dict): Mapping N5K ID to N5K name.


    Returns:
        tuple: (aligned_metadata_content_dict, instance_to_final_n5k_assignment_dict)
               Returns (None, None) on critical error.
               instance_to_final_n5k_assignment_dict maps original_mask_index to {'id': n5k_id, 'name': n5k_name}
    """
    if not original_n5k_metadata or not original_n5k_metadata.get('ingredients'):
        logging.warning(f"[{dish_id}] Original N5K metadata is empty or has no ingredients. Cannot align.")
        return None, None

    # Helper to quickly look up original N5K ingredients by their N5K name or ID
    # Assumes original_n5k_metadata['ingredients'] now has 'ingredient_id' and 'ingredient_name'
    original_n5k_ingredients_by_id = {ing['ingredient_id']: ing for ing in original_n5k_metadata['ingredients']}
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_by_id.keys())}")
    
    # Stores the final N5K ID and name that an original SAM mask index maps to.
    # e.g., {0: {'id': 'n5k_id_apple', 'name': 'apple'}, 1: None, ...}
    instance_to_final_n5k_assignment = {} 

    # Defines the final N5K ingredients based on mappings.
    # Key: final_n5k_id (e.g., N_principal's ID or N_single's ID).
    # Value: {'name': final_n5k_name, 'id': final_n5k_id, 
    #         'source_original_n5k_ingredient_ids': set of original N5K ingredient IDs from the dish}
    final_n5k_ingredient_definitions = {}

    for foodsam_instance in detected_foodsam_instances:
        original_mask_idx = foodsam_instance['original_mask_index']
        foodsam_cat_name = foodsam_instance['foodsam_category_name']
        instance_to_final_n5k_assignment[original_mask_idx] = None # Default to no mapping

        if foodsam_cat_name.lower() == "background" or foodsam_cat_name == "unknown_foodsam_id":
            continue

        candidate_n5k_names_for_foodsam_cat = foodsam_name_to_n5k_names_map.get(foodsam_cat_name, [])
        if not candidate_n5k_names_for_foodsam_cat:
            logging.debug(f"[{dish_id}] FoodSAM category '{foodsam_cat_name}' (mask_idx {original_mask_idx}): No N5K mapping found in foodsam_to_n5k_map.")
            continue
        
        logging.debug(f"[{dish_id}] FoodSAM category '{foodsam_cat_name}' (mask_idx {original_mask_idx}): Maps to N5K candidate names: {candidate_n5k_names_for_foodsam_cat}")
        # Find which of these N5K candidates are actually in the current dish's original N5K metadata
        matching_original_n5k_ingredients_in_dish = []
        for n5k_cand_name in candidate_n5k_names_for_foodsam_cat:
            n5k_cand_id = n5k_name_to_id_map.get(n5k_cand_name)
            if n5k_cand_id and n5k_cand_id in original_n5k_ingredients_by_id:
                matching_original_n5k_ingredients_in_dish.append(original_n5k_ingredients_by_id[n5k_cand_id])
        
        if not matching_original_n5k_ingredients_in_dish:
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (mask_idx {original_mask_idx}): Candidate N5K names {candidate_n5k_names_for_foodsam_cat} are not present in this dish's original N5K metadata.")
            continue

        # Apply user's logic for N_single, N_multi
        final_id_for_instance = None
        final_name_for_instance = None
        source_ids_to_add_for_this_instance = set()

        if len(matching_original_n5k_ingredients_in_dish) == 1: # N_single case
            n_single_orig_ing = matching_original_n5k_ingredients_in_dish[0]
            final_id_for_instance = n_single_orig_ing['ingredient_id']
            final_name_for_instance = n_single_orig_ing['ingredient_name']
            source_ids_to_add_for_this_instance.add(n_single_orig_ing['ingredient_id'])
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (mask_idx {original_mask_idx}): N_single match to N5K ID '{final_id_for_instance}' ('{final_name_for_instance}') from dish.")
        else: # N_multi case
            # Identify N_principal (e.g., highest weight in original dish)
            # Weights are floats, handle potential ties if any specific rule needed. For now, first one with max weight.
            n_principal_orig_ing = max(matching_original_n5k_ingredients_in_dish, key=lambda x: x['weight_g'])
            final_id_for_instance = n_principal_orig_ing['ingredient_id']
            final_name_for_instance = n_principal_orig_ing['ingredient_name']
            for orig_ing in matching_original_n5k_ingredients_in_dish: # All in N_multi contribute
                source_ids_to_add_for_this_instance.add(orig_ing['ingredient_id'])
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (mask_idx {original_mask_idx}): N_multi match. Principal N5K ID '{final_id_for_instance}' ('{final_name_for_instance}'). Sources: {source_ids_to_add_for_this_instance}")
        
        if final_id_for_instance:
            instance_to_final_n5k_assignment[original_mask_idx] = {'id': final_id_for_instance, 'name': final_name_for_instance}
            
            # Update final_n5k_ingredient_definitions
            if final_id_for_instance not in final_n5k_ingredient_definitions:
                final_n5k_ingredient_definitions[final_id_for_instance] = {
                    'id': final_id_for_instance, 
                    'name': final_name_for_instance, 
                    'source_original_n5k_ingredient_ids': set()
                }
            final_n5k_ingredient_definitions[final_id_for_instance]['source_original_n5k_ingredient_ids'].update(source_ids_to_add_for_this_instance)

    # --- Construct the final ingredient list for JSON by summing up from original N5K data ---
    final_ingredients_for_json_list = []
    logging.debug(f"[{dish_id}] Final N5K ingredient definitions (pre-summing): {final_n5k_ingredient_definitions}")
    for final_id, definition_data in final_n5k_ingredient_definitions.items():
        current_final_ing_json = {
            'id': final_id, 
            'name': definition_data['name'],
            'weight_g': 0.0, 'calories_kcal': 0.0, 'fat_g': 0.0, 'carbs_g': 0.0, 'protein_g': 0.0
        }
        unique_original_ids_contributing = definition_data['source_original_n5k_ingredient_ids']
        
        for orig_ing_id in unique_original_ids_contributing:
            original_ing_details = original_n5k_ingredients_by_id.get(orig_ing_id)
            if original_ing_details:
                current_final_ing_json['weight_g'] += original_ing_details['weight_g']
                current_final_ing_json['calories_kcal'] += original_ing_details['calories_kcal']
                current_final_ing_json['fat_g'] += original_ing_details['fat_g']
                current_final_ing_json['carbs_g'] += original_ing_details['carbs_g']
                current_final_ing_json['protein_g'] += original_ing_details['protein_g']
            else:
                logging.warning(f"[{dish_id}] Original N5K ingredient ID {orig_ing_id} not found while summing for final ingredient {final_id}. This shouldn't happen.")
        
        # Round to reasonable precision, e.g., 2 decimal places for grams, 1 for kcal
        for key in ['weight_g', 'fat_g', 'carbs_g', 'protein_g']:
            current_final_ing_json[key] = round(current_final_ing_json[key], 2)
        current_final_ing_json['calories_kcal'] = round(current_final_ing_json['calories_kcal'], 1)

        final_ingredients_for_json_list.append(current_final_ing_json)

    # --- Calculate total nutritional values for the aligned/merged dish ---
    merged_totals = {'weight_g': 0.0, 'calories_kcal': 0.0, 'fat_g': 0.0, 'carbs_g': 0.0, 'protein_g': 0.0}
    for ing in final_ingredients_for_json_list:
        merged_totals['weight_g'] += ing['weight_g']
        merged_totals['calories_kcal'] += ing['calories_kcal']
        merged_totals['fat_g'] += ing['fat_g']
        merged_totals['carbs_g'] += ing['carbs_g']
        merged_totals['protein_g'] += ing['protein_g']
    
    for key in ['weight_g', 'fat_g', 'carbs_g', 'protein_g']:
        merged_totals[key] = round(merged_totals[key], 2)
    merged_totals['calories_kcal'] = round(merged_totals['calories_kcal'], 1)

    # --- Phase 4: Calculate Scores ---
    # Nutrients: M, K, L, G, P (Mass, Kcal, Lipides(Fat), Glucides(Carbs), Proteines)
    # Original totals:
    orig_M = original_n5k_metadata.get('dish_total_mass_g', 0.0)
    orig_K = original_n5k_metadata.get('dish_total_calories_kcal', 0.0)
    orig_L = original_n5k_metadata.get('dish_total_fat_g', 0.0)
    orig_G = original_n5k_metadata.get('dish_total_carbs_g', 0.0)
    orig_P = original_n5k_metadata.get('dish_total_protein_g', 0.0)

    # Merged totals:
    merged_M = merged_totals['weight_g']
    merged_K = merged_totals['calories_kcal']
    merged_L = merged_totals['fat_g']
    merged_G = merged_totals['carbs_g']
    merged_P = merged_totals['protein_g']

    nutrients_comparison = [
        (merged_M, orig_M), (merged_K, orig_K), (merged_L, orig_L),
        (merged_G, orig_G), (merged_P, orig_P)
    ]
    squared_relative_errors = []
    for merged_val, orig_val in nutrients_comparison:
        if orig_val != 0:
            re = (merged_val - orig_val) / orig_val
        else: # orig_val == 0
            if merged_val == 0:
                re = 0.0
            else: # orig_val is 0, merged_val is not -> penalty
                re = RE_X_ORIG_ZERO_MERGED_NON_ZERO_PENALTY 
        squared_relative_errors.append(re**2)
    
    msre_nutritional = sum(squared_relative_errors) / len(squared_relative_errors) if squared_relative_errors else 0.0
    score_nutr = math.exp(-SCORE_K_MSRE * msre_nutritional)

    # Ingredient Score (Jaccard Index)
    set_ingr_merged = {ing['id'] for ing in final_ingredients_for_json_list}
    set_ingr_orig = {ing['ingredient_id'] for ing in original_n5k_metadata['ingredients']}
    
    intersection_len = len(set_ingr_merged.intersection(set_ingr_orig))
    union_len = len(set_ingr_merged.union(set_ingr_orig))
    jaccard_ingredient = intersection_len / union_len if union_len > 0 else 0.0
    score_ingr = jaccard_ingredient # Jaccard is already 0-1

    # Final Confidence Score
    final_confidence = (SCORE_W_NUTR * score_nutr) + (SCORE_W_INGR * score_ingr)

    # --- Assemble final metadata.json content ---
    aligned_metadata_content = {
        "dish_id": dish_id,
        "ingredients": final_ingredients_for_json_list,
        "total_nutritional_values": merged_totals,
        "confidence_scores": {
            "final_confidence": round(final_confidence, 4),
            "nutritional_score": round(score_nutr, 4),
            "ingredient_score": round(score_ingr, 4),
            "msre_nutritional": round(msre_nutritional, 4),
            "jaccard_ingredient": round(jaccard_ingredient, 4)
        }
    }
    logging.info(f"[{dish_id}] Generated aligned N5K metadata with scores. Ingredients count: {len(final_ingredients_for_json_list)}. Final confidence: {final_confidence:.4f}")
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