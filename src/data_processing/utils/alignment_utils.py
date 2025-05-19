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

    # Helper to quickly look up original N5K ingredients by their N5K ID (string)
    # Assumes original_n5k_metadata['ingredients'] now has 'ingredient_id' and 'ingredient_name'
    original_n5k_ingredients_by_id = {str(ing['ingredient_id']): ing for ing in original_n5k_metadata['ingredients']}
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_by_id.keys())}")
    
    # Dictionary to store the final N5K ID and name assigned to each FoodSAM instance mask
    # Key: original_mask_index from FoodSAM instance
    # Value: {'id': final_n5k_id, 'name': final_n5k_name}
    instance_to_final_n5k_assignment = {}

    # Stores aggregated nutritional info for newly defined N5K ingredients based on SAM instances
    # Key: final_n5k_id (from N_principal or N_single)
    # Value: { 'definition': (see metadata_merged format), 'contributing_original_n5k_ids': set() }
    final_n5k_ingredient_definitions = {}

    # --- Iterate through each FoodSAM detected instance ---
    for instance in detected_foodsam_instances: # foodsam_instances is the list from extract_foodsam_instances_with_masks
        foodsam_cat_id = instance['foodsam_category_id'] # This is the FoodSAM integer ID
        foodsam_cat_name = instance['foodsam_category_name']
        original_mask_idx = instance['original_mask_index']
        # mask_array = instance['mask'] # The boolean mask array itself

        instance_to_final_n5k_assignment[original_mask_idx] = None # Default

        if foodsam_cat_name == 'background' or foodsam_cat_id == 0: # Skip background
            logging.debug(f"[{dish_id}] Skipping FoodSAM background instance (FoodSAM ID {foodsam_cat_id}, original_mask_idx {original_mask_idx}).")
            continue

        logging.debug(f"[{dish_id}] Processing FoodSAM instance: ID={foodsam_cat_id}, Name='{foodsam_cat_name}', Original Mask Index={original_mask_idx}")

        # 1. Find the mapping entry for the current FoodSAM ID from foodsam_to_n5k_map
        # foodsam_to_n5k_map (passed as foodsam_name_to_n5k_names_map) is keyed by stringified FoodSAM ID.
        # Each value has 'foodsam_name' and 'mapped_n5k_categories' [{n5k_id, n5k_name}, ...]
        
        entry_data_for_foodsam_id = foodsam_name_to_n5k_names_map.get(str(foodsam_cat_id))
        
        list_of_candidate_n5k_info_from_mapping = []
        if entry_data_for_foodsam_id:
            if entry_data_for_foodsam_id.get('foodsam_name') == foodsam_cat_name:
                list_of_candidate_n5k_info_from_mapping = entry_data_for_foodsam_id.get('mapped_n5k_categories', [])
                logging.debug(f"[{dish_id}] Found mapping for FoodSAM ID {foodsam_cat_id} ('{foodsam_cat_name}'). Potential N5K mapped categories from JSON: {len(list_of_candidate_n5k_info_from_mapping)}")
            else:
                logging.warning(f"[{dish_id}] FoodSAM ID {foodsam_cat_id} found in mapping, but name mismatch: map has '{entry_data_for_foodsam_id.get('foodsam_name')}', instance has '{foodsam_cat_name}'. Treating as no direct mapping by ID.")
        else:
            # Fallback: if not found by ID, try to find by name (e.g. if mapping file structure is inconsistent)
            logging.debug(f"[{dish_id}] No direct mapping entry for FoodSAM ID {str(foodsam_cat_id)}. Trying fallback by name '{foodsam_cat_name}'.")
            found_by_name_fallback = False
            for _map_key, entry_val in foodsam_name_to_n5k_names_map.items(): # Iterate through all entries
                if entry_val.get('foodsam_name') == foodsam_cat_name:
                    list_of_candidate_n5k_info_from_mapping = entry_val.get('mapped_n5k_categories', [])
                    logging.debug(f"[{dish_id}] Fallback: Found mapping by FoodSAM name '{foodsam_cat_name}' (original map key was {_map_key}). Potential N5K mapped categories: {len(list_of_candidate_n5k_info_from_mapping)}")
                    found_by_name_fallback = True
                    break
            if not found_by_name_fallback:
                logging.warning(f"[{dish_id}] FoodSAM category '{foodsam_cat_name}' (ID: {foodsam_cat_id}, original_mask_idx {original_mask_idx}): No N5K mapping found in foodsam_to_n5k_map even after fallback. Skipping this instance.")
                continue # Skip to next FoodSAM instance

        if not list_of_candidate_n5k_info_from_mapping:
            logging.debug(f"[{dish_id}] FoodSAM category '{foodsam_cat_name}' (ID: {foodsam_cat_id}, original_mask_idx {original_mask_idx}): Mapping file lists no N5K categories for it. Skipping instance.")
            continue

        # 2. Filter these mapped N5K categories to those actually present in the *current dish's* original N5K metadata
        actual_n5k_ingredients_in_dish_matching_foodsam_detection = []
        for n5k_candidate_info in list_of_candidate_n5k_info_from_mapping:
            candidate_n5k_id_from_mapping = str(n5k_candidate_info.get('n5k_id')) # Ensure it's a string for dict lookup
            # Convert the mapping ID to the ground truth format (e.g., "448" -> "ingr_0000000448")
            candidate_n5k_id_formatted = f"ingr_{int(candidate_n5k_id_from_mapping):08d}"
            
            if candidate_n5k_id_formatted in original_n5k_ingredients_by_id:
                actual_n5k_ingredients_in_dish_matching_foodsam_detection.append(original_n5k_ingredients_by_id[candidate_n5k_id_formatted])
        
        if not actual_n5k_ingredients_in_dish_matching_foodsam_detection:
            candidate_names_str = ', '.join([f"'{c.get('n5k_name', 'Unknown')}' (ID: {c.get('n5k_id', 'N/A')})" for c in list_of_candidate_n5k_info_from_mapping])
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (ID: {foodsam_cat_id}, original_mask_idx {original_mask_idx}): The N5K categories from mapping ({candidate_names_str}) are not present in this dish's actual N5K ingredients. Skipping instance.")
            continue
        
        logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (ID: {foodsam_cat_id}, original_mask_idx {original_mask_idx}): Filtered to {len(actual_n5k_ingredients_in_dish_matching_foodsam_detection)} actual N5K ingredients present in the dish: {[ing['ingredient_name'] for ing in actual_n5k_ingredients_in_dish_matching_foodsam_detection]}")

        # 3. Now, apply N_single / N_multi logic using actual_n5k_ingredients_in_dish_matching_foodsam_detection
        
        final_id_for_instance_str = None
        final_name_for_instance = None
        # N5K IDs from original dish that contribute to this merged ingredient
        source_n5k_ids_for_this_instance = set() 

        if len(actual_n5k_ingredients_in_dish_matching_foodsam_detection) == 1: # N_single case
            n_single_orig_ing = actual_n5k_ingredients_in_dish_matching_foodsam_detection[0]
            final_id_for_instance_str = str(n_single_orig_ing['ingredient_id'])
            final_name_for_instance = n_single_orig_ing['ingredient_name']
            source_n5k_ids_for_this_instance.add(final_id_for_instance_str)
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (original_mask_idx {original_mask_idx}): N_single match to N5K ID '{final_id_for_instance_str}' ('{final_name_for_instance}') from dish.")
        else: # N_multi case
            # Identify N_principal (e.g., highest weight in original dish from the filtered list)
            # Ensure weight_g is treated as float for comparison
            try:
                n_principal_orig_ing = max(actual_n5k_ingredients_in_dish_matching_foodsam_detection, key=lambda x: float(x['weight_g']))
            except ValueError as e:
                logging.error(f"[{dish_id}] Error converting weight_g to float for N_multi principal selection. Dish ingredients: {actual_n5k_ingredients_in_dish_matching_foodsam_detection}. Error: {e}. Skipping this FoodSAM instance.")
                continue

            final_id_for_instance_str = str(n_principal_orig_ing['ingredient_id'])
            final_name_for_instance = n_principal_orig_ing['ingredient_name']
            for orig_ing in actual_n5k_ingredients_in_dish_matching_foodsam_detection: # All in N_multi contribute to source
                source_n5k_ids_for_this_instance.add(str(orig_ing['ingredient_id']))
            logging.debug(f"[{dish_id}] FoodSAM '{foodsam_cat_name}' (original_mask_idx {original_mask_idx}): N_multi match. Principal N5K ID '{final_id_for_instance_str}' ('{final_name_for_instance}'). Contributing source N5K IDs from dish: {source_n5k_ids_for_this_instance}")
        
        if final_id_for_instance_str:
            instance_to_final_n5k_assignment[original_mask_idx] = {'id': final_id_for_instance_str, 'name': final_name_for_instance}
            
            # Aggregate nutritional info for this final_id_for_instance_str
            if final_id_for_instance_str not in final_n5k_ingredient_definitions:
                # Find the original N5K ingredient that corresponds to final_id_for_instance_str to use as a base for name
                # (It must be one of the original_n5k_ingredients_by_id)
                base_n5k_ingredient_for_definition = original_n5k_ingredients_by_id.get(final_id_for_instance_str)
                if not base_n5k_ingredient_for_definition:
                    logging.error(f"[{dish_id}] CRITICAL: final_id_for_instance_str {final_id_for_instance_str} not found in original_n5k_ingredients_by_id. This should not happen. Skipping aggregation for this.")
                    continue

                final_n5k_ingredient_definitions[final_id_for_instance_str] = {
                    'ingredient_name': base_n5k_ingredient_for_definition['ingredient_name'], # Use N_Principal's name or N_Single's name
                    'weight_g': 0.0,
                    'calories_kcal': 0.0,
                    'fat_g': 0.0,
                    'carbs_g': 0.0,
                    'protein_g': 0.0,
                    'contributing_original_n5k_ids': set(), # Keep track of which original N5K items contributed to this
                    'n5k_id_for_semseg': n5k_name_to_id_map.get(base_n5k_ingredient_for_definition['ingredient_name']) # Store the ID used for semseg painting
                }
            
            # Add all contributing original N5K IDs from this instance's match
            final_n5k_ingredient_definitions[final_id_for_instance_str]['contributing_original_n5k_ids'].update(source_n5k_ids_for_this_instance)
            # Nutritional values will be summed up after all instances are processed, based on 'contributing_original_n5k_ids'

    # --- Post-process: Sum nutritional values for merged ingredients ---
    logging.debug(f"[{dish_id}] Final N5K ingredient definitions (pre-summing nutritional values): {final_n5k_ingredient_definitions}")
    
    merged_ingredients_list_for_json = []
    # Initialize accumulators for the dish totals *before* iterating through ingredients
    total_summed_weight_for_dish = 0.0
    total_summed_calories_for_dish = 0.0
    total_summed_fat_for_dish = 0.0
    total_summed_carbs_for_dish = 0.0
    total_summed_protein_for_dish = 0.0

    for final_n5k_id_str, definition_shell in final_n5k_ingredient_definitions.items():
        # These are for summing up contributions to *one* merged ingredient definition
        current_ing_summed_calories = 0.0
        current_ing_summed_weight = 0.0
        current_ing_summed_fat = 0.0
        current_ing_summed_carbs = 0.0
        current_ing_summed_protein = 0.0
        contributing_original_n5k_ids = definition_shell['contributing_original_n5k_ids']
        
        for orig_ing_id in contributing_original_n5k_ids:
            original_ing_details = original_n5k_ingredients_by_id.get(orig_ing_id)
            if original_ing_details:
                current_ing_summed_calories += original_ing_details['calories_kcal']
                current_ing_summed_weight += original_ing_details['weight_g']
                current_ing_summed_fat += original_ing_details['fat_g']
                current_ing_summed_carbs += original_ing_details['carbs_g']
                current_ing_summed_protein += original_ing_details['protein_g']
            else:
                logging.warning(f"[{dish_id}] Original N5K ingredient ID {orig_ing_id} not found while summing for final ingredient {final_n5k_id_str}. This shouldn't happen.")
        
        # Round to reasonable precision for the current merged ingredient
        current_ing_summed_calories = round(current_ing_summed_calories, 1)
        current_ing_summed_weight = round(current_ing_summed_weight, 2)
        current_ing_summed_fat = round(current_ing_summed_fat, 2)
        current_ing_summed_carbs = round(current_ing_summed_carbs, 2)
        current_ing_summed_protein = round(current_ing_summed_protein, 2)

        merged_ingredients_list_for_json.append({
            'id': final_n5k_id_str, # This is the N5K ID
            'name': definition_shell['ingredient_name'],
            'weight_g': current_ing_summed_weight,
            'calories_kcal': current_ing_summed_calories,
            'fat_g': current_ing_summed_fat,
            'carbs_g': current_ing_summed_carbs,
            'protein_g': current_ing_summed_protein
        })
        
        # Accumulate to dish totals
        total_summed_weight_for_dish += current_ing_summed_weight
        total_summed_calories_for_dish += current_ing_summed_calories
        total_summed_fat_for_dish += current_ing_summed_fat
        total_summed_carbs_for_dish += current_ing_summed_carbs
        total_summed_protein_for_dish += current_ing_summed_protein

    # --- Calculate total nutritional values for the aligned/merged dish ---
    # These are now the correctly summed totals for the entire dish from merged ingredients
    merged_totals = {
        'weight_g': round(total_summed_weight_for_dish, 2),
        'calories_kcal': round(total_summed_calories_for_dish, 1),
        'fat_g': round(total_summed_fat_for_dish, 2),
        'carbs_g': round(total_summed_carbs_for_dish, 2),
        'protein_g': round(total_summed_protein_for_dish, 2)
    }
    
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
    set_ingr_merged = {ing['id'] for ing in merged_ingredients_list_for_json}
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