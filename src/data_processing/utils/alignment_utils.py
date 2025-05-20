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
                                           foodsam_id_to_n5k_map, # Changed from foodsam_name_to_n5k_names_map
                                           n5k_name_to_id_map, n5k_id_to_name_map):
    """
    Generates N5K-aligned metadata.json content by matching FoodSAM instances to N5K ingredients
    within the dish's original metadata, applying fusion logic for multiple matches,
    and calculating confidence scores.

    Args:
        dish_id (str): The ID of the dish.
        original_n5k_metadata (dict): Parsed metadata from Nutrition5k for the dish.
        detected_foodsam_instances (list): List of dicts from extract_foodsam_instances_with_masks.
                                         Each dict has 'original_mask_index', 'foodsam_category_id', 
                                         'foodsam_category_name', 'instance_mask'.
        foodsam_id_to_n5k_map (dict): Mapping FoodSAM-103 category ID (str) to its N5K mappings.
                                      e.g., {"25": {"foodsam_name": "apple", "mapped_n5k_categories": [...]}}
        n5k_name_to_id_map (dict): Mapping N5K category name to N5K ID.
        n5k_id_to_name_map (dict): Mapping N5K ID to N5K name.

    Returns:
        tuple: (aligned_metadata_content_dict, instance_to_final_n5k_assignment_dict)
               Returns (None, None) on critical error.
    """
    if not original_n5k_metadata or not original_n5k_metadata.get('ingredients'):
        logging.warning(f"[{dish_id}] Original N5K metadata is empty or has no ingredients. Cannot align.")
        # Return structure that indicates 0 ingredients but allows semseg/bbox to be empty.
        empty_metadata = {
            "dish_id": dish_id, "ingredients": [], "total_nutritional_values": {},
            "confidence_scores": {"final_confidence": 0, "nutritional_score": 0, "ingredient_score": 0}
        }
        return empty_metadata, {} # Empty assignment map

    original_n5k_ingredients_in_dish_by_id = {
        str(ing['ingredient_id']): ing for ing in original_n5k_metadata['ingredients']
    }
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_in_dish_by_id.keys())}")

    instance_to_final_n5k_assignment = {}  # Maps original_mask_index to final N5K ID (str, e.g., "ingr_...")
    # Using N5K ID 0 for background/unmatched for visual modalities
    # For metadata, we collect contributions to actual ingredients.

    # Intermediate structure to hold contributions before consolidation
    # Key: Chosen N5K ingredient ID (e.g., "ingr_0000000054")
    # Value: { 'id': str, 'name': str, 'contributing_original_n5k_ingredients': [list of original N5K ingredient dicts], 'mask_indices': [list of original_mask_index] }
    final_n5k_ingredient_definitions = {}

    if not detected_foodsam_instances:
        logging.warning(f"[{dish_id}] No FoodSAM instances detected. Proceeding with empty alignment for metadata.")
    
    for foodsam_instance in detected_foodsam_instances:
        original_mask_idx = foodsam_instance['original_mask_index']
        foodsam_cat_id_int = foodsam_instance['foodsam_category_id']
        foodsam_cat_id_str = str(foodsam_cat_id_int)
        foodsam_cat_name = foodsam_instance['foodsam_category_name']

        # Default assignment for the mask is background (N5K ID 0)
        # Note: For n5k_semseg_from_sam, we need integer IDs.
        # The 'id' in instance_to_final_n5k_assignment should be the N5K integer ID or a special non-integer string.
        # Let's clarify: instance_to_final_n5k_assignment maps original_mask_index -> {'id': n5k_int_id_or_special_str, 'name': n5k_name}
        # For now, let's assign the integer N5K ID 0 for "background" or "unmatched".
        # If a non-integer N5K ID like "plate-only" is used later, generate_n5k_semseg_from_sam handles it.
        instance_to_final_n5k_assignment[original_mask_idx] = {'id': 0, 'name': 'background'}


        if foodsam_cat_id_int == 0: # Skip FoodSAM background
            logging.debug(f"[{dish_id}] Skipping FoodSAM background instance (mask_idx: {original_mask_idx}).")
            continue

        mapping_entry_for_foodsam_id = foodsam_id_to_n5k_map.get(foodsam_cat_id_str)
        if not mapping_entry_for_foodsam_id:
            logging.debug(f"[{dish_id}] No entry in foodsam_id_to_n5k_map for FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}'). Mask idx {original_mask_idx} remains background.")
            continue

        potential_n5k_mappings = mapping_entry_for_foodsam_id.get('mapped_n5k_categories', [])
        if not potential_n5k_mappings:
            logging.debug(f"[{dish_id}] FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}') has no 'mapped_n5k_categories' in map. Mask idx {original_mask_idx} remains background.")
            continue

        dish_specific_n5k_matches = []
        for n5k_candidate_from_mapping in potential_n5k_mappings:
            n5k_id_value_from_map = n5k_candidate_from_mapping.get('n5k_id') # Get the raw value
            # logging.debug(f"[{dish_id}] --- Processing candidate N5K mapping for FoodSAM {foodsam_cat_name}({foodsam_cat_id_str}): {n5k_candidate_from_mapping}. Raw n5k_id from map: '{n5k_id_value_from_map}' (type: {type(n5k_id_value_from_map)})")

            if n5k_id_value_from_map is None:
                logging.warning(f"[{dish_id}] --- Found a mapping candidate for FoodSAM ID {foodsam_cat_id_str} with missing 'n5k_id': {n5k_candidate_from_mapping}. Skipping this candidate.")
                continue 

            n5k_id_from_candidate_map_str = str(n5k_id_value_from_map) 
            
            try:
                n5k_id_formatted_for_dish_lookup = f"ingr_{int(n5k_id_from_candidate_map_str):08d}"
            except ValueError:
                logging.warning(f"[{dish_id}] --- Could not convert n5k_id '{n5k_id_from_candidate_map_str}' to int for formatting. Candidate: {n5k_candidate_from_mapping}. Skipping this candidate.")
                continue 

            # logging.debug(f"[{dish_id}] --- Attempting to match formatted N5K ID: '{n5k_id_formatted_for_dish_lookup}'")
            # For detailed comparison, show available keys if match fails on a specific item
            # original_dish_ingredient_keys = list(original_n5k_ingredients_in_dish_by_id.keys())
            # logging.debug(f"[{dish_id}] --- Available original N5K ingredient IDs in dish: {original_dish_ingredient_keys}")
            
            is_present = n5k_id_formatted_for_dish_lookup in original_n5k_ingredients_in_dish_by_id
            # logging.debug(f"[{dish_id}] --- Is '{n5k_id_formatted_for_dish_lookup}' present in original dish ingredients? {is_present}")

            if is_present:
                dish_specific_n5k_matches.append(original_n5k_ingredients_in_dish_by_id[n5k_id_formatted_for_dish_lookup])
        
        logging.debug(f"[{dish_id}] FoodSAM inst (mask {original_mask_idx}, cat: {foodsam_cat_name} ID:{foodsam_cat_id_str}) found {len(dish_specific_n5k_matches)} specific N5K ingredients in current dish: {[m['ingredient_name'] for m in dish_specific_n5k_matches]}")

        chosen_n5k_ingredient_for_instance = None # This will be the N5K ingredient object (dict) from original_n5k_metadata
        contributing_ingredients_for_definition = [] # List of original N5K ingredient dicts

        if not dish_specific_n5k_matches:
            # Case 2.d.i: No Match in Dish. Instance mask already assigned to background N5K ID 0.
            logging.debug(f"[{dish_id}] Instance (mask {original_mask_idx}, FoodSAM cat {foodsam_cat_name}) found no N5K match in current dish's ingredients.")
            # instance_to_final_n5k_assignment already set to background for this mask_idx
        
        elif len(dish_specific_n5k_matches) == 1:
            # Case 2.d.ii: One N5K Match from Dish (N_single)
            N_single = dish_specific_n5k_matches[0]
            chosen_n5k_ingredient_for_instance = N_single
            contributing_ingredients_for_definition = [N_single]
            logging.debug(f"[{dish_id}] Instance (mask {original_mask_idx}, FoodSAM cat {foodsam_cat_name}) -> ONE N5K match in dish: {N_single['ingredient_name']} ({N_single['ingredient_id']})")
        
        else: # len(dish_specific_n5k_matches) > 1
            # Case 2.d.iii: Multiple N5K Matches from Dish (N_multi)
            N_multi = dish_specific_n5k_matches
            # Select N_principal (highest weight, then first in list as tie-breaker)
            N_principal = sorted(N_multi, key=lambda x: x.get('weight_g', 0), reverse=True)[0]
            chosen_n5k_ingredient_for_instance = N_principal
            contributing_ingredients_for_definition = N_multi # All of N_multi contribute to the fused definition
            logging.debug(f"[{dish_id}] Instance (mask {original_mask_idx}, FoodSAM cat {foodsam_cat_name}) -> MULTIPLE N5K matches in dish. Principal: {N_principal['ingredient_name']} ({N_principal['ingredient_id']}). Fusion with {len(N_multi)} ingredients.")

        # If a N5K ingredient was chosen for this instance (either N_single or N_principal)
        if chosen_n5k_ingredient_for_instance:
            final_n5k_id_for_instance = str(chosen_n5k_ingredient_for_instance['ingredient_id']) # e.g., "ingr_0000000054"
            final_n5k_name_for_instance = chosen_n5k_ingredient_for_instance['ingredient_name']
            
            # Update assignment for visual modalities (semseg, bbox)
            # The ID here should be the N5K integer ID.
            try:
                # Extract pure integer ID from "ingr_000000xxxx"
                n5k_int_id = int(final_n5k_id_for_instance.split('_')[-1])
                instance_to_final_n5k_assignment[original_mask_idx] = {'id': n5k_int_id, 'name': final_n5k_name_for_instance}
            except ValueError:
                logging.error(f"[{dish_id}] Could not parse integer N5K ID from {final_n5k_id_for_instance} for mask {original_mask_idx}. Assigning background.")
                instance_to_final_n5k_assignment[original_mask_idx] = {'id': 0, 'name': 'background_parse_error'}


            # Aggregate for final metadata list
            if final_n5k_id_for_instance not in final_n5k_ingredient_definitions:
                final_n5k_ingredient_definitions[final_n5k_id_for_instance] = {
                    'id': final_n5k_id_for_instance, # This is the ground truth N5K ID, e.g. "ingr_0000000054"
                    'name': final_n5k_name_for_instance, # Name of N_single or N_principal
                    'contributing_original_n5k_ingredients': [],
                    'mask_indices': []
                }
            
            # Add the original N5K ingredients that form this definition
            # Ensure each original ingredient is added only once per definition, even if multiple instances map to it.
            # The contributing_ingredients_for_definition are specific to THIS FoodSAM instance's match resolution.
            # If another FoodSAM instance also maps to the SAME final_n5k_id_for_instance, it might have its own set of contributing_ingredients_for_definition (e.g. if it also resulted in a fusion).
            # The goal is that final_n5k_ingredient_definitions[final_n5k_id].contributing_original_n5k_ingredients should be the union of all such lists for that final_n5k_id.
            
            current_contributors = final_n5k_ingredient_definitions[final_n5k_id_for_instance]['contributing_original_n5k_ingredients']
            for orig_ing in contributing_ingredients_for_definition:
                # Avoid duplicates if multiple instances map to the same principal and contribute the same set from N_multi
                if not any(existing_ing['ingredient_id'] == orig_ing['ingredient_id'] for existing_ing in current_contributors):
                    current_contributors.append(orig_ing)
            
            final_n5k_ingredient_definitions[final_n5k_id_for_instance]['mask_indices'].append(original_mask_idx)

    # --- Consolidate Final Ingredients for metadata.json ---
    final_ingredients_for_json_list = []
    if not final_n5k_ingredient_definitions:
        logging.info(f"[{dish_id}] No FoodSAM instances were successfully mapped to any N5K ingredients in the dish. Final ingredient list will be empty.")
    
    for n5k_final_id_str, definition_details in final_n5k_ingredient_definitions.items():
        summed_weight = sum(ing.get('weight_g', 0) for ing in definition_details['contributing_original_n5k_ingredients'])
        summed_calories = sum(ing.get('calories_kcal', 0) for ing in definition_details['contributing_original_n5k_ingredients'])
        summed_fat = sum(ing.get('fat_g', 0) for ing in definition_details['contributing_original_n5k_ingredients'])
        summed_carbs = sum(ing.get('carbs_g', 0) for ing in definition_details['contributing_original_n5k_ingredients'])
        summed_protein = sum(ing.get('protein_g', 0) for ing in definition_details['contributing_original_n5k_ingredients'])

        # The ID and name for the final JSON list come from the definition (which was based on N_single or N_principal)
        final_ingredients_for_json_list.append({
            'id': definition_details['id'], # e.g. "ingr_0000000054" (the N5K ground truth ID)
            'name': definition_details['name'],
            'weight_g': round(summed_weight, 2),
            'calories_kcal': round(summed_calories, 1),
            'fat_g': round(summed_fat, 2),
            'carbs_g': round(summed_carbs, 2),
            'protein_g': round(summed_protein, 2),
            'source_original_n5k_ingredient_ids': [ing['ingredient_id'] for ing in definition_details['contributing_original_n5k_ingredients']],
            'mapped_foodsam_mask_indices': definition_details['mask_indices']
        })

    # --- Calculate Totals & Confidence Scores ---
    merged_totals = {
        'weight_g': round(sum(ing['weight_g'] for ing in final_ingredients_for_json_list), 2),
        'calories_kcal': round(sum(ing['calories_kcal'] for ing in final_ingredients_for_json_list), 1),
        'fat_g': round(sum(ing['fat_g'] for ing in final_ingredients_for_json_list), 2),
        'carbs_g': round(sum(ing['carbs_g'] for ing in final_ingredients_for_json_list), 2),
        'protein_g': round(sum(ing['protein_g'] for ing in final_ingredients_for_json_list), 2)
    }

    orig_M = original_n5k_metadata.get('dish_total_mass_g', 0.0)
    orig_K = original_n5k_metadata.get('dish_total_calories_kcal', 0.0)
    orig_L = original_n5k_metadata.get('dish_total_fat_g', 0.0)
    orig_G = original_n5k_metadata.get('dish_total_carbs_g', 0.0)
    orig_P = original_n5k_metadata.get('dish_total_protein_g', 0.0)

    nutrients_comparison = [
        (merged_totals.get('weight_g', 0), orig_M),
        (merged_totals.get('calories_kcal', 0), orig_K),
        (merged_totals.get('fat_g', 0), orig_L),
        (merged_totals.get('carbs_g', 0), orig_G),
        (merged_totals.get('protein_g', 0), orig_P)
    ]
    
    squared_relative_errors = []
    for merged_val, orig_val in nutrients_comparison:
        if orig_val != 0:
            re = (merged_val - orig_val) / orig_val
        else:
            re = RE_X_ORIG_ZERO_MERGED_NON_ZERO_PENALTY if merged_val != 0 else 0.0
        squared_relative_errors.append(re**2)
    
    msre_nutritional = sum(squared_relative_errors) / len(squared_relative_errors) if squared_relative_errors else 0.0
    score_nutr = math.exp(-SCORE_K_MSRE * msre_nutritional)

    set_ingr_merged_ids = {ing['id'] for ing in final_ingredients_for_json_list}
    set_ingr_orig_ids = {str(ing['ingredient_id']) for ing in original_n5k_metadata.get('ingredients', [])}
    
    intersection_len = len(set_ingr_merged_ids.intersection(set_ingr_orig_ids))
    union_len = len(set_ingr_merged_ids.union(set_ingr_orig_ids))
    jaccard_ingredient = intersection_len / union_len if union_len > 0 else (1.0 if not set_ingr_merged_ids and not set_ingr_orig_ids else 0.0) # Both empty = perfect match
    score_ingr = jaccard_ingredient

    final_confidence = (SCORE_W_NUTR * score_nutr) + (SCORE_W_INGR * score_ingr)

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

    logging.info(f"[{dish_id}] Generated aligned N5K metadata. Ingredients count: {len(final_ingredients_for_json_list)}. Final confidence: {final_confidence:.4f}")
    # Log details of instance_to_final_n5k_assignment if it's small or for a sample
    if len(instance_to_final_n5k_assignment) < 10:
        logging.debug(f"[{dish_id}] Instance to N5K assignment map: {instance_to_final_n5k_assignment}")
    else:
        logging.debug(f"[{dish_id}] Instance to N5K assignment map contains {len(instance_to_final_n5k_assignment)} entries.")

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