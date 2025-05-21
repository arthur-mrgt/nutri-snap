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
                                           foodsam_id_to_n5k_map,
                                           n5k_name_to_id_map, n5k_id_to_name_map):
    """
    Generates N5K-aligned metadata.json content by:
    1. Identifying all potential mappings (claims) between FoodSAM instances and N5K ingredients in the dish, ranked by preference.
    2. Resolving these claims so each N5K ingredient in the dish is claimed by at most one FoodSAM instance (the one with the best rank for it).
    3. Allowing a FoodSAM instance to claim multiple N5K ingredients. These form a group.
    4. For each group, determining a principal N5K ingredient (by weight).
    5. Summing nutritional data for all uniquely claimed N5K ingredients within their respective groups.
    6. Calculating confidence scores.
    """
    if not original_n5k_metadata or not original_n5k_metadata.get('ingredients'):
        logging.warning(f"[{dish_id}] Original N5K metadata is empty or has no ingredients. Cannot align.")
        empty_metadata = {
            "dish_id": dish_id, "ingredients": [], "total_nutritional_values": {},
            "confidence_scores": {"final_confidence": 0, "nutritional_score": 0, "ingredient_score": 0}
        }
        return empty_metadata, {}

    original_n5k_ingredients_in_dish_by_id = {
        str(ing['ingredient_id']): ing for ing in original_n5k_metadata['ingredients']
    }
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_in_dish_by_id.keys())}")

    # --- Phase 2: Build All Potential Claims with Ranks ---
    potential_claims = [] # List of tuples: (rank, foodsam_original_mask_idx, foodsam_cat_name, foodsam_cat_id, original_n5k_ingredient_object_from_dish)
    
    for foodsam_instance in detected_foodsam_instances:
        original_mask_idx = foodsam_instance['original_mask_index']
        foodsam_cat_id_int = foodsam_instance['foodsam_category_id']
        foodsam_cat_name = foodsam_instance['foodsam_category_name']

        if foodsam_cat_id_int == 0: # Skip FoodSAM background
            logging.debug(f"[{dish_id}] Skipping FoodSAM background instance (mask_idx: {original_mask_idx}) for claim building.")
            continue

        foodsam_cat_id_str = str(foodsam_cat_id_int)
        mapping_entry_for_foodsam_id = foodsam_id_to_n5k_map.get(foodsam_cat_id_str)

        if not mapping_entry_for_foodsam_id:
            logging.debug(f"[{dish_id}] No entry in foodsam_id_to_n5k_map for FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}'). No claims generated for mask idx {original_mask_idx}.")
            continue
        
        mapped_n5k_categories = mapping_entry_for_foodsam_id.get('mapped_n5k_categories', [])
        if not mapped_n5k_categories:
            logging.debug(f"[{dish_id}] FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}') has no 'mapped_n5k_categories' in map. No claims generated for mask idx {original_mask_idx}.")
            continue

        for rank, n5k_candidate_from_mapping in enumerate(mapped_n5k_categories):
            n5k_id_value_from_map = n5k_candidate_from_mapping.get('n5k_id')
            if n5k_id_value_from_map is None:
                logging.warning(f"[{dish_id}] --- Found a mapping candidate for FoodSAM ID {foodsam_cat_id_str} with missing 'n5k_id': {n5k_candidate_from_mapping}. Skipping this candidate claim.")
                continue
            
            n5k_id_from_candidate_map_str = str(n5k_id_value_from_map)
            try:
                n5k_id_formatted_for_dish_lookup = f"ingr_{int(n5k_id_from_candidate_map_str):010d}"
            except ValueError:
                logging.warning(f"[{dish_id}] --- Could not convert n5k_id '{n5k_id_from_candidate_map_str}' to int for formatting. Candidate: {n5k_candidate_from_mapping}. Skipping this candidate claim.")
                continue

            if n5k_id_formatted_for_dish_lookup in original_n5k_ingredients_in_dish_by_id:
                original_n5k_ingredient_object = original_n5k_ingredients_in_dish_by_id[n5k_id_formatted_for_dish_lookup]
                potential_claims.append((rank, original_mask_idx, foodsam_cat_name, foodsam_cat_id_int, original_n5k_ingredient_object))
                logging.debug(f"[{dish_id}] Potential claim: Rank {rank}, Mask {original_mask_idx} (FS: {foodsam_cat_name}), N5K: {original_n5k_ingredient_object['ingredient_name']} ({original_n5k_ingredient_object['ingredient_id']})")

    # --- Phase 3: Resolve Claims and Populate foodsam_instance_claims ---
    # Sort by original_n5k_ingredient_id first, then by rank.
    # This groups all claims for the same N5K ingredient, with the best rank appearing first.
    potential_claims.sort(key=lambda x: (x[4]['ingredient_id'], x[0]))

    foodsam_instance_claims = {fs_inst['original_mask_index']: [] for fs_inst in detected_foodsam_instances}
    claimed_original_n5k_ids = set()

    logging.debug(f"[{dish_id}] Starting claim resolution. Total potential claims: {len(potential_claims)}")
    for rank, mask_idx, fs_cat_name, fs_cat_id, n5k_obj in potential_claims:
        original_n5k_ingr_id = n5k_obj['ingredient_id']
        if original_n5k_ingr_id not in claimed_original_n5k_ids:
            foodsam_instance_claims[mask_idx].append(n5k_obj)
            claimed_original_n5k_ids.add(original_n5k_ingr_id)
            logging.debug(f"[{dish_id}] Claimed: N5K '{n5k_obj['ingredient_name']}' ({original_n5k_ingr_id}) by FoodSAM instance mask {mask_idx} (FS: {fs_cat_name}, Rank {rank})")
        else:
            logging.debug(f"[{dish_id}] Skipped claim: N5K '{n5k_obj['ingredient_name']}' ({original_n5k_ingr_id}) already claimed. Attempt by mask {mask_idx} (FS: {fs_cat_name}, Rank {rank})")
    
    # --- Phase 4: Consolidate Claims into Final Ingredient Definitions ---
    final_n5k_ingredient_definitions = {} # Key: Principal N5K ID, Value: dict of details
    instance_to_final_n5k_assignment = {inst['original_mask_index']: {'id': 0, 'name': 'background'} for inst in detected_foodsam_instances}

    for mask_idx, list_of_claimed_n5k_objects in foodsam_instance_claims.items():
        if not list_of_claimed_n5k_objects:
            # This FoodSAM instance claimed nothing, visual assignment remains background (already set)
            continue

        # Determine Principal N5K ingredient for this FoodSAM instance's claimed group (by weight)
        N_principal_obj = sorted(list_of_claimed_n5k_objects, key=lambda x: x.get('weight_g', 0), reverse=True)[0]
        
        current_group_final_id = N_principal_obj['ingredient_id']
        current_group_final_name = N_principal_obj['ingredient_name']

        # Update visual assignment for this mask_idx
        try:
            n5k_int_id = int(current_group_final_id.split('_')[-1])
            instance_to_final_n5k_assignment[mask_idx] = {'id': n5k_int_id, 'name': current_group_final_name}
        except ValueError:
            logging.error(f"[{dish_id}] Could not parse integer N5K ID from {current_group_final_id} for mask {mask_idx} during visual assignment. Assigning background.")
            # instance_to_final_n5k_assignment already defaults to background
            
        # Consolidate into final_n5k_ingredient_definitions
        if current_group_final_id not in final_n5k_ingredient_definitions:
            final_n5k_ingredient_definitions[current_group_final_id] = {
                'id': current_group_final_id,
                'name': current_group_final_name,
                'contributing_original_n5k_objects_list': [],
                'mapped_foodsam_mask_indices': set()
            }
        
        # Add this FoodSAM instance's claimed N5K objects to the definition
        # Ensure each original N5K object is added only once to the list for the principal.
        # (This is guaranteed because claimed_original_n5k_ids ensures an object is only claimed once across all FoodSAM instances)
        for n5k_obj in list_of_claimed_n5k_objects:
            # Check if this n5k_obj (by id) is already in the list for this principal
            # This check is vital if multiple FoodSAM instances could theoretically point to the same principal *and* share some sub-ingredients
            # However, with the current claiming logic, each n5k_obj in list_of_claimed_n5k_objects is unique to *this* FoodSAM instance's direct claims.
            # The merging happens if *different* FoodSAM instances resolve to the *same* principal ID.
            
            # Let's simplify: the list_of_claimed_n5k_objects for *this mask_idx* are all its unique claims.
            # These all contribute to the definition headed by current_group_final_id.
            # We need to merge these into the global definition for current_group_final_id.
            target_def_list = final_n5k_ingredient_definitions[current_group_final_id]['contributing_original_n5k_objects_list']
            if not any(existing_obj['ingredient_id'] == n5k_obj['ingredient_id'] for existing_obj in target_def_list):
                target_def_list.append(n5k_obj)
        
        final_n5k_ingredient_definitions[current_group_final_id]['mapped_foodsam_mask_indices'].add(mask_idx)

    # --- Phase 5: Generate Final JSON List and Calculate Scores ---
    final_ingredients_for_json_list = []
    if not final_n5k_ingredient_definitions:
        logging.info(f"[{dish_id}] No N5K ingredients were successfully claimed and processed for the final metadata list.")

    for final_id_str, definition_details in final_n5k_ingredient_definitions.items():
        summed_weight = sum(ing.get('weight_g', 0) for ing in definition_details['contributing_original_n5k_objects_list'])
        summed_calories = sum(ing.get('calories_kcal', 0) for ing in definition_details['contributing_original_n5k_objects_list'])
        summed_fat = sum(ing.get('fat_g', 0) for ing in definition_details['contributing_original_n5k_objects_list'])
        summed_carbs = sum(ing.get('carbs_g', 0) for ing in definition_details['contributing_original_n5k_objects_list'])
        summed_protein = sum(ing.get('protein_g', 0) for ing in definition_details['contributing_original_n5k_objects_list'])

        final_ingredients_for_json_list.append({
            'id': definition_details['id'],
            'name': definition_details['name'],
            'weight_g': round(summed_weight, 2),
            'calories_kcal': round(summed_calories, 1),
            'fat_g': round(summed_fat, 2),
            'carbs_g': round(summed_carbs, 2),
            'protein_g': round(summed_protein, 2),
            'source_original_n5k_ingredient_ids': sorted(list(set(ing['ingredient_id'] for ing in definition_details['contributing_original_n5k_objects_list']))),
            'mapped_foodsam_mask_indices': sorted(list(definition_details['mapped_foodsam_mask_indices']))
        })
    
    # Calculate Totals & Confidence Scores (this part remains the same)
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
            # If original is 0, RE is 1 if merged is non-zero (penalty), or 0 if merged is also 0.
            re = RE_X_ORIG_ZERO_MERGED_NON_ZERO_PENALTY if merged_val != 0 else 0.0
        squared_relative_errors.append(re**2)
    
    msre_nutritional = sum(squared_relative_errors) / len(squared_relative_errors) if squared_relative_errors else 0.0
    score_nutr = math.exp(-SCORE_K_MSRE * msre_nutritional)

    set_ingr_merged_ids = {ing['id'] for ing in final_ingredients_for_json_list} # IDs of the principal ingredients
    set_ingr_orig_ids = {str(ing['ingredient_id']) for ing in original_n5k_metadata.get('ingredients', [])}
    
    # Jaccard index based on the principal ingredient IDs identified vs original ingredient IDs
    intersection_len = len(set_ingr_merged_ids.intersection(set_ingr_orig_ids))
    union_len = len(set_ingr_merged_ids.union(set_ingr_orig_ids))
    
    # If both sets are empty (e.g., empty dish, and we found nothing), it's a perfect match (Jaccard=1)
    # If one is empty and the other is not, Jaccard is 0 (unless union is 0, handled by 'else 0.0')
    jaccard_ingredient = intersection_len / union_len if union_len > 0 else (1.0 if not set_ingr_merged_ids and not set_ingr_orig_ids else 0.0)
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

    logging.info(f"[{dish_id}] Generated aligned N5K metadata (New Strategy). Ingredients count: {len(final_ingredients_for_json_list)}. Final confidence: {final_confidence:.4f}")
    if len(instance_to_final_n5k_assignment) < 10 or not instance_to_final_n5k_assignment:
        logging.debug(f"[{dish_id}] Instance to N5K assignment map: {instance_to_final_n5k_assignment}")
    else:
        logging.debug(f"[{dish_id}] Instance to N5K assignment map contains {len(instance_to_final_n5k_assignment)} entries.")

    # Remove old verbose logging that is no longer relevant
    # The new logging for claim building and resolution should provide better insights.
    # Previous logging loop for n5k_candidate_from_mapping is removed by this refactor.

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