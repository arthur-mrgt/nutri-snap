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


def generate_aligned_n5k_metadata_with_scores(dish_id, original_n5k_metadata, foodsam_category_observations,
                                           foodsam_id_to_n5k_map,
                                           n5k_string_id_to_int_id_map, # New: for final mapping
                                           n5k_id_to_name_map): # Used for names, assuming it maps string N5K ID to name
    """
    Generates N5K-aligned metadata.json content and a FoodSAM to N5K ID mapping.
    It uses raw FoodSAM category observations (not tied to SAM masks at this stage).
    Method:
    1. Identify potential claims from each FoodSAM category observation to N5K ingredients in the dish.
    2. Resolve claims: N5K ingredients are claimed by the best-ranked FoodSAM observation.
    3. Group claimed N5K ingredients under each successful FoodSAM observation.
    4. Determine a principal N5K ingredient for each such group (by weight).
    5. If multiple observations of the same FoodSAM category ID lead to different groups,
       select one "winning" group (principal N5K) for that FoodSAM category based on max total weight.
    6. Construct final metadata with these winning principal N5K ingredients and summed nutrition.
    7. Create a map from FoodSAM category ID to the final N5K integer ID.
    8. Calculate confidence scores.

    Args:
        dish_id (str): Dish identifier.
        original_n5k_metadata (dict): Parsed ground truth N5K metadata for the dish.
        foodsam_category_observations (list): List of dicts, e.g.,
            [{'foodsam_category_id': '16', 'foodsam_category_name': 'cheese butter', 'observation_index': 0}, ...].
            'observation_index' is a unique identifier for each detected FoodSAM category instance.
        foodsam_id_to_n5k_map (dict): Maps FoodSAM-103 string ID to ranked N5K string IDs.
        n5k_string_id_to_int_id_map (dict): Maps N5K string ID (e.g., "ingr_0000000025") to N5K integer ID (e.g., 25).
        n5k_id_to_name_map (dict): Maps N5K string ID to N5K name.

    Returns:
        tuple: (final_metadata_dict, foodsam_cat_id_to_n5k_int_id_map)
               Returns (empty_metadata_dict, {}) on error or if no alignment.
    """
    empty_metadata_default = {
        "dish_id": dish_id, "ingredients": [], "total_nutritional_values": {},
        "confidence_scores": {"final_confidence": 0, "nutritional_score": 0, "ingredient_score": 0, "msre_nutritional": 0, "jaccard_ingredient": 0}
    }

    if not original_n5k_metadata or not original_n5k_metadata.get('ingredients'):
        logging.warning(f"[{dish_id}] Original N5K metadata is empty or has no ingredients. Cannot align.")
        return empty_metadata_default, {}

    original_n5k_ingredients_in_dish_by_id = {
        str(ing['ingredient_id']): ing for ing in original_n5k_metadata['ingredients']
    }
    if not original_n5k_ingredients_in_dish_by_id:
        logging.warning(f"[{dish_id}] No N5K ingredients parsed from original_n5k_metadata. Cannot align.")
        return empty_metadata_default, {}
    logging.debug(f"[{dish_id}] Original N5K ingredients in dish (by ID): {list(original_n5k_ingredients_in_dish_by_id.keys())}")

    # --- Phase 1: Build All Potential Claims with Ranks from FoodSAM Observations ---
    potential_claims = [] # List of tuples: (rank, observation_idx, foodsam_cat_id, foodsam_cat_name, original_n5k_ingredient_object_from_dish)
    
    if not foodsam_category_observations:
        logging.warning(f"[{dish_id}] No FoodSAM category observations provided. No claims will be generated.")
        # Proceed to return default empty metadata, as no alignment is possible.
        # Scores will be calculated based on empty final_ingredients_for_json_list later.

    for observation in foodsam_category_observations:
        obs_idx = observation['observation_index']
        foodsam_cat_id_str = observation['foodsam_category_id'] # Already a string from parser
        foodsam_cat_name = observation['foodsam_category_name']

        if foodsam_cat_id_str == "0": # Assuming "0" is FoodSAM background/unclassified
            logging.debug(f"[{dish_id}] Skipping FoodSAM background observation (obs_idx: {obs_idx}, ID: {foodsam_cat_id_str}) for claim building.")
            continue

        mapping_entry_for_foodsam_id = foodsam_id_to_n5k_map.get(foodsam_cat_id_str)

        if not mapping_entry_for_foodsam_id:
            logging.debug(f"[{dish_id}] No entry in foodsam_id_to_n5k_map for FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}'). No claims generated for obs_idx {obs_idx}.")
            continue
        
        mapped_n5k_categories = mapping_entry_for_foodsam_id.get('mapped_n5k_categories', [])
        if not mapped_n5k_categories:
            logging.debug(f"[{dish_id}] FoodSAM ID {foodsam_cat_id_str} ('{foodsam_cat_name}') has no 'mapped_n5k_categories' in map. No claims generated for obs_idx {obs_idx}.")
            continue

        for rank, n5k_candidate_from_mapping in enumerate(mapped_n5k_categories):
            n5k_id_value_from_map = n5k_candidate_from_mapping.get('n5k_id') # This is an N5K integer ID (e.g. 25)
            if n5k_id_value_from_map is None:
                logging.warning(f"[{dish_id}] Mapping candidate for FoodSAM ID {foodsam_cat_id_str} missing 'n5k_id': {n5k_candidate_from_mapping}. Skipping.")
                continue
            
            try:
                # The n5k_id from mapping is an integer. Format it to match keys in original_n5k_ingredients_in_dish_by_id
                n5k_string_id_formatted_for_dish_lookup = f"ingr_{int(n5k_id_value_from_map):010d}"
            except ValueError:
                logging.warning(f"[{dish_id}] Could not convert n5k_id '{n5k_id_value_from_map}' from mapping to int for formatting. Candidate: {n5k_candidate_from_mapping}. Skipping.")
                continue

            if n5k_string_id_formatted_for_dish_lookup in original_n5k_ingredients_in_dish_by_id:
                original_n5k_ingredient_object = original_n5k_ingredients_in_dish_by_id[n5k_string_id_formatted_for_dish_lookup]
                # Store foodsam_cat_id_str directly
                potential_claims.append((rank, obs_idx, foodsam_cat_id_str, foodsam_cat_name, original_n5k_ingredient_object))
                logging.debug(f"[{dish_id}] Potential claim: Rank {rank}, ObsIdx {obs_idx} (FS ID: {foodsam_cat_id_str}, Name: {foodsam_cat_name}), N5K: {original_n5k_ingredient_object['ingredient_name']} ({original_n5k_ingredient_object['ingredient_id']})")

    # --- Phase 2: Resolve Claims (N5K-Centric Adjudication) ---
    potential_claims.sort(key=lambda x: (x[4]['ingredient_id'], x[0])) # Sort by N5K ID, then rank

    # food_observation_claims: obs_idx -> {"foodsam_category_id": id, "foodsam_category_name": name, "claimed_items": [n5k_obj1, ...]}
    foodsam_observation_claims = {} 
    for obs in foodsam_category_observations: # Initialize for all observations
        foodsam_observation_claims[obs['observation_index']] = {
            "foodsam_category_id": obs['foodsam_category_id'],
            "foodsam_category_name": obs['foodsam_category_name'],
            "claimed_items": []
        }

    claimed_original_n5k_ids = set() # Tracks N5K string IDs that have been claimed

    logging.debug(f"[{dish_id}] Starting claim resolution. Total potential claims: {len(potential_claims)}")
    for rank, obs_idx, fs_cat_id, fs_cat_name, n5k_obj in potential_claims:
        original_n5k_ingr_id_str = n5k_obj['ingredient_id'] # This is the N5K string ID like "ingr_..."
        if original_n5k_ingr_id_str not in claimed_original_n5k_ids:
            foodsam_observation_claims[obs_idx]["claimed_items"].append(n5k_obj)
            claimed_original_n5k_ids.add(original_n5k_ingr_id_str)
            logging.debug(f"[{dish_id}] Claimed: N5K '{n5k_obj['ingredient_name']}' ({original_n5k_ingr_id_str}) by FoodSAM Observation {obs_idx} (FS ID: {fs_cat_id}, Name: {fs_cat_name}, Rank {rank})")
        else:
            logging.debug(f"[{dish_id}] Skipped claim: N5K '{n5k_obj['ingredient_name']}' ({original_n5k_ingr_id_str}) already claimed. Attempt by Obs {obs_idx} (FS ID: {fs_cat_id}, Rank {rank})")

    # --- Phase 3: Group Claims by FoodSAM Category ID and Select Principal N5K for Each Group ---
    # foodsam_category_to_candidate_groups: fs_cat_id -> [group1, group2, ...]
    # group: {"principal_n5k_string_id", "principal_n5k_name", "total_weight", "summed_nutrition", "contributing_n5k_objects"}
    foodsam_category_to_candidate_groups = {}

    for obs_idx, data in foodsam_observation_claims.items():
        fs_cat_id = data["foodsam_category_id"]
        claimed_items = data["claimed_items"]

        if not claimed_items:
            continue # This FoodSAM observation claimed nothing

        # Determine Principal N5K ingredient for this observation's group (by weight)
        principal_n5k_obj = sorted(claimed_items, key=lambda x: x.get('weight_g', 0), reverse=True)[0]
        principal_n5k_string_id = principal_n5k_obj['ingredient_id']
        principal_n5k_name = principal_n5k_obj['ingredient_name'] # Or use n5k_id_to_name_map[principal_n5k_string_id]

        total_weight_of_group = sum(item.get('weight_g', 0) for item in claimed_items)
        summed_nutrition_of_group = {
            'calories_kcal': sum(item.get('calories_kcal', 0) for item in claimed_items),
            'fat_g': sum(item.get('fat_g', 0) for item in claimed_items),
            'carbs_g': sum(item.get('carbs_g', 0) for item in claimed_items),
            'protein_g': sum(item.get('protein_g', 0) for item in claimed_items),
        }
        
        candidate_group_details = {
            "principal_n5k_string_id": principal_n5k_string_id,
            "principal_n5k_name": principal_n5k_name,
            "total_weight_of_group": total_weight_of_group,
            "summed_nutrition_of_group": summed_nutrition_of_group,
            "contributing_original_n5k_objects": claimed_items, # Full objects for later processing
            "source_foodsam_category_id": fs_cat_id # Keep track of the FS cat ID
        }

        if fs_cat_id not in foodsam_category_to_candidate_groups:
            foodsam_category_to_candidate_groups[fs_cat_id] = []
        foodsam_category_to_candidate_groups[fs_cat_id].append(candidate_group_details)

    # --- Phase 4: Finalize Metadata Ingredients and Create FoodSAM to N5K Integer ID Map ---
    final_ingredients_for_json_list = []
    foodsam_cat_id_to_n5k_int_id_map = {} # The mapping to be returned

    if not foodsam_category_to_candidate_groups:
        logging.info(f"[{dish_id}] No FoodSAM categories successfully claimed any N5K ingredients.")
    
    for fs_cat_id, list_of_candidate_groups in foodsam_category_to_candidate_groups.items():
        if not list_of_candidate_groups:
            continue

        # Select the "winning" group for this FoodSAM category ID based on highest total weight
        winning_candidate_group = sorted(list_of_candidate_groups, key=lambda x: x["total_weight_of_group"], reverse=True)[0]

        final_principal_n5k_string_id = winning_candidate_group["principal_n5k_string_id"]
        final_principal_n5k_name = winning_candidate_group["principal_n5k_name"]
        final_total_weight = winning_candidate_group["total_weight_of_group"]
        final_summed_nutrition = winning_candidate_group["summed_nutrition_of_group"]
        
        # Get N5K integer ID for the mapping
        # The n5k_string_id_to_int_id_map keys are like "ingr_0000000025", values are int 25
        final_n5k_int_id = n5k_string_id_to_int_id_map.get(final_principal_n5k_string_id)
        if final_n5k_int_id is None:
            logging.warning(f"[{dish_id}] Could not find integer ID for N5K string ID {final_principal_n5k_string_id} (FS Cat: {fs_cat_id}). Skipping this for FoodSAM->N5K map.")
        else:
            # Rule: If multiple FoodSAM categories map to the same final N5K principal,
            # the map should reflect this. The current logic maps each FS_cat_ID to its chosen N5K principal.
            # If a FoodSAM ID was already mapped (e.g. from a higher-weighted group of another FS ID that resolved to the same N5K principal),
            # this will overwrite. This is acceptable under current design: each FS cat ID gets one N5K mapping.
            if fs_cat_id in foodsam_cat_id_to_n5k_int_id_map and foodsam_cat_id_to_n5k_int_id_map[fs_cat_id] != final_n5k_int_id:
                 logging.warning(f"[{dish_id}] FoodSAM Cat ID {fs_cat_id} was already mapped to {foodsam_cat_id_to_n5k_int_id_map[fs_cat_id]}, now re-mapping to {final_n5k_int_id} due to winning group selection. This should ideally not happen if an FS Cat ID only generates one winning group.")
            foodsam_cat_id_to_n5k_int_id_map[fs_cat_id] = final_n5k_int_id

        # Add to final ingredients list for metadata.json
        final_ingredients_for_json_list.append({
            'id': final_principal_n5k_string_id, # N5K String ID
            'name': final_principal_n5k_name,
            'weight_g': round(final_total_weight, 2),
            'calories_kcal': round(final_summed_nutrition.get('calories_kcal',0), 1),
            'fat_g': round(final_summed_nutrition.get('fat_g',0), 2),
            'carbs_g': round(final_summed_nutrition.get('carbs_g',0), 2),
            'protein_g': round(final_summed_nutrition.get('protein_g',0), 2),
            'source_original_n5k_ingredient_ids': sorted(list(set(ing['ingredient_id'] for ing in winning_candidate_group["contributing_original_n5k_objects"]))),
            'foodsam_category_id_source': fs_cat_id # The FoodSAM category ID that led to this entry
        })

    # --- Phase 5: Calculate Totals & Confidence Scores ---
    # This part of logic (scoring) remains largely the same.
    merged_totals = {
        'weight_g': round(sum(ing.get('weight_g', 0) for ing in final_ingredients_for_json_list), 2),
        'calories_kcal': round(sum(ing.get('calories_kcal', 0) for ing in final_ingredients_for_json_list), 1),
        'fat_g': round(sum(ing.get('fat_g', 0) for ing in final_ingredients_for_json_list), 2),
        'carbs_g': round(sum(ing.get('carbs_g', 0) for ing in final_ingredients_for_json_list), 2),
        'protein_g': round(sum(ing.get('protein_g', 0) for ing in final_ingredients_for_json_list), 2)
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
            "msre_nutritional": round(msre_nutritional, 4), # Added for more detail
            "jaccard_ingredient": round(jaccard_ingredient, 4) # Added for more detail
        }
    }

    logging.info(f"[{dish_id}] Generated aligned N5K metadata. Ingredients count: {len(final_ingredients_for_json_list)}. Final confidence: {final_confidence:.4f}")
    logging.debug(f"[{dish_id}] FoodSAM Category ID to Final N5K Integer ID map: {foodsam_cat_id_to_n5k_int_id_map}")

    return aligned_metadata_content, foodsam_cat_id_to_n5k_int_id_map


def generate_n5k_semseg_from_foodsam_pred(raw_foodsam103_semseg_path, 
                                          foodsam_cat_id_to_final_n5k_int_id_map,
                                          n5k_int_id_to_name_map=None): # For logging N5K names
    """
    Generates an N5K category-based semantic segmentation image by re-coloring
    FoodSAM's raw semantic prediction map (semantic_pred_food103.png).

    Args:
        raw_foodsam103_semseg_path (str): Path to semantic_pred_food103.png (H, W, FoodSAM-103 class IDs).
        foodsam_cat_id_to_final_n5k_int_id_map (dict): Maps FoodSAM-103 category string ID 
                                                      to the final N5K integer ID.
        n5k_int_id_to_name_map (dict, optional): Maps N5K integer ID to N5K name (for logging).

    Returns:
        np.array: N5K semantic segmentation image (H, W) with final N5K integer IDs,
                  or None on error.
    """
    try:
        foodsam_pred_img = cv2.imread(raw_foodsam103_semseg_path, cv2.IMREAD_GRAYSCALE)
        if foodsam_pred_img is None:
            logging.error(f"Could not load FoodSAM-103 semantic prediction image: {raw_foodsam103_semseg_path}")
            return None
        
        image_shape = foodsam_pred_img.shape
        
        # Determine max N5K ID to select appropriate dtype for the output semseg
        max_n5k_id_val = 0
        if foodsam_cat_id_to_final_n5k_int_id_map: # Check if map is not empty
            valid_ids = [id_val for id_val in foodsam_cat_id_to_final_n5k_int_id_map.values() if isinstance(id_val, int)]
            if valid_ids:
                 max_n5k_id_val = max(valid_ids) if valid_ids else 0
        
        # N5K IDs go up to 555. uint16 is safer. Background is 0.
        dtype_semseg = np.uint16 if max_n5k_id_val > 255 else np.uint8
        n5k_final_semseg_img = np.zeros(image_shape, dtype=dtype_semseg)

        # Iterate over unique FoodSAM-103 category IDs present in the raw prediction
        unique_foodsam_ids_in_pred = np.unique(foodsam_pred_img)
        
        recolored_pixels_count = 0
        for foodsam_id_int in unique_foodsam_ids_in_pred:
            if foodsam_id_int == 0: # Skip FoodSAM background if it has special meaning (usually 0)
                continue

            foodsam_id_str = str(foodsam_id_int) # Map uses string keys for FoodSAM IDs
            
            final_n5k_int_id = foodsam_cat_id_to_final_n5k_int_id_map.get(foodsam_id_str)

            if final_n5k_int_id is not None and isinstance(final_n5k_int_id, int):
                # Find all pixels belonging to this FoodSAM ID and set them to the final N5K ID
                pixel_mask = (foodsam_pred_img == foodsam_id_int)
                n5k_final_semseg_img[pixel_mask] = final_n5k_int_id
                num_pixels_for_this_id = np.sum(pixel_mask)
                recolored_pixels_count += num_pixels_for_this_id
                if n5k_int_id_to_name_map and num_pixels_for_this_id > 0 :
                    n5k_name = n5k_int_id_to_name_map.get(final_n5k_int_id, "Unknown N5K ID")
                    logging.debug(f"Recolored FoodSAM ID {foodsam_id_str} to N5K ID {final_n5k_int_id} ('{n5k_name}') for {num_pixels_for_this_id} pixels.")
            else:
                # If a FoodSAM ID in the image is not in the map, those pixels remain 0 (background)
                logging.debug(f"FoodSAM ID {foodsam_id_str} from pred image not found in final N5K map or invalid N5K ID. Pixels will remain background.")

        if recolored_pixels_count == 0 and np.any(foodsam_pred_img > 0): # Image had content but nothing mapped
             logging.warning(f"Generated N5K semantic segmentation for {os.path.basename(raw_foodsam103_semseg_path)} is all background, but original FoodSAM pred was not empty. Check mappings.")
        elif recolored_pixels_count > 0:
            logging.info(f"Generated N5K semantic segmentation from FoodSAM pred: {os.path.basename(raw_foodsam103_semseg_path)}. {recolored_pixels_count} pixels recolored.")
        else: # Original FoodSAM pred was likely all background
             logging.info(f"Generated N5K semantic segmentation for {os.path.basename(raw_foodsam103_semseg_path)} (likely all background, as was original FoodSAM pred).")
             
        return n5k_final_semseg_img

    except FileNotFoundError:
        logging.error(f"Raw FoodSAM semantic prediction file not found: {raw_foodsam103_semseg_path}")
    except Exception as e:
        logging.error(f"Error generating N5K semantic segmentation from FoodSAM pred: {e}", exc_info=True)
    return None


def generate_n5k_bboxes_from_sam_and_semseg(dish_id, raw_sam_masks_npy_path, 
                                            n5k_semseg_image, # The newly generated N5K semseg
                                            n5k_int_id_to_name_map=None): 
    """
    Generates bounding_box.json in 4M format.
    Uses SAM masks for geometry and the N5K semantic segmentation image (generated by
    generate_n5k_semseg_from_foodsam_pred) to determine the majority N5K ID for each SAM mask.

    Args:
        dish_id (str): The ID of the dish.
        raw_sam_masks_npy_path (str): Path to masks.npy (raw SAM masks from FoodSAM).
        n5k_semseg_image (np.array): The N5K semantic segmentation image (H, W, N5K integer IDs).
                                     Output of generate_n5k_semseg_from_foodsam_pred.
        n5k_int_id_to_name_map (dict, optional): Maps N5K integer ID to N5K name (for "class_name").

    Returns:
        dict: Content for bounding_box.json in 4M format: {"instances": [{"boxes": ..., "class_name": ..., "score": ...}]}
              Returns None on error or if no valid bboxes are generated.
    """
    instances_for_json = []
    try:
        sam_masks = np.load(raw_sam_masks_npy_path) # (N, H, W) boolean array
        if sam_masks.ndim != 3 or sam_masks.shape[0] == 0:
            logging.warning(f"[{dish_id}] SAM masks at {raw_sam_masks_npy_path} are not valid or empty for N5K bbox. Shape: {sam_masks.shape}")
            return {"instances": []}
        
        if n5k_semseg_image is None:
            logging.error(f"[{dish_id}] Provided N5K semantic segmentation image is None. Cannot generate bounding boxes.")
            return {"instances": []}

        if sam_masks.shape[1:3] != n5k_semseg_image.shape:
            logging.error(f"[{dish_id}] Shape mismatch: SAM masks {sam_masks.shape[1:3]} vs N5K SemSeg {n5k_semseg_image.shape}. Cannot generate bounding boxes.")
            return {"instances": []}

        num_masks = sam_masks.shape[0]
        for i in range(num_masks):
            sam_instance_mask = sam_masks[i] # (H, W) boolean array
            if not np.any(sam_instance_mask): # Skip empty SAM masks
                continue
            
            # Determine majority N5K ID under this SAM mask by looking at the n5k_semseg_image
            n5k_ids_in_mask_area = n5k_semseg_image[sam_instance_mask]
            
            if n5k_ids_in_mask_area.size == 0:
                logging.debug(f"[{dish_id}] SAM Mask {i} has no corresponding pixels in N5K semseg image or is empty. Skipping for bbox.")
                continue
            
            unique_n5k_ids, counts = np.unique(n5k_ids_in_mask_area, return_counts=True)
            majority_n5k_id_int = unique_n5k_ids[np.argmax(counts)]

            if majority_n5k_id_int == 0: # N5K ID 0 is typically background
                logging.debug(f"[{dish_id}] SAM Mask {i} has majority N5K ID 0 (background). Skipping for bbox.")
                continue

            n5k_class_name = "unknown_n5k_id"
            if n5k_int_id_to_name_map:
                n5k_class_name = n5k_int_id_to_name_map.get(majority_n5k_id_int, f"n5k_id_{majority_n5k_id_int}")
            else:
                n5k_class_name = f"n5k_id_{majority_n5k_id_int}" # Fallback if no name map

            # Calculate bounding box from the SAM boolean mask
            mask_u8 = sam_instance_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logging.debug(f"[{dish_id}] No contours found for SAM mask {i} (N5K ID: {majority_n5k_id_int}). Skipping bbox.")
                continue
            
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            # 4M format: [x_min, y_min, x_max, y_max]
            bbox_coords = [int(x), int(y), int(x + w), int(y + h)]

            instances_for_json.append({
                "boxes": bbox_coords,
                "class_name": n5k_class_name,
                "score": 1.0 # Placeholder score, as we don't have a direct confidence for this N5K assignment here
            })
            logging.debug(f"[{dish_id}] Generated bbox for SAM mask {i} -> N5K ID {majority_n5k_id_int} ('{n5k_class_name}'). Box: {bbox_coords}")
        
        logging.info(f"[{dish_id}] Generated {len(instances_for_json)} N5K bounding boxes for 4M format.")
        return {"instances": instances_for_json}

    except FileNotFoundError:
        logging.error(f"[{dish_id}] SAM masks file not found for N5K bbox: {raw_sam_masks_npy_path}")
        return {"instances": []}
    except Exception as e:
        logging.error(f"[{dish_id}] Error generating N5K bounding boxes: {e}", exc_info=True)
        return {"instances": []} # Return empty structure on other errors 