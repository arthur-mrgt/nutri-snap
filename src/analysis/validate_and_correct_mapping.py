import json
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
MAPPING_FILE_PATH = os.path.normpath(os.path.join(script_dir, '..', '..', 'data', 'category_id_files', 'foodsam_to_n5k_mapping.json'))
FOODSAM_REF_PATH = os.path.normpath(os.path.join(script_dir, '..', '..', 'external_libs', 'FoodSAM', 'FoodSAM', 'FoodSAM_tools', 'category_id_files', 'foodseg103_category_id.txt'))
N5K_REF_PATH = os.path.normpath(os.path.join(script_dir, '..', '..', 'data', 'category_id_files', 'nutrition5k_category.txt'))
OUTPUT_MAPPING_FILE_PATH = os.path.normpath(os.path.join(script_dir, '..', '..', 'data', 'category_id_files', 'foodsam_to_n5k_mapping_corrected.json'))

def load_reference_data(file_path, description):
    """Loads reference data, returning name_to_id and id_to_name maps."""
    name_to_id_map = {}
    id_to_name_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    item_id, item_name = parts[0].strip(), parts[1].strip()
                    if not item_id.isdigit():
                        print(f"Warning: Non-digit ID '{item_id}' found in {description} file at line {line_num} ('{item_name}'). Skipping.")
                        continue
                    # Use lowercased name for name_to_id key for case-insensitive lookup
                    # but store original name in id_to_name_map for canonical representation
                    if item_name.lower() in name_to_id_map:
                        print(f"Warning: Duplicate name '{item_name.lower()}' in {description} file (IDs: {name_to_id_map[item_name.lower()]}, {item_id}). Using first occurrence.")
                    else:
                        name_to_id_map[item_name.lower()] = item_id
                    
                    if item_id in id_to_name_map:
                        print(f"Warning: Duplicate ID '{item_id}' in {description} file (Names: {id_to_name_map[item_id]}, {item_name}). Using first occurrence.")
                    else:
                        id_to_name_map[item_id] = item_name
                elif line.strip():
                    print(f"Warning: Malformed line in {description} file at line {line_num}: {line.strip()}")
        print(f"Successfully loaded {len(id_to_name_map)} items from {description} file: {file_path}")
        return name_to_id_map, id_to_name_map
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}. Cannot proceed.")
        return None, None
    except Exception as e:
        print(f"Error reading {description} file {file_path}: {e}")
        return None, None

def validate_and_correct_mappings():
    print("Starting validation and correction process (names from JSON are leading)...")
    modifications_made = False # Initialize modification tracker

    foodsam_name_to_id_ref, foodsam_id_to_name_ref = load_reference_data(FOODSAM_REF_PATH, "FoodSAM reference")
    n5k_name_to_id_ref, n5k_id_to_name_ref = load_reference_data(N5K_REF_PATH, "Nutrition5k reference")

    if not foodsam_name_to_id_ref or not n5k_name_to_id_ref:
        print("Halting due to missing critical reference data.")
        return

    current_mapping_data = {}
    try:
        with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
            current_mapping_data = json.load(f)
        print(f"Successfully loaded mapping file: {MAPPING_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: Mapping file {MAPPING_FILE_PATH} not found. Cannot proceed.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {MAPPING_FILE_PATH}: {e}. Cannot proceed.")
        return
    except Exception as e:
        print(f"Error reading mapping file {MAPPING_FILE_PATH}: {e}")
        return

    # This will store the corrected data, keyed by corrected FoodSAM ID.
    # It handles merging if multiple JSON entries map to the same canonical FoodSAM ID.
    processed_foodsam_entries = {}

    for original_json_key, entry_data_from_json in current_mapping_data.items():
        current_foodsam_name_from_json = entry_data_from_json.get("foodsam_name")

        if not current_foodsam_name_from_json:
            print(f"Warning: Entry with original JSON key '{original_json_key}' is missing 'foodsam_name'. Skipping.")
            continue

        corrected_foodsam_id = foodsam_name_to_id_ref.get(current_foodsam_name_from_json.lower())
        
        if not corrected_foodsam_id:
            print(f"Warning: FoodSAM name '{current_foodsam_name_from_json}' (from JSON key '{original_json_key}') not found in FoodSAM reference. Skipping this entry.")
            continue
        
        canonical_foodsam_name = foodsam_id_to_name_ref.get(corrected_foodsam_id, current_foodsam_name_from_json)

        original_foodsam_name_from_json = entry_data_from_json.get("foodsam_name") # Temporary variable
        if original_foodsam_name_from_json != canonical_foodsam_name:
            modifications_made = True
            print(f"Info: FoodSAM name for ID {corrected_foodsam_id} updated from '{original_foodsam_name_from_json}' to canonical '{canonical_foodsam_name}'.")

        if corrected_foodsam_id not in processed_foodsam_entries:
            processed_foodsam_entries[corrected_foodsam_id] = {
                "foodsam_id": corrected_foodsam_id, # Initial assignment
                "foodsam_name": canonical_foodsam_name, # Initial assignment
                "mapped_n5k_categories": [],
                "_seen_n5k_ids_in_list": set()
            }
        # Ensure canonical name is used if entry already exists (less likely with name-first approach but good for robustness)
        elif processed_foodsam_entries[corrected_foodsam_id]["foodsam_name"] != canonical_foodsam_name:
            modifications_made = True # Name was different in a merged entry
            processed_foodsam_entries[corrected_foodsam_id]["foodsam_name"] = canonical_foodsam_name
            print(f"Info: FoodSAM ID {corrected_foodsam_id} already processed. Ensured canonical name '{canonical_foodsam_name}'.")


        # Correct internal foodsam_id field and check for modifications
        original_internal_foodsam_id = entry_data_from_json.get("foodsam_id")
        if original_internal_foodsam_id is not None:
            if str(original_internal_foodsam_id) != corrected_foodsam_id:
                modifications_made = True
                print(f"Info: FoodSAM entry for '{canonical_foodsam_name}' (ID: {corrected_foodsam_id}): Internal 'foodsam_id' field '{original_internal_foodsam_id}' from JSON key '{original_json_key}' updated to '{corrected_foodsam_id}'.")
        elif corrected_foodsam_id is not None: # Was missing, now filled
            modifications_made = True
            print(f"Info: FoodSAM entry for '{canonical_foodsam_name}' (ID: {corrected_foodsam_id}): Internal 'foodsam_id' field was missing, now set to '{corrected_foodsam_id}'.")
        processed_foodsam_entries[corrected_foodsam_id]["foodsam_id"] = corrected_foodsam_id # Ensure it's the corrected one

        current_n5k_list_from_json = entry_data_from_json.get("mapped_n5k_categories", [])
        if not isinstance(current_n5k_list_from_json, list):
            print(f"Warning: 'mapped_n5k_categories' for FoodSAM '{canonical_foodsam_name}' (ID: {corrected_foodsam_id}) from JSON key '{original_json_key}' is not a list. Skipping N5K items for this particular source entry.")
            current_n5k_list_from_json = []

        for n5k_item_from_json in current_n5k_list_from_json:
            if not isinstance(n5k_item_from_json, dict):
                print(f"Warning: Malformed N5K item (not a dict): '{n5k_item_from_json}' under FoodSAM '{canonical_foodsam_name}'. Skipping.")
                continue
            
            current_n5k_name_from_json = n5k_item_from_json.get("n5k_name")
            original_n5k_id_from_json_val = n5k_item_from_json.get("n5k_id") # Keep original type for comparison if needed, but usually string
            original_n5k_id_from_json_str = str(original_n5k_id_from_json_val) if original_n5k_id_from_json_val is not None else ""

            if not current_n5k_name_from_json:
                print(f"Warning: N5K item is missing 'n5k_name' under FoodSAM '{canonical_foodsam_name}': {n5k_item_from_json}. Skipping.")
                continue

            corrected_n5k_id = n5k_name_to_id_ref.get(current_n5k_name_from_json.lower())
            if not corrected_n5k_id:
                print(f"Warning: N5K name '{current_n5k_name_from_json}' (under FoodSAM '{canonical_foodsam_name}') not found in N5K reference. Skipping this N5K item.")
                continue
            
            canonical_n5k_name = n5k_id_to_name_ref.get(corrected_n5k_id, current_n5k_name_from_json)

            # Check for n5k_id correction
            if original_n5k_id_from_json_val is not None:
                if original_n5k_id_from_json_str != corrected_n5k_id:
                    modifications_made = True
                    print(f"Info: N5K item '{canonical_n5k_name}' (under FoodSAM '{canonical_foodsam_name}'): ID corrected from '{original_n5k_id_from_json_str}' to '{corrected_n5k_id}'.")
            elif corrected_n5k_id is not None: # ID was missing and now filled
                modifications_made = True
                print(f"Info: N5K item '{canonical_n5k_name}' (under FoodSAM '{canonical_foodsam_name}'): ID was missing, now set to '{corrected_n5k_id}'.")

            # Check for n5k_name correction
            original_n5k_name_from_json = n5k_item_from_json.get("n5k_name") # Temporary variable
            if original_n5k_name_from_json != canonical_n5k_name:
                modifications_made = True
                print(f"Info: N5K item ID '{corrected_n5k_id}' (under FoodSAM '{canonical_foodsam_name}'): Name corrected from '{original_n5k_name_from_json}' to '{canonical_n5k_name}'.")
                
            target_foodsam_entry = processed_foodsam_entries[corrected_foodsam_id]
            if corrected_n5k_id not in target_foodsam_entry["_seen_n5k_ids_in_list"]:
                target_foodsam_entry["mapped_n5k_categories"].append({
                    "n5k_id": corrected_n5k_id,
                    "n5k_name": canonical_n5k_name
                })
                target_foodsam_entry["_seen_n5k_ids_in_list"].add(corrected_n5k_id)
            else:
                modifications_made = True # A duplicate N5K item was encountered and will be skipped
                print(f"Info: Duplicate N5K ID '{corrected_n5k_id}' (name: '{canonical_n5k_name}') within/for FoodSAM '{canonical_foodsam_name}'. Not added again.")

    # Final clean-up and ensure all reference FoodSAM IDs are present
    final_json_output = {}
    if foodsam_id_to_name_ref: # Proceed only if FoodSAM reference was loaded
        for ref_foodsam_id, ref_foodsam_name in foodsam_id_to_name_ref.items():
            if ref_foodsam_id in processed_foodsam_entries:
                entry_to_add = processed_foodsam_entries[ref_foodsam_id]
                del entry_to_add["_seen_n5k_ids_in_list"] # Remove helper set
                final_json_output[ref_foodsam_id] = entry_to_add
            else:
                modifications_made = True # Adding a missing FoodSAM entry from reference
                print(f"Info: FoodSAM ID '{ref_foodsam_id}' (name: '{ref_foodsam_name}') from reference was not found/processed from input JSON. Adding as empty.")
                final_json_output[ref_foodsam_id] = {
                    "foodsam_id": ref_foodsam_id,
                    "foodsam_name": ref_foodsam_name,
                    "mapped_n5k_categories": []
                }
    else: # Fallback if FoodSAM reference failed to load, just use what was processed
        print("Warning: FoodSAM reference data was not available. Output will only contain entries successfully processed from input JSON.")
        for fs_id, entry_data in processed_foodsam_entries.items():
            del entry_data["_seen_n5k_ids_in_list"]
            final_json_output[fs_id] = entry_data
    
    # Save the corrected data ONLY if modifications were made
    if modifications_made:
        try:
            os.makedirs(os.path.dirname(OUTPUT_MAPPING_FILE_PATH), exist_ok=True)
            with open(OUTPUT_MAPPING_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_json_output, f, indent=2)
            print(f"Successfully wrote validated and corrected data to {OUTPUT_MAPPING_FILE_PATH}")
            print(f"""You can now compare this file with the original and replace if satisfied:
                  Original: {MAPPING_FILE_PATH}
                  Corrected: {OUTPUT_MAPPING_FILE_PATH}""")

            # --- Report on unmapped N5K ingredients ---
            if n5k_id_to_name_ref: # Ensure N5K reference was loaded
                all_mapped_n5k_ids_in_final_output = set()
                for foodsam_entry in final_json_output.values():
                    if "mapped_n5k_categories" in foodsam_entry and isinstance(foodsam_entry["mapped_n5k_categories"], list):
                        for n5k_item in foodsam_entry["mapped_n5k_categories"]:
                            if isinstance(n5k_item, dict) and "n5k_id" in n5k_item:
                                all_mapped_n5k_ids_in_final_output.add(n5k_item["n5k_id"])
                
                all_n5k_ref_ids = set(n5k_id_to_name_ref.keys())
                unmapped_n5k_ids = all_n5k_ref_ids - all_mapped_n5k_ids_in_final_output
                number_of_unmapped_n5k_ingredients = len(unmapped_n5k_ids)

                print(f"\n--- Nutrition5k Ingredients Usage Report ---")
                print(f"Total Nutrition5k ingredients in reference file: {len(all_n5k_ref_ids)}")
                print(f"Number of unique Nutrition5k ingredients mapped in the corrected output: {len(all_mapped_n5k_ids_in_final_output)}")
                print(f"Number of Nutrition5k ingredients not appearing in any mapping: {number_of_unmapped_n5k_ingredients}")

                if unmapped_n5k_ids:
                    print("\nList of unmapped Nutrition5k ingredients (ID: Name):")
                    sorted_unmapped_list = sorted(list(unmapped_n5k_ids), key=lambda x: int(x) if x.isdigit() else x)
                    for n5k_id in sorted_unmapped_list:
                        print(f"  {n5k_id}: {n5k_id_to_name_ref.get(n5k_id, 'Unknown Name')}")
        except Exception as e:
            print(f"Error writing corrected mapping file {OUTPUT_MAPPING_FILE_PATH}: {e}")
    else:
        print(f"No modifications were necessary for {MAPPING_FILE_PATH}. Output file not created.")

if __name__ == "__main__":
    validate_and_correct_mappings() 