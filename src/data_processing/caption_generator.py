import os
import json
from pathlib import Path

BASE_PROCESSED_PATH = Path("/work/com-304/nutri-snap/data/processed/")
SUBSETS = ["train", "test"]

def transform_ingredient_id(raw_id):
    """
    Transforms an ingredient ID from 'ingr_0000000192' to '192'.
    """
    if not raw_id.startswith("ingr_"):
        return raw_id # Or raise an error, depending on expected format
    numeric_part = raw_id.replace("ingr_", "")
    try:
        return str(int(numeric_part))
    except ValueError:
        # Handle cases where the part after "ingr_" is not purely numeric if necessary
        return numeric_part # Fallback or raise error

def generate_caption_for_dish(metadata_file_path):
    """
    Generates a caption string from a dish metadata JSON file.
    Example: "192:62.0 205:15.3"
    """
    try:
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file not found: {metadata_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from: {metadata_file_path}")
        return None

    ingredients = metadata.get("ingredients", [])
    caption_parts = []

    for ingredient in ingredients:
        raw_id = ingredient.get("id")
        weight = ingredient.get("weight_g")

        if raw_id is None or weight is None:
            print(f"Warning: Missing 'id' or 'weight_g' for an ingredient in {metadata_file_path}")
            continue

        transformed_id = transform_ingredient_id(raw_id)
        caption_parts.append(f"{transformed_id}:{weight}")

    return " ".join(caption_parts)

def main():
    """
    Main function to process all dishes in train and test sets.
    """
    for subset in SUBSETS:
        print(f"Processing subset: {subset}...")
        metadata_base_path = BASE_PROCESSED_PATH / subset / "metadata"
        caption_base_path = BASE_PROCESSED_PATH / subset / "caption"

        if not metadata_base_path.is_dir():
            print(f"Warning: Metadata directory not found for {subset}: {metadata_base_path}")
            continue

        for dish_folder in metadata_base_path.iterdir():
            if dish_folder.is_dir():
                dish_id = dish_folder.name # e.g., "dish_1558031526"
                metadata_file_name = f"{dish_id}.json"
                metadata_file_path = dish_folder / metadata_file_name

                if not metadata_file_path.is_file():
                    print(f"Warning: Metadata file not found: {metadata_file_path}")
                    continue

                print(f"  Processing dish: {dish_id}")
                caption_content = generate_caption_for_dish(metadata_file_path)

                if caption_content is not None:
                    output_caption_folder = caption_base_path / dish_id
                    output_caption_folder.mkdir(parents=True, exist_ok=True)

                    output_caption_file_path = output_caption_folder / metadata_file_name # Save as dish_id.json

                    try:
                        with open(output_caption_file_path, 'w') as f:
                            # The user specified the caption string directly as the content of the .json file
                            f.write(caption_content)
                        print(f"    Successfully generated caption: {output_caption_file_path}")
                    except IOError:
                        print(f"    Error: Could not write caption file: {output_caption_file_path}")
                else:
                    print(f"    Failed to generate caption for {dish_id}")
        print(f"Finished processing subset: {subset}.\n")

if __name__ == "__main__":
    main()
