"""
Common utility functions for file operations, logging, etc.
"""
import os
import shutil
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def create_dir_if_not_exists(dir_path):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Created directory: {dir_path}")

def copy_file(src_path, dest_path):
    """Copies a file from src_path to dest_path."""
    try:
        shutil.copy2(src_path, dest_path)
        logging.debug(f"Copied {src_path} to {dest_path}")
    except FileNotFoundError:
        logging.error(f"Source file not found for copying: {src_path}")
    except Exception as e:
        logging.error(f"Error copying file {src_path} to {dest_path}: {e}")

def save_json(data, json_path):
    """Saves a dictionary to a JSON file."""
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Saved JSON to {json_path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {json_path}: {e}")

def load_json(json_path):
    """Loads data from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON from {json_path}: {e}")
        return None

def load_category_info(file_path, delimiter='\t'):
    """Loads category information from a text file (e.g., ID<delimiter>Name).

    Args:
        file_path (str): Path to the category file.
        delimiter (str): Delimiter used in the file.

    Returns:
        tuple: (dict: id_to_name, dict: name_to_id)
               Returns two empty dicts if file not found or parsing error.
    """
    id_to_name = {}
    name_to_id = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter, 1)
                if len(parts) == 2:
                    cat_id_str, cat_name = parts[0].strip(), parts[1].strip()
                    try:
                        # Attempt to convert ID to int, if not, keep as string
                        try:
                            cat_id = int(cat_id_str)
                        except ValueError:
                            cat_id = cat_id_str # Keep as string if not an int (e.g. 'plate only')
                        
                        if cat_id in id_to_name and id_to_name[cat_id] != cat_name:
                            logging.warning(f"Duplicate ID {cat_id} in {file_path}: '{id_to_name[cat_id]}' vs '{cat_name}'. Using first occurrence.")
                        elif cat_id not in id_to_name:
                             id_to_name[cat_id] = cat_name
                        
                        if cat_name in name_to_id and name_to_id[cat_name] != cat_id:
                            logging.warning(f"Duplicate name '{cat_name}' in {file_path} for IDs: {name_to_id[cat_name]} vs {cat_id}. Using first occurrence.")
                        elif cat_name not in name_to_id:
                            name_to_id[cat_name] = cat_id
                            
                    except ValueError:
                        logging.warning(f"Could not parse category ID from '{cat_id_str}' in {file_path}. Skipping line: {line}")
                else:
                    logging.warning(f"Could not parse line in {file_path}: {line}. Expected format: ID<delimiter>Name.")
        logging.info(f"Loaded {len(id_to_name)} categories from {file_path}")
    except FileNotFoundError:
        logging.error(f"Category file not found: {file_path}")
    except Exception as e:
        logging.error(f"Error loading category info from {file_path}: {e}")
    return id_to_name, name_to_id

def read_text_file_lines(file_path):
    """Reads lines from a text file, stripping whitespace."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.warning(f"Text file not found: {file_path}")
        return []
