"""
Common utility functions for file operations, logging, etc.
"""
import os
import shutil
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def read_text_file_lines(file_path):
    """Reads lines from a text file, stripping whitespace."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.warning(f"Text file not found: {file_path}")
        return []
