"""
Configuration file for the dataset processing pipeline.
"""
import os

# --- Project Root ---
# Assumes this script (config.py) is in src/data_processing/utils/,
# so the project_root is three levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# --- Category and Mapping Files ---
# Directory containing various category ID and mapping files.
CATEGORY_ID_FILES_DIR = os.path.join(PROJECT_ROOT, "data", "category_id_files")
N5K_CATEGORY_TXT_FILENAME = "nutrition5k_category.txt" # N5K ingredient categories.
FOODSAM103_CATEGORY_TXT_FILENAME = "foodseg103_category_id.txt" # FoodSAM-103 categories.
FOODSAM_TO_N5K_MAPPING_FILENAME = "foodsam_to_n5k_mapping.json" # Mapping from FoodSAM to N5K categories.

N5K_CATEGORY_TXT = os.path.join(CATEGORY_ID_FILES_DIR, N5K_CATEGORY_TXT_FILENAME)
FOODSAM103_CATEGORY_TXT = os.path.join(CATEGORY_ID_FILES_DIR, FOODSAM103_CATEGORY_TXT_FILENAME)
FOODSAM_TO_N5K_MAPPING_JSON = os.path.join(CATEGORY_ID_FILES_DIR, FOODSAM_TO_N5K_MAPPING_FILENAME)

# --- Input Nutrition5k Dataset Paths ---
# Root directory of the raw Nutrition5k dataset.
N5K_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "nutrition5k_dataset")
N5K_IMAGERY_DIR = os.path.join(N5K_ROOT, "imagery", "realsense_overhead") # N5K dish imagery.
N5K_METADATA_DIR = os.path.join(N5K_ROOT, "metadata") # N5K original metadata files.
N5K_SPLITS_DIR = os.path.join(N5K_ROOT, "dish_ids", "splits") # N5K train/test split files.

# --- FoodSAM Paths & Configuration ---
# Path to the root of the cloned FoodSAM repository.
FOODSAM_DIR = os.path.join(PROJECT_ROOT, "external_libs", "FoodSAM")

SAM_CHECKPOINT = os.path.join(FOODSAM_DIR, "ckpts", "sam_vit_h_4b8939.pth") # Path to SAM checkpoint.
# SEMANTIC_CHECKPOINT is relative to FOODSAM_DIR as used in its original scripts.
SEMANTIC_CHECKPOINT_FILENAME = os.path.join("ckpts", "SETR_MLA", "iter_80000.pth") 
# SEMANTIC_CONFIG is relative to FOODSAM_DIR as used in its original scripts.
SEMANTIC_CONFIG_FILENAME = os.path.join("configs", "SETR_MLA_768x768_80k_base.py") 

# FOODSAM_CATEGORY_TXT is relative to FOODSAM_DIR, pointing to FoodSAM's internal category file.
FOODSAM_CATEGORY_TXT_FILENAME = os.path.join("FoodSAM", "FoodSAM_tools", "category_id_files", "foodseg103_category_id.txt")
# FOODSAM_COLOR_LIST_PATH is relative to FOODSAM_DIR, pointing to FoodSAM's internal color list.
FOODSAM_COLOR_LIST_PATH_FILENAME = os.path.join("FoodSAM", "FoodSAM_tools", "color_list.npy")

MODEL_TYPE_SAM = 'vit_h' # SAM model type (e.g., 'vit_h', 'vit_l', 'vit_b').

# --- Output Dataset Paths ---
# Root directory for the final processed dataset.
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# Directory for intermediate outputs from FoodSAM (raw masks, raw semantic predictions, etc.).
INTERMEDIATE_FOODSAM_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, "intermediate_foodsam_outputs")
# Subdirectory name for storing original N5K metadata in the intermediate output structure.
INTERMEDIATE_N5K_GROUND_TRUTH_METADATA_DIR_NAME = "n5k_ground_truth_metadata"

# --- Other Configurations ---
# Minimum area (in pixels) for a contour to be considered a SAM instance.
MIN_CONTOUR_AREA_SAM_INSTANCE = 50
# Default score for bounding boxes if not provided by a model.
DEFAULT_BBOX_SCORE = 1.0
# Device for FoodSAM processing ("cuda" or "cpu").
FOODSAM_DEVICE = "cuda"

# Parameters for enhance_masks function from FoodSAM_tools.
ENHANCE_MASKS_AREA_THR = 0      # Area threshold for mask enhancement.
ENHANCE_MASKS_RATIO_THR = 0.5 # Ratio threshold for mask enhancement.
ENHANCE_MASKS_TOP_K = 80        # Top K masks to keep during enhancement.
