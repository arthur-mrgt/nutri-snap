import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_depth_images_conditional(raw_data_root_str, processed_data_root_str, splits_to_process):
    """
    Copie les images de profondeur 'depth_raw.png' du dataset raw vers le dataset processed,
    conditionnellement à l'existence d'un dossier RGB correspondant dans le même split.

    Args:
        raw_data_root_str (str): Chemin racine du dataset raw 
                                 (ex: '/work/com-304/nutri-snap/data/raw/nutrition5k_dataset/imagery/realsense_overhead').
        processed_data_root_str (str): Chemin racine du dataset processed 
                                     (ex: '/work/com-304/nutri-snap/data/processed').
        splits_to_process (list): Liste des noms de splits à traiter (ex: ['train', 'test']).
    """
    raw_data_root = Path(raw_data_root_str)
    processed_data_root = Path(processed_data_root_str)

    if not raw_data_root.exists():
        print(f"ERREUR: Le dossier source raw '{raw_data_root}' n'existe pas.")
        return

    overall_copied_count = 0
    overall_skipped_count = 0
    overall_missing_rgb_count = 0

    for split_name in splits_to_process:
        print(f"\n--- Traitement du split : {split_name} ---")
        
        processed_rgb_split_root = processed_data_root / split_name / 'rgb'
        processed_depth_split_root = processed_data_root / split_name / 'depth'

        if not processed_rgb_split_root.exists():
            print(f"AVERTISSEMENT: Le dossier RGB pour le split '{split_name}' ({processed_rgb_split_root}) n'existe pas. Aucun plat ne sera traité pour ce split.")
            continue

        # Lister les dossiers de plats existants dans le dossier RGB du split processed
        existing_rgb_dish_folders = [d for d in processed_rgb_split_root.iterdir() if d.is_dir() and d.name.startswith('dish_')]
        
        if not existing_rgb_dish_folders:
            print(f"Aucun dossier de plat (commençant par 'dish_') trouvé dans {processed_rgb_split_root} pour le split '{split_name}'.")
            continue
            
        print(f"Trouvé {len(existing_rgb_dish_folders)} plats avec une modalité RGB dans le split '{split_name}'.")
        processed_depth_split_root.mkdir(parents=True, exist_ok=True)
        print(f"Les images de profondeur pour le split '{split_name}' seront copiées dans : {processed_depth_split_root}")

        split_copied_count = 0
        split_skipped_count = 0 # Pour les depth_raw.png non trouvés
        split_missing_rgb_dish_in_raw = 0 # Pour les cas où le dish_XXXX du RGB processed n'est pas dans le raw

        for rgb_dish_folder in tqdm(existing_rgb_dish_folders, desc=f"Processing dishes for {split_name}"):
            dish_name = rgb_dish_folder.name  # ex: dish_1556572657

            # Chemin source de l'image de profondeur dans le dataset raw
            source_dish_folder_in_raw = raw_data_root / dish_name
            source_depth_file = source_dish_folder_in_raw / 'depth_raw.png'

            if not source_dish_folder_in_raw.exists():
                # print(f"Le dossier du plat '{dish_name}' n'existe pas dans la source raw '{raw_data_root}'. Saut.")
                split_missing_rgb_dish_in_raw +=1
                continue

            if source_depth_file.exists():
                # Créer le dossier de destination spécifique au plat dans processed/split/depth/
                target_dish_folder_in_processed_depth = processed_depth_split_root / dish_name
                target_dish_folder_in_processed_depth.mkdir(parents=True, exist_ok=True)

                # Nom du fichier de destination
                target_depth_file_name = f"{dish_name}.png"
                target_depth_file_path = target_dish_folder_in_processed_depth / target_depth_file_name

                # Copier le fichier
                try:
                    shutil.copy2(source_depth_file, target_depth_file_path)
                    split_copied_count += 1
                except Exception as e:
                    print(f"ERREUR lors de la copie de {source_depth_file} vers {target_depth_file_path}: {e}")
                    split_skipped_count += 1
            else:
                # print(f"Fichier depth_raw.png non trouvé pour {dish_name} dans '{source_dish_folder_in_raw}'.")
                split_skipped_count += 1
        
        print(f"Résumé pour le split '{split_name}':")
        print(f"  Images de profondeur copiées : {split_copied_count}")
        print(f"  'depth_raw.png' non trouvés (ou erreur de copie) : {split_skipped_count}")
        if split_missing_rgb_dish_in_raw > 0:
            print(f"  Plats présents dans '{processed_rgb_split_root}' mais pas dans '{raw_data_root}': {split_missing_rgb_dish_in_raw}")

        overall_copied_count += split_copied_count
        overall_skipped_count += split_skipped_count
        overall_missing_rgb_count += split_missing_rgb_dish_in_raw
            
    print(f"\n--- Opération Globale Terminée ---")
    print(f"Nombre total d'images de profondeur copiées : {overall_copied_count}")
    print(f"Nombre total de 'depth_raw.png' non trouvés (ou erreur de copie) : {overall_skipped_count}")
    if overall_missing_rgb_count > 0 :
         print(f"Nombre total de plats 'processed RGB' non trouvés dans la source 'raw': {overall_missing_rgb_count}")


if __name__ == '__main__':
    # Configurez vos chemins ici
    RAW_DATASET_IMAGERY_ROOT = '/work/com-304/nutri-snap/data/raw/nutrition5k_dataset/imagery/realsense_overhead'
    PROCESSED_DATASET_ROOT = '/work/com-304/nutri-snap/data/processed'
    
    # Spécifiez les splits que vous voulez traiter
    SPLITS_TO_CONSIDER = ['train', 'test'] 
    # Vous pouvez ajouter 'val' si vous l'utilisez : ['train', 'val', 'test']
    
    print("Début du script de copie conditionnelle des images de profondeur...")
    copy_depth_images_conditional(RAW_DATASET_IMAGERY_ROOT, PROCESSED_DATASET_ROOT, SPLITS_TO_CONSIDER)
    print("Script de copie terminé.")
