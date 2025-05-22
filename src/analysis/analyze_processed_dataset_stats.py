import pandas as pd
import numpy as np
import os
import argparse

def analyze_processed_ids(file_path):
    """
    Analyzes the processed_ids.txt file to extract and print various statistics.

    Args:
        file_path (str): The path to the processed_ids.txt file.
    """
    # Verify if the file exists
    if not os.path.exists(file_path):
        print(f"ERREUR: Le fichier {file_path} n'a pas été trouvé. Veuillez vérifier le chemin.")
        return

    print(f"Fichier {file_path} trouvé. Début de l'analyse...")

    column_names = [
        'dish_id', 'status',
        'final_confidence', 'nutritional_score', 'ingredient_score',
        'msre_nutritional', 'jaccard_ingredient', 'nb_ingredient'
    ]

    try:
        # Read the file, ignoring lines starting with '#' (comments/header)
        df = pd.read_csv(file_path, comment='#', delim_whitespace=True, names=column_names, header=None)
        print("Fichier chargé avec succès.")

        # Filter for successfully processed dishes
        successful_df = df[df['status'] == 'successful'].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        if successful_df.empty:
            print("Aucun plat traité avec succès trouvé dans le fichier.")
            return
            
        print(f"{len(successful_df)} plats traités avec succès trouvés.")

        # Convert metric columns to numeric. Errors will become NaN.
        metric_columns = ['final_confidence', 'nutritional_score', 'ingredient_score', 'msre_nutritional', 'jaccard_ingredient', 'nb_ingredient']
        for col in metric_columns:
            successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')

        # Remove rows where conversion failed for metrics (NaN)
        original_count = len(successful_df)
        successful_df.dropna(subset=metric_columns, inplace=True)
        cleaned_count = len(successful_df)
        if original_count > cleaned_count:
            print(f"{original_count - cleaned_count} lignes ont été supprimées en raison de valeurs non numériques dans les colonnes de métriques.")
        
        if successful_df.empty:
            print("Aucun plat avec des métriques valides après nettoyage.")
            return

        print(f"{len(successful_df)} plats avec des métriques valides après nettoyage.")

        # --- Moyennes et Médianes des Métriques ---
        print("\nStatistiques pour les plats traités avec succès:\n")

        mean_stats = successful_df[metric_columns].mean()
        median_stats = successful_df[metric_columns].median()

        print("Moyennes:")
        for metric, value in mean_stats.items():
            print(f"- {metric}: {value:.4f}")

        print("\nMédianes:")
        for metric, value in median_stats.items():
            print(f"- {metric}: {value:.4f}")

        # --- Statistiques Filtrées ---
        print("\nStatistiques pour les plats filtrés (final_confidence >= 0.65 ET nb_ingredient > 1):\n")
        
        filtered_stats_df = successful_df[
            (successful_df['final_confidence'] >= 0.65) &
            (successful_df['nb_ingredient'] > 1)
        ]

        count_filtered_dishes = len(filtered_stats_df)
        avg_ingredients_filtered = filtered_stats_df['nb_ingredient'].mean() if count_filtered_dishes > 0 else 0

        print(f"- Nombre de plats correspondants: {count_filtered_dishes}")
        print(f"- Nombre moyen d'ingrédients pour ces plats: {avg_ingredients_filtered:.4f}")

    except FileNotFoundError:
        print(f"ERREUR: Le fichier {file_path} n'a pas été trouvé lors de la tentative de lecture.")
    except pd.errors.EmptyDataError:
        print(f"ERREUR: Le fichier {file_path} est vide ou ne contient aucune colonne à analyser après avoir ignoré les commentaires.")
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture ou du traitement du fichier: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyser le fichier processed_ids.txt et afficher des statistiques.")
    
    # Chemin par défaut. 
    # Si le script est dans src/analysis/ et le fichier processed_ids.txt dans data/processed/
    # (donc, ../../data/processed/processed_ids.txt depuis src/analysis/)
    default_file_path = "../../data/processed/processed_ids.txt" 
    
    # Pour utiliser le chemin spécifique que vous avez mentionné, décommentez et modifiez la ligne suivante:
    # default_file_path = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"

    parser.add_argument(
        "--file_path", 
        type=str, 
        default=default_file_path,
        help="Chemin vers le fichier processed_ids.txt"
    )
    args = parser.parse_args()
    
    # Si vous souhaitez utiliser un chemin absolu fixe sans argument en ligne de commande, 
    # vous pouvez commenter les lignes de l'argparse et décommenter et modifier la ligne ci-dessous :
    # fixed_path = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"
    # print(f"Utilisation du chemin fixe : {fixed_path}")
    # analyze_processed_ids(fixed_path)
    
    # Ligne pour exécuter avec l'argument ou le défaut:
    # print(f"Analyse du fichier : {args.file_path}") 
    # analyze_processed_ids(args.file_path)

    # Ligne pour exécuter avec le chemin absolu que vous avez spécifié
    # (commentez les lignes de l'argparse ci-dessus si vous utilisez celle-ci):
    user_specific_path = "/work/com-304/nutri-snap/data/processed/processed_ids.txt"
    print(f"Analyse du fichier : {user_specific_path}")
    analyze_processed_ids(user_specific_path) 