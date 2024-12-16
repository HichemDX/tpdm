import pandas as pd
import numpy as np
from collections import Counter

# Chargement du dataset
data = pd.read_csv('./NewDatasetExos.csv', sep=';')

# Vérification des colonnes disponibles
print("Colonnes disponibles :", data.columns)

# Remplacer les NaN dans la colonne 'Category' par une valeur par défaut, par exemple 'Unknown'
data['Category'] = data['Category'].fillna('Unknown')

# Convertir les colonnes numériques en valeurs numériques pour les calculs de distance
columns_to_convert = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyro_x', 'Gyro_y', 'Gyro_z']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 1. Fonction pour calculer la distance entre deux instances (Manhattan + Hamming)
def calculate_distance(instance1, instance2, numerical_cols, categorical_cols):
    # Calcul de la distance de Manhattan pour les colonnes numériques
    manhattan_distance = sum(abs(instance1[col] - instance2[col]) for col in numerical_cols)

    # Calcul de la distance de Hamming pour les colonnes catégoriques
    hamming_distance = sum(instance1[col] != instance2[col] for col in categorical_cols)

    # Retourne la somme des deux distances
    return manhattan_distance + hamming_distance

# 2. Fonction pour trier les instances selon la distance calculée
def sort_by_distance(dataset, target_instance, numerical_cols, categorical_cols):
    distances = []
    for idx, row in dataset.iterrows():
        distance = calculate_distance(row, target_instance, numerical_cols, categorical_cols)
        distances.append((idx, distance))

    # Trier les instances en fonction de la distance croissante
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return sorted_distances

# 3. Fonction pour déterminer la classe dominante parmi les K voisins
def get_dominant_class(neighbors, dataset, label_col):
    # Récupérer les classes des K voisins
    classes = [dataset.iloc[idx][label_col] for idx, _ in neighbors]
    # Retourner la classe la plus fréquente parmi les K voisins
    return Counter(classes).most_common(1)[0][0]

# 4. Implémentation de l'algorithme k-NN
def knn(dataset, target_instance, k, numerical_cols, categorical_cols, label_col):
    # Trier les instances par distance
    sorted_distances = sort_by_distance(dataset, target_instance, numerical_cols, categorical_cols)

    # Sélectionner les K voisins les plus proches
    k_nearest_neighbors = sorted_distances[:k]

    # Déterminer la classe dominante parmi les K voisins
    dominant_class = get_dominant_class(k_nearest_neighbors, dataset, label_col)
    return dominant_class

# 5. Déduire la classe de l'instance cible avec k = 3 puis k = 10
# Définir l'instance cible
target_instance = {
    'ep (ms)': '2024-11-20 18:09:51.000',
    'Acc_x': -0.137,
    'Acc_y': 1.066,
    'Acc_z': 0.8215,
    'Gyro_x': -6.597,
    'Gyro_y': 0.808,
    'Gyro_z': 1.985,
    'ID': 'B',
    'Label': 'medium',
    'Set': 30.0,
    'Category': 'Unknown'
}

# Définir les colonnes numériques et catégoriques
numerical_cols = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyro_x', 'Gyro_y', 'Gyro_z']
categorical_cols = ['ID', 'Category']  # Assurez-vous que 'Category' est bien référencée ici
label_col = 'Label'

# Appliquer l'algorithme k-NN pour k = 3 et k = 10
predicted_class_k3 = knn(data, target_instance, k=3, numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                         label_col=label_col)
predicted_class_k10 = knn(data, target_instance, k=10, numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                          label_col=label_col)

# Afficher les classes prédites
print(f"Classe prédite pour k=3 : {predicted_class_k3}")
print(f"Classe prédite pour k=10 : {predicted_class_k10}")
