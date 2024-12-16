import pandas as pd
import numpy as np

# Chargement des données
data = pd.read_csv('./NewDatasetExos.csv', sep=';')

# Supprimez les colonnes non pertinentes
data = data.drop(columns=["Exercise", "ep (ms)", "ID"], errors='ignore')

# Conversion des colonnes numériques
numerical_cols = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyro_x', 'Gyro_y', 'Gyro_z']
categorical_cols = ['Label', 'Set']  # Ajustez si nécessaire

# Assurez-vous que les colonnes numériques sont bien au format numérique
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')


# 1. Fonction de calcul des distances (réutilisation de TP5)
def calculate_distance(instance1, instance2, numerical_cols):
    """Calcule la distance Euclidienne ou Manhattan entre deux instances."""
    euclidean_distance = np.sqrt(sum((instance1[col] - instance2[col]) ** 2 for col in numerical_cols))
    return euclidean_distance


# 2. Fonction pour calculer le centroïde d'un cluster
def calculate_centroid(cluster, numerical_cols):
    """Calcule le centroïde d'un cluster donné."""
    return cluster[numerical_cols].mean()


# 3. Fonction pour attribuer un cluster à une instance donnée
def assign_cluster(instance, centroids, numerical_cols):
    """Attribue l'instance au cluster le plus proche."""
    distances = [calculate_distance(instance, centroids.iloc[i], numerical_cols) for i in range(len(centroids))]
    return np.argmin(distances)


# 4. Implémentation de l'algorithme k-means
def kmeans(data, k, numerical_cols):
    """Algorithme k-means pour la classification en clusters."""
    # Initialisation aléatoire des centroïdes
    centroids = data[numerical_cols].sample(n=k, random_state=42).reset_index(drop=True)
    prev_centroids = None
    clusters = None

    while True:
        # Attribution des clusters
        clusters = data.apply(lambda row: assign_cluster(row, centroids, numerical_cols), axis=1)

        # Calcul des nouveaux centroïdes
        new_centroids = []
        for cluster_id in range(k):
            cluster_data = data[clusters == cluster_id]
            if not cluster_data.empty:
                new_centroid = calculate_centroid(cluster_data, numerical_cols)
                new_centroids.append(new_centroid)
            else:
                # Si un cluster est vide, réinitialisez le centroïde
                new_centroids.append(centroids.iloc[cluster_id])
        new_centroids = pd.DataFrame(new_centroids)

        # Vérification de convergence
        if prev_centroids is not None and all(
                np.allclose(new_centroids.iloc[i], centroids.iloc[i]) for i in range(k)
        ):
            break

        prev_centroids = centroids
        centroids = new_centroids

    # Ajouter les clusters au jeu de données
    data['Cluster'] = clusters
    return data


# 5. Test de l'algorithme avec k=2, k=5, et k=6
kmeans_result_k2 = kmeans(data.copy(), k=2, numerical_cols=numerical_cols)
kmeans_result_k5 = kmeans(data.copy(), k=5, numerical_cols=numerical_cols)
kmeans_result_k6 = kmeans(data.copy(), k=6, numerical_cols=numerical_cols)

# Sauvegarder les résultats pour inspection
kmeans_result_k2.to_csv("kmeans_result_k2.csv", index=False)
kmeans_result_k5.to_csv("kmeans_result_k5.csv", index=False)
kmeans_result_k6.to_csv("kmeans_result_k6.csv", index=False)

# Affichage des résultats
print("Résultats pour k=2 :")
print(kmeans_result_k2['Cluster'].value_counts())

print("\nRésultats pour k=5 :")
print(kmeans_result_k5['Cluster'].value_counts())

print("\nRésultats pour k=6 :")
print(kmeans_result_k6['Cluster'].value_counts())
