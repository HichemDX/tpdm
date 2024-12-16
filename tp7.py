import pandas as pd
import numpy as np

# Step 1: Load and preprocess the dataset
data = pd.read_csv('./NewDatasetExos.csv', sep=';')

# Specify columns to use (including numerical and categorical if needed)
columns_to_use = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyro_x', 'Gyro_y', 'Gyro_z']
data = data[columns_to_use]

# Convert to numeric and handle non-numeric values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()  # Drop rows with NaN values

# Identify numerical and categorical columns
numerical_cols = columns_to_use  # Specify numerical columns here
categorical_cols = []  # If you have categorical columns, list them here

# Step 2: Distance Function
def calculate_distance(instance1, instance2, numerical_cols, categorical_cols):
    """Calculate a combined distance using Manhattan for numerical and Hamming for categorical columns."""
    # Manhattan distance for numerical columns
    manhattan_distance = sum(abs(instance1[col] - instance2[col]) for col in numerical_cols)
    # Hamming distance for categorical columns
    hamming_distance = sum(instance1[col] != instance2[col] for col in categorical_cols)
    return manhattan_distance + hamming_distance

# Step 3: Calculate the centroid of a set of instances
def calculate_centroid(cluster, numerical_cols):
    """
    Calculate the centroid of a cluster for numerical data.
    Parameters:
    - cluster: List of data points (2D NumPy array).
    """
    return {col: np.mean([instance[col] for instance in cluster]) for col in numerical_cols}

# Step 4: Find the cluster to which a given instance is the closest
def find_closest_cluster(instance, centroids, numerical_cols, categorical_cols):
    """
    Find the closest cluster for a given instance.
    Parameters:
    - instance: Data point.
    - centroids: List of centroids.
    """
    distances = [calculate_distance(instance, centroid, numerical_cols, categorical_cols) for centroid in centroids]
    return np.argmin(distances)

# Step 5: Implement the k-means algorithm
def k_means(data, k, numerical_cols, categorical_cols, max_iterations=1000):
    """
    Implement k-means clustering.
    Parameters:
    - data: DataFrame of dataset.
    - k: Number of clusters.
    - numerical_cols: List of numerical columns.
    - categorical_cols: List of categorical columns.
    - max_iterations: Maximum iterations to converge.
    """
    data = data.to_dict('records')  # Convert to a list of dictionaries for flexible column access
    np.random.seed(42)  # Set random seed for reproducibility

    # Step 1: Randomly initialize centroids
    centroids = [data[idx] for idx in np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iterations):
        # Step 2: Assign points to the nearest centroid
        clusters = [[] for _ in range(k)]
        for instance in data:
            cluster_idx = find_closest_cluster(instance, centroids, numerical_cols, categorical_cols)
            clusters[cluster_idx].append(instance)

        # Step 3: Calculate new centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroids.append(calculate_centroid(cluster, numerical_cols))
            else:
                # Reinitialize empty cluster
                new_centroids.append(data[np.random.choice(len(data))])

        # Check for convergence
        if all(calculate_distance(c1, c2, numerical_cols, categorical_cols) < 1e-6
               for c1, c2 in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    # Label the dataset with the cluster assignments
    labeled_data = []
    for cluster_idx, cluster in enumerate(clusters):
        for instance in cluster:
            labeled_data.append({**instance, 'Cluster': cluster_idx})

    return clusters, centroids, pd.DataFrame(labeled_data)

# Step 6: Test the algorithm with k=2, k=5, and k=6
results = {}
for k in [2, 5, 6]:
    print(f"\nTesting with k={k}")
    clusters, centroids, labeled_data = k_means(data, k, numerical_cols, categorical_cols)
    cluster_sizes = [len(cluster) for cluster in clusters]
    print(f"Number of points in each cluster: {cluster_sizes}")
    results[k] = labeled_data
