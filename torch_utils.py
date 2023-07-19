import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def cluster_points(data, method='kmeans', max_clusters=10):
    """
    Perform clustering on a set of data points.

    Args:
        data (numpy.ndarray): Array of data points with shape (N, 2).
        method (str): Clustering method to use. Options: 'kmeans', 'dbscan', 'agglomerative'.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        dict: A dictionary containing the clustering results.
            - 'labels': Cluster labels.
            - 'optimal_clusters': Optimal number of clusters based on the Elbow Method (applicable to K-means only).
            - 'silhouette_score': Silhouette score (applicable to K-means and Agglomerative Clustering only).
    """
    if method == 'kmeans':
        silhouette_scores = []
        distortions = []

        for n_clusters in range(2, max_clusters+1):
            model = KMeans(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(data)
            distortions.append(model.inertia_)
            if n_clusters > 1:
                silhouette_scores.append(silhouette_score(data, labels))

        optimal_clusters = np.argmin(distortions) + 2 if len(distortions) > 1 else 1

        results = {
            'labels': labels,
            'optimal_clusters': optimal_clusters,
            'silhouette_score': silhouette_scores[-1] if silhouette_scores else None
        }
    elif method == 'dbscan':
        model = DBSCAN()
        labels = model.fit_predict(data)

        results = {
            'labels': labels,
            'optimal_clusters': len(np.unique(labels)),
            'silhouette_score': None
        }
    elif method == 'agglomerative':
        model = AgglomerativeClustering()
        labels = model.fit_predict(data)

        silhouette_score_value = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else None

        results = {
            'labels': labels,
            'optimal_clusters': len(np.unique(labels)),
            'silhouette_score': silhouette_score_value
        }
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans', 'dbscan', or 'agglomerative'.")

    return results
