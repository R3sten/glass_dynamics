import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
from glass_dynamics.core.logger import logger

def get_spearman_linkage(X: pd.DataFrame) -> tuple:
    """
    Computes the Spearman correlation matrix and the hierarchical clustering linkage.
    Spearman is used to capture monotonic, non-linear correlations.

    Args:
        X (pd.DataFrame): DataFrame containing only the numerical features.

    Returns:
        tuple: (correlation_matrix, linkage_matrix)
    """
    logger.info("Computing Spearman correlation matrix...")
    corr_matrix = X.corr(method='spearman')

    # Ensure correlation matrix is symmetric and clip values to [-1, 1] for numerical stability
    corr_matrix = corr_matrix.clip(-1, 1)

    # Distance matrix: features with correlation = 1 have distance = 0
    # Features with correlation = -1 also have distance = 0 (we care about magnitude)
    distance_matrix = 1 - np.abs(corr_matrix.values)

    # Compute Ward linkage
    logger.info("Computing hierarchical clustering linkage (Ward)...")
    linkage_matrix = hierarchy.ward(squareform(distance_matrix))

    return corr_matrix, linkage_matrix

def select_features_from_clusters(X: pd.DataFrame, y: pd.Series, threshold: float = 0.2) -> list:
    """
    Groups features into clusters based on their correlation distance and
    selects one representative feature per cluster (the one most correlated with the target).

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Target variable (e.g., Tg_fitted) to evaluate feature importance.
        threshold (float): Distance threshold for clustering.
                           E.g., 0.2 means features with |corr| > 0.8 are clustered together.

    Returns:
        list: A list of selected feature names (one per cluster).
    """
    #TODO: da capire cosa è questo spearman
    corr_matrix, linkage_matrix = get_spearman_linkage(X)

    # Form flat clusters from the hierarchical clustering defined by the linkage matrix
    cluster_labels = hierarchy.fcluster(linkage_matrix, threshold, criterion='distance')

    cluster_to_features = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_to_features[cluster_id].append(X.columns[idx])

    selected_features = []

    logger.info(f"Formed {len(cluster_to_features)} clusters with distance threshold {threshold}.")

    # For each cluster, pick the feature most correlated with the target 'y'
    for cluster_id, features in cluster_to_features.items():
        if len(features) == 1:
            selected_features.append(features[0])
        else:
            # Calculate absolute correlation of these features with the target
            correlations = X[features].apply(lambda col: np.abs(col.corr(y)))
            best_feature = correlations.idxmax()
            selected_features.append(best_feature)
            logger.debug(f"Cluster {cluster_id}: {features} -> Selected: {best_feature}")

    return selected_features