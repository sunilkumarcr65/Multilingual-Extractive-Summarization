from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    import hdbscan

    HAS_HDBSCAN = True

except ImportError:
    HAS_HDBSCAN = False


# ---------------------------------------------------------
# Result Object
# ---------------------------------------------------------

@dataclass
class ClusteringResult:
    method: str
    labels: np.ndarray
    silhouette: float


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2 normalize embeddings.
    """

    return normalize(
        embeddings,
        norm="l2",
    )


def safe_cluster_count(
    n_samples: int,
    n_clusters: int,
) -> int:
    """
    Ensure valid cluster count.
    """

    return max(
        2,
        min(n_clusters, n_samples - 1),
    )


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score safely.
    """

    unique_labels = np.unique(
        labels[labels >= 0]
    )

    if len(unique_labels) < 2:
        return -1.0

    try:
        return float(
            silhouette_score(
                embeddings,
                labels,
                metric="cosine",
            )
        )

    except Exception:
        return -1.0


# ---------------------------------------------------------
# Clustering Algorithms
# ---------------------------------------------------------

def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
) -> ClusteringResult:
    """
    KMeans clustering.
    """

    n_clusters = safe_cluster_count(
        len(embeddings),
        n_clusters,
    )

    normalized = normalize_embeddings(
        embeddings
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=15,
        max_iter=300,
        init="k-means++",
    )

    labels = model.fit_predict(
        normalized
    )

    score = compute_silhouette(
        normalized,
        labels,
    )

    return ClusteringResult(
        method="kmeans",
        labels=labels,
        silhouette=score,
    )


def agglomerative_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
) -> ClusteringResult:
    """
    Agglomerative clustering.
    """

    n_clusters = safe_cluster_count(
        len(embeddings),
        n_clusters,
    )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )

    labels = model.fit_predict(
        embeddings
    )

    score = compute_silhouette(
        embeddings,
        labels,
    )

    return ClusteringResult(
        method="agglomerative",
        labels=labels,
        silhouette=score,
    )


def hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
) -> ClusteringResult:
    """
    HDBSCAN clustering.

    Automatically discovers
    cluster count.

    Noise points are reassigned
    to nearest cluster.
    """

    if not HAS_HDBSCAN:
        raise ImportError(
            "hdbscan is not installed."
        )

    normalized = normalize_embeddings(
        embeddings
    )

    model = hdbscan.HDBSCAN(
        min_cluster_size=max(
            2,
            min_cluster_size,
        ),
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = model.fit_predict(
        normalized
    )

    # -----------------------------------------------------
    # Reassign noise points
    # -----------------------------------------------------

    unique_clusters = [
        c
        for c in np.unique(labels)
        if c >= 0
    ]

    if not unique_clusters:

        fallback = kmeans_clustering(
            embeddings,
            max(2, len(embeddings) // 4),
        )

        return fallback

    if (labels == -1).any():

        centroids = np.stack([
            embeddings[labels == c].mean(axis=0)
            for c in unique_clusters
        ])

        for idx in np.where(labels == -1)[0]:

            distances = np.linalg.norm(
                centroids - embeddings[idx],
                axis=1,
            )

            nearest_cluster = unique_clusters[
                int(np.argmin(distances))
            ]

            labels[idx] = nearest_cluster

    score = compute_silhouette(
        normalized,
        labels,
    )

    return ClusteringResult(
        method="hdbscan",
        labels=labels,
        silhouette=score,
    )


# ---------------------------------------------------------
# Automatic Selection
# ---------------------------------------------------------

def choose_best_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
) -> ClusteringResult:
    """
    Automatically choose best
    clustering algorithm.

    Strategy:
    ---------------------------------------------
    small docs   -> agglomerative
    medium docs  -> compare KMeans vs Agglomerative
    large docs   -> compare KMeans vs HDBSCAN
    ---------------------------------------------
    """

    n_samples = len(embeddings)

    # -----------------------------------------------------
    # Trivial case
    # -----------------------------------------------------

    if n_samples < 2:

        return ClusteringResult(
            method="trivial",
            labels=np.zeros(
                n_samples,
                dtype=int,
            ),
            silhouette=-1.0,
        )

    # -----------------------------------------------------
    # Small document
    # -----------------------------------------------------

    if n_samples < 10:

        result = agglomerative_clustering(
            embeddings,
            n_clusters,
        )

        print(
            f"[CLUSTERING] Selected: "
            f"{result.method}"
        )

        return result

    # -----------------------------------------------------
    # Medium document
    # -----------------------------------------------------

    if n_samples < 50:

        km_result = kmeans_clustering(
            embeddings,
            n_clusters,
        )

        agg_result = agglomerative_clustering(
            embeddings,
            n_clusters,
        )

        best = (
            agg_result
            if agg_result.silhouette
            >= km_result.silhouette
            else km_result
        )

        print(
            f"[CLUSTERING] Selected: "
            f"{best.method}"
        )

        return best

    # -----------------------------------------------------
    # Large document
    # -----------------------------------------------------

    km_result = kmeans_clustering(
        embeddings,
        n_clusters,
    )

    if HAS_HDBSCAN:

        min_cluster_size = max(
            3,
            n_samples // (n_clusters * 2),
        )

        hdb_result = hdbscan_clustering(
            embeddings,
            min_cluster_size=min_cluster_size,
        )

        best = (
            hdb_result
            if hdb_result.silhouette
            >= km_result.silhouette
            else km_result
        )

        print(
            f"[CLUSTERING] Selected: "
            f"{best.method}"
        )

        return best

    print(
        f"[CLUSTERING] Selected: "
        f"{km_result.method}"
    )

    return km_result