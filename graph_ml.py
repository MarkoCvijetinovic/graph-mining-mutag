from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import random

from graph_desc import compute_descriptors_for_dataset
from graph_utils import load_mutag_pyg, draw_molecules_grid

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, OPTICS

import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

DROP_ALWAYS = {"index", "y", "avg_clustering", "transitivity"}  # drop constant or non-features


def get_feature_cols(df: pd.DataFrame, drop: Optional[set[str]] = None) -> List[str]:
    drop = DROP_ALWAYS if drop is None else (DROP_ALWAYS | drop)
    return [c for c in df.columns if c not in drop]


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray          # shape (n_samples,), -1 is noise
    X2: np.ndarray              # PCA 2D for plotting potentially
    feature_cols: List[str]


def cluster_optics(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    min_samples: int = 5,
    xi: float = 0.05,
    min_cluster_size: Optional[int] = None,
    random_state: int = 42,
) -> ClusteringResult:
    """
    Cluster using OPTICS (no need to pre-set k). Returns labels and a 2D PCA embedding.
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(df)

    X = df[list(feature_cols)].to_numpy(dtype=float)

    # Pipeline ensures scaling is done correctly
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("optics", OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
        )),
    ])

    labels = pipe.fit_predict(X)

    # 2D embedding for visualization (optional but useful)
    X_scaled = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

    return ClusteringResult(labels=labels, X2=X2, feature_cols=list(feature_cols))


def pick_random_cluster_indices(
    labels: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    exclude_noise: bool = True,
    min_size: int = 4,
) -> np.ndarray:
    """
    Pick one cluster label at random and return indices belonging to it.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique, counts = np.unique(labels, return_counts=True)

    candidates: List[int] = []
    for lab, cnt in zip(unique, counts):
        if exclude_noise and lab == -1:
            continue
        if cnt >= min_size:
            candidates.append(int(lab))

    if not candidates:
        raise ValueError("No clusters found with the requested constraints.")

    chosen = rng.choice(candidates)
    return np.where(labels == chosen)[0]


if __name__ == "__main__":
    data_list = load_mutag_pyg("train")
    df = compute_descriptors_for_dataset(data_list)

    DROP_COLS = {
        "index",
        "y",
        "avg_clustering",
        "transitivity",
    }

    FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]


    # ---- Classification (Logistic Regression and Naive Bayes) ----

    X = df[FEATURE_COLS]
    y = df["y"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print("Accuracy for Logistic Regression:", round(scores.mean(), 2), "±", round(scores.std(), 2))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print("Accuracy for GaussianNB:", round(scores.mean(), 2), "±", round(scores.std(), 2))

    # ---- Clusterization (OPTICS) ----

    # Cluster without choosing k
    res = cluster_optics(df, min_samples=5, xi=0.05)

    # Pick one cluster at random and sample molecules from it
    idxs_in_cluster = pick_random_cluster_indices(res.labels, min_size=6)

    # Choose up to 12 molecules from this cluster to display
    rng = np.random.default_rng(42)
    chosen = rng.choice(idxs_in_cluster, size=min(12, len(idxs_in_cluster)), replace=False)

    # Map descriptor-row indices -> dataset indices
    dataset_indices = df.loc[chosen, "index"].to_list()
    cluster_mols = [data_list[i] for i in dataset_indices]

    # Draw them
    draw_molecules_grid(cluster_mols, n_rows=3, n_cols=4, title_prefix="Cluster member")