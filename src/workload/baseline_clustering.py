"""
GPU-FinOps | OBJECTIVE 1 — K-MEANS + GMM BASELINES

Purpose
-------
1. Load the same validated 12-feature Objective-1 HDBSCAN matrix.
2. Load the final HDBSCAN assignments.
3. Fit:
      - K-Means, k=10
      - Gaussian Mixture Model, n_components=10
4. Evaluate:
      A. Full eligible population
      B. HDBSCAN non-noise subset
5. Save models, assignments, metrics, and comparison table.
6. Stop.

No HDBSCAN tuning.
No K-Means/GMM hyperparameter search.
No stability analysis.
No Objective 2/3 processing.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_matrix.csv"
)

HDBSCAN_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_assignments.csv"
)

KMEANS_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "objective1_kmeans_model.joblib"
)

GMM_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "objective1_gmm_model.joblib"
)

KMEANS_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_kmeans_assignments.csv"
)

GMM_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_gmm_assignments.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "objective1_baseline_clustering_metrics.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_clustering_comparison.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42
N_CLUSTERS = 10
N_INIT = 10

EXPECTED_ROWS = 599_288
EXPECTED_FEATURES = 12

FULL_EVAL_CAP = 20_000
NON_NOISE_EVAL_CAP = 20_000

CORE_FEATURES = [
    "plan_cpu_mean",
    "plan_mem_mean",
    "gpu_demand_scale",
    "gpu_utilization_intensity",
    "gpu_memory_intensity",
    "memory_intensity",
    "cpu_usage_transformed",
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "io_intensity",
    "runtime_log",
    "task_fanout",
]


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def ensure_directories() -> None:
    paths = [
        KMEANS_MODEL_FILE,
        GMM_MODEL_FILE,
        KMEANS_ASSIGNMENTS_FILE,
        GMM_ASSIGNMENTS_FILE,
        METRICS_FILE,
        COMPARISON_FILE,
    ]

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_matrix() -> pd.DataFrame:
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Objective-1 matrix not found:\n{MATRIX_FILE}"
        )

    df = pd.read_csv(
        MATRIX_FILE,
        low_memory=False,
    )

    if df.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            "Unexpected HDBSCAN matrix shape.\n"
            f"Expected: {(EXPECTED_ROWS, EXPECTED_FEATURES)}\n"
            f"Found   : {df.shape}"
        )

    values = df.to_numpy(dtype=np.float64)

    if np.isnan(values).any():
        raise ValueError("HDBSCAN matrix contains NaN values.")

    if not np.isfinite(values).all():
        raise ValueError("HDBSCAN matrix contains Inf/non-finite values.")

    return df


def load_hdbscan_assignments() -> pd.DataFrame:
    if not HDBSCAN_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Final HDBSCAN assignments not found:\n"
            f"{HDBSCAN_ASSIGNMENTS_FILE}"
        )

    df = pd.read_csv(
        HDBSCAN_ASSIGNMENTS_FILE,
        low_memory=False,
    )

    required = {
        "matrix_row_index",
        "cluster_id",
        "cluster_probability",
    }

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            f"HDBSCAN assignments missing required columns: {missing}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} assignments, got {len(df):,}."
        )

    if not np.array_equal(
        df["matrix_row_index"].to_numpy(dtype=int),
        np.arange(EXPECTED_ROWS, dtype=int),
    ):
        raise ValueError(
            "HDBSCAN assignment matrix_row_index is not aligned "
            "with the Objective-1 matrix."
        )

    return df


# =============================================================================
# EVALUATION SAMPLING
# =============================================================================

def stratified_sample_indices(
    n_rows: int,
    max_size: int,
    labels: np.ndarray,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, str]:
    if n_rows <= max_size:
        return np.arange(n_rows, dtype=int), "full_population"

    all_indices = np.arange(n_rows, dtype=int)

    try:
        _, selected_indices = train_test_split(
            all_indices,
            test_size=max_size,
            random_state=seed,
            stratify=labels,
        )
        return np.sort(selected_indices), "stratified"
    except ValueError:
        rng = np.random.default_rng(seed)
        return (
            np.sort(rng.choice(n_rows, size=max_size, replace=False)),
            "random_fallback",
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    evaluation_name: str,
    max_eval_size: int = FULL_EVAL_CAP,
    evaluation_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate clustering metrics on a deterministic evaluation sample.

    Metrics:
        Silhouette
        Davies-Bouldin
        Calinski-Harabasz
    """
    n_rows = len(labels)

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {
            "evaluation": evaluation_name,
            "evaluation_rows": n_rows,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "evaluation_indices": np.arange(n_rows, dtype=int),
            "sampling_method": "full_population",
            "error": "Fewer than two clusters.",
        }

    if evaluation_indices is None:
        indices, sampling_method = stratified_sample_indices(
            n_rows=n_rows,
            max_size=max_eval_size,
            labels=labels,
        )
    else:
        indices = np.asarray(evaluation_indices, dtype=int)
        sampling_method = "shared_stratified"

    X_eval = X[indices]
    y_eval = labels[indices]

    # The deterministic sample may theoretically omit a cluster.
    if len(np.unique(y_eval)) < 2:
        return {
            "evaluation": evaluation_name,
            "evaluation_rows": len(indices),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "evaluation_indices": indices,
            "sampling_method": sampling_method,
            "error": (
                "Evaluation sample contained fewer than two clusters."
            ),
        }

    result: dict[str, Any] = {
        "evaluation": evaluation_name,
        "evaluation_rows": len(indices),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "evaluation_indices": indices,
        "sampling_method": sampling_method,
        "error": "",
    }

    try:
        result["silhouette"] = float(
            silhouette_score(
                X_eval,
                y_eval,
            )
        )
    except Exception as exc:
        result["error"] += f"silhouette: {exc}; "

    try:
        result["davies_bouldin"] = float(
            davies_bouldin_score(
                X_eval,
                y_eval,
            )
        )
    except Exception as exc:
        result["error"] += f"davies_bouldin: {exc}; "

    try:
        result["calinski_harabasz"] = float(
            calinski_harabasz_score(
                X_eval,
                y_eval,
            )
        )
    except Exception as exc:
        result["error"] += f"calinski_harabasz: {exc}; "

    return result


def calculate_cluster_distribution(
    labels: np.ndarray,
) -> dict[str, Any]:
    labels = labels[labels != -1]
    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    total = len(labels)

    return {
        "cluster_count": int(len(unique_labels)),
        "smallest_cluster_size": int(counts.min()),
        "median_cluster_size": float(np.median(counts)),
        "largest_cluster_size": int(counts.max()),
        "largest_cluster_fraction": float(
            counts.max() / total
        ),
    }


# =============================================================================
# K-MEANS
# =============================================================================

def fit_kmeans(
    X: np.ndarray,
) -> KMeans:
    model = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )

    model.fit(X)

    return model


# =============================================================================
# GMM
# =============================================================================

def fit_gmm(
    X: np.ndarray,
) -> GaussianMixture:
    model = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    model.fit(X)

    return model


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = time.perf_counter()

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — K-MEANS + GMM BASELINES"
    )
    print("=" * 80)

    print(
        f"\nFixed comparison configuration:"
        f"\n  clusters        : {N_CLUSTERS}"
        f"\n  random_state    : {RANDOM_STATE}"
        f"\n  K-Means n_init  : {N_INIT}"
        f"\n  GMM n_init      : {N_INIT}"
    )

    ensure_directories()

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading Objective-1 matrix...")
    matrix_df = load_matrix()

    X = matrix_df.to_numpy(
        dtype=np.float64
    )

    print(
        f"Rows    : {X.shape[0]:,}"
        f"\nFeatures: {X.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # 2. Load HDBSCAN assignments
    # -------------------------------------------------------------------------
    print("\n[2/7] Loading final HDBSCAN assignments...")
    hdbscan_df = load_hdbscan_assignments()

    hdbscan_labels = (
        hdbscan_df["cluster_id"]
        .to_numpy(dtype=int)
    )

    hdbscan_non_noise_mask = hdbscan_labels != -1

    non_noise_count = int(
        hdbscan_non_noise_mask.sum()
    )

    print(
        f"HDBSCAN noise jobs    : "
        f"{int((~hdbscan_non_noise_mask).sum()):,}"
    )
    print(
        f"HDBSCAN non-noise jobs: "
        f"{non_noise_count:,}"
    )

    # -------------------------------------------------------------------------
    # 3. Load K-Means
    # -------------------------------------------------------------------------
    print("\n[3/7] Loading existing K-Means model...")
    if not KMEANS_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Existing K-Means model not found:\n{KMEANS_MODEL_FILE}"
        )
    kmeans_model = joblib.load(KMEANS_MODEL_FILE)

    kmeans_labels = (
        kmeans_model.predict(X).astype(int)
    )

    print(f"Loaded K-Means model with {kmeans_model.n_clusters} clusters.")

    # -------------------------------------------------------------------------
    # 4. Load GMM
    # -------------------------------------------------------------------------
    print("\n[4/7] Loading existing GMM model...")
    if not GMM_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Existing GMM model not found:\n{GMM_MODEL_FILE}"
        )
    gmm_model = joblib.load(GMM_MODEL_FILE)

    gmm_labels = (
        gmm_model.predict(X).astype(int)
    )

    print(f"Loaded GMM model with {gmm_model.n_components} components.")

    # -------------------------------------------------------------------------
    # 5. Save models and assignments
    # -------------------------------------------------------------------------
    print("\n[5/7] Saving assignments...")

    kmeans_assignments = pd.DataFrame(
        {
            "matrix_row_index": np.arange(
                EXPECTED_ROWS,
                dtype=int,
            ),
            "cluster_id": kmeans_labels,
        }
    )

    gmm_probabilities = gmm_model.predict_proba(X)
    gmm_confidence = gmm_probabilities.max(axis=1)

    gmm_assignments = pd.DataFrame(
        {
            "matrix_row_index": np.arange(
                EXPECTED_ROWS,
                dtype=int,
            ),
            "cluster_id": gmm_labels,
            "cluster_probability": gmm_confidence,
        }
    )

    kmeans_assignments.to_csv(
        KMEANS_ASSIGNMENTS_FILE,
        index=False,
    )

    gmm_assignments.to_csv(
        GMM_ASSIGNMENTS_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6. Evaluate
    # -------------------------------------------------------------------------
    print("\n[6/7] Evaluating clustering baselines...")

    metric_rows: list[dict[str, Any]] = []

    # -----------------------------
    # HDBSCAN — FULL
    # -----------------------------
    hdbscan_full_dist = calculate_cluster_distribution(
        hdbscan_labels
    )

    hdbscan_full_metrics = calculate_clustering_metrics(
        X=X,
        labels=hdbscan_labels,
        evaluation_name="full_population",
        max_eval_size=FULL_EVAL_CAP,
    )

    metric_rows.append(
        {
            "method": "HDBSCAN",
            "evaluation": "full_population",
            "model_cluster_count": hdbscan_full_dist["cluster_count"],
            "noise_fraction": float(
                (~hdbscan_non_noise_mask).mean()
            ),
            "largest_cluster_fraction": hdbscan_full_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": hdbscan_full_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": hdbscan_full_dist[
                "median_cluster_size"
            ],
            "silhouette": hdbscan_full_metrics[
                "silhouette"
            ],
            "davies_bouldin": hdbscan_full_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": hdbscan_full_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": hdbscan_full_metrics[
                "evaluation_rows"
            ],
            "sampling_method": hdbscan_full_metrics["sampling_method"],
            "error": hdbscan_full_metrics["error"],
        }
    )

    # -----------------------------
    # K-Means — FULL
    # -----------------------------
    kmeans_full_dist = calculate_cluster_distribution(
        kmeans_labels
    )

    kmeans_full_metrics = calculate_clustering_metrics(
        X=X,
        labels=kmeans_labels,
        evaluation_name="full_population",
        max_eval_size=FULL_EVAL_CAP,
    )

    metric_rows.append(
        {
            "method": "K-Means",
            "evaluation": "full_population",
            "model_cluster_count": N_CLUSTERS,
            "noise_fraction": 0.0,
            "largest_cluster_fraction": kmeans_full_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": kmeans_full_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": kmeans_full_dist[
                "median_cluster_size"
            ],
            "silhouette": kmeans_full_metrics[
                "silhouette"
            ],
            "davies_bouldin": kmeans_full_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": kmeans_full_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": kmeans_full_metrics[
                "evaluation_rows"
            ],
            "sampling_method": kmeans_full_metrics["sampling_method"],
            "error": kmeans_full_metrics["error"],
        }
    )

    # -----------------------------
    # GMM — FULL
    # -----------------------------
    gmm_full_dist = calculate_cluster_distribution(
        gmm_labels
    )

    gmm_full_metrics = calculate_clustering_metrics(
        X=X,
        labels=gmm_labels,
        evaluation_name="full_population",
        max_eval_size=FULL_EVAL_CAP,
    )

    metric_rows.append(
        {
            "method": "GMM",
            "evaluation": "full_population",
            "model_cluster_count": N_CLUSTERS,
            "noise_fraction": 0.0,
            "largest_cluster_fraction": gmm_full_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": gmm_full_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": gmm_full_dist[
                "median_cluster_size"
            ],
            "silhouette": gmm_full_metrics[
                "silhouette"
            ],
            "davies_bouldin": gmm_full_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": gmm_full_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": gmm_full_metrics[
                "evaluation_rows"
            ],
            "sampling_method": gmm_full_metrics["sampling_method"],
            "error": gmm_full_metrics["error"],
        }
    )

    # -------------------------------------------------------------------------
    # Fair non-noise subset
    # -------------------------------------------------------------------------
    X_non_noise = X[hdbscan_non_noise_mask]

    hdbscan_non_noise_labels = (
        hdbscan_labels[hdbscan_non_noise_mask]
    )

    kmeans_non_noise_labels = (
        kmeans_labels[hdbscan_non_noise_mask]
    )

    gmm_non_noise_labels = (
        gmm_labels[hdbscan_non_noise_mask]
    )

    non_noise_eval_indices, non_noise_sampling_method = (
        stratified_sample_indices(
            n_rows=len(X_non_noise),
            max_size=NON_NOISE_EVAL_CAP,
            labels=hdbscan_non_noise_labels,
        )
    )

    # HDBSCAN non-noise
    hdbscan_nn_dist = calculate_cluster_distribution(
        hdbscan_non_noise_labels
    )

    hdbscan_nn_metrics = calculate_clustering_metrics(
        X=X_non_noise,
        labels=hdbscan_non_noise_labels,
        evaluation_name="hdbscan_non_noise_subset",
        max_eval_size=NON_NOISE_EVAL_CAP,
        evaluation_indices=non_noise_eval_indices,
    )

    metric_rows.append(
        {
            "method": "HDBSCAN",
            "evaluation": "hdbscan_non_noise_subset",
            "model_cluster_count": hdbscan_nn_dist["cluster_count"],
            "noise_fraction": 0.0,
            "largest_cluster_fraction": hdbscan_nn_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": hdbscan_nn_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": hdbscan_nn_dist[
                "median_cluster_size"
            ],
            "silhouette": hdbscan_nn_metrics[
                "silhouette"
            ],
            "davies_bouldin": hdbscan_nn_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": hdbscan_nn_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": hdbscan_nn_metrics[
                "evaluation_rows"
            ],
            "sampling_method": non_noise_sampling_method,
            "error": hdbscan_nn_metrics["error"],
        }
    )

    # K-Means on HDBSCAN non-noise subset
    kmeans_nn_dist = calculate_cluster_distribution(
        kmeans_non_noise_labels
    )

    kmeans_nn_metrics = calculate_clustering_metrics(
        X=X_non_noise,
        labels=kmeans_non_noise_labels,
        evaluation_name="hdbscan_non_noise_subset",
        max_eval_size=NON_NOISE_EVAL_CAP,
        evaluation_indices=non_noise_eval_indices,
    )

    metric_rows.append(
        {
            "method": "K-Means",
            "evaluation": "hdbscan_non_noise_subset",
            "model_cluster_count": N_CLUSTERS,
            "noise_fraction": 0.0,
            "largest_cluster_fraction": kmeans_nn_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": kmeans_nn_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": kmeans_nn_dist[
                "median_cluster_size"
            ],
            "silhouette": kmeans_nn_metrics[
                "silhouette"
            ],
            "davies_bouldin": kmeans_nn_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": kmeans_nn_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": kmeans_nn_metrics[
                "evaluation_rows"
            ],
            "sampling_method": non_noise_sampling_method,
            "error": kmeans_nn_metrics["error"],
        }
    )

    # GMM on HDBSCAN non-noise subset
    gmm_nn_dist = calculate_cluster_distribution(
        gmm_non_noise_labels
    )

    gmm_nn_metrics = calculate_clustering_metrics(
        X=X_non_noise,
        labels=gmm_non_noise_labels,
        evaluation_name="hdbscan_non_noise_subset",
        max_eval_size=NON_NOISE_EVAL_CAP,
        evaluation_indices=non_noise_eval_indices,
    )

    metric_rows.append(
        {
            "method": "GMM",
            "evaluation": "hdbscan_non_noise_subset",
            "model_cluster_count": N_CLUSTERS,
            "noise_fraction": 0.0,
            "largest_cluster_fraction": gmm_nn_dist[
                "largest_cluster_fraction"
            ],
            "smallest_cluster_size": gmm_nn_dist[
                "smallest_cluster_size"
            ],
            "median_cluster_size": gmm_nn_dist[
                "median_cluster_size"
            ],
            "silhouette": gmm_nn_metrics[
                "silhouette"
            ],
            "davies_bouldin": gmm_nn_metrics[
                "davies_bouldin"
            ],
            "calinski_harabasz": gmm_nn_metrics[
                "calinski_harabasz"
            ],
            "evaluation_rows": gmm_nn_metrics[
                "evaluation_rows"
            ],
            "sampling_method": non_noise_sampling_method,
            "error": gmm_nn_metrics["error"],
        }
    )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Comparison summary
    # -------------------------------------------------------------------------
    comparison_df = metrics_df[
        [
            "method",
            "evaluation",
            "model_cluster_count",
            "noise_fraction",
            "largest_cluster_fraction",
            "smallest_cluster_size",
            "median_cluster_size",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "evaluation_rows",
            "sampling_method",
            "error",
        ]
    ].copy()

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. Final reporting
    # -------------------------------------------------------------------------
    print("\n[7/7] Baseline comparison completed.")

    print("\n" + "=" * 80)
    print("OBJECTIVE 1 — CORRECTED BASELINE COMPARISON")
    print("=" * 80)

    print("\nComparison:")
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  K-Means model      : {KMEANS_MODEL_FILE}")
    print(f"  GMM model          : {GMM_MODEL_FILE}")
    print(f"  K-Means assignments: {KMEANS_ASSIGNMENTS_FILE}")
    print(f"  GMM assignments    : {GMM_ASSIGNMENTS_FILE}")
    print(f"  Metrics            : {METRICS_FILE}")
    print(f"  Comparison         : {COMPARISON_FILE}")

    total_seconds = time.perf_counter() - start_time

    print(
        f"\nTotal runtime: {total_seconds:.2f} seconds."
    )

    print(
        "\nSTOP: Corrected Objective-1 baseline evaluation completed; "
        "no stability analysis or downstream objectives were performed."
    )

    del X
    del X_non_noise
    del kmeans_model
    del gmm_model
    gc.collect()


if __name__ == "__main__":
    main()