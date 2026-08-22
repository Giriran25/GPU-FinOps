"""
GPU-FinOps | OBJECTIVE 1 — FINAL HDBSCAN MODEL + CLUSTER PROFILING

Purpose:
    1. Load the validated Objective-1 HDBSCAN matrix.
    2. Reconstruct the same eligible job population used to build the matrix.
    3. Fit the frozen final HDBSCAN configuration on all eligible jobs.
    4. Save model, assignments, metrics, summary, and cluster profiles.
    5. Stop.

Frozen final configuration:
    min_cluster_size = 15000
    min_samples      = 300
    cluster_selection_method = "eom"
    metric = "euclidean"

No tuning is performed in this script.
No K-Means/GMM is performed.
No Objective-2/Objective-3 processing is performed.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

try:
    from hdbscan.validity import validity_index
except ImportError:
    validity_index = None


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ENGINEERED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "gpu_finops_feature_engineered.csv"
)

HDBSCAN_MATRIX_FILE = (
    PROJECT_ROOT / "data" / "processed" / "objective1_hdbscan_matrix.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "objective1_hdbscan_model.joblib"
)

ASSIGNMENTS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "objective1_hdbscan_assignments.csv"
)

METRICS_FILE = (
    PROJECT_ROOT / "results" / "metrics" / "objective1_hdbscan_metrics.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT / "results" / "tables" / "objective1_hdbscan_summary.csv"
)

PROFILE_FILE = (
    PROJECT_ROOT / "results" / "tables" / "objective1_cluster_profiles.csv"
)

ALIGNMENT_AUDIT_FILE = (
    PROJECT_ROOT / "results" / "tables" / "objective1_final_alignment_audit.csv"
)


# =============================================================================
# FROZEN FINAL CONFIGURATION
# =============================================================================

MIN_CLUSTER_SIZE = 15_000
MIN_SAMPLES = 300
CLUSTER_SELECTION_METHOD = "eom"
METRIC = "euclidean"

RANDOM_STATE = 42

EXPECTED_ROWS = 599_288
EXPECTED_FEATURES = 12

# Bounded metric evaluation to avoid unnecessary quadratic memory usage.
DBCV_EVAL_CAP = 5_000
SECONDARY_METRIC_CAP = 20_000


# =============================================================================
# FEATURE DEFINITION
# =============================================================================

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
# UTILITIES
# =============================================================================

def ensure_directories() -> None:
    """Create required output directories."""
    for path in [
        MODEL_FILE,
        ASSIGNMENTS_FILE,
        METRICS_FILE,
        SUMMARY_FILE,
        PROFILE_FILE,
        ALIGNMENT_AUDIT_FILE,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_feature_engineered_data() -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    if not FEATURE_ENGINEERED_FILE.exists():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found:\n{FEATURE_ENGINEERED_FILE}"
        )

    df = pd.read_csv(
        FEATURE_ENGINEERED_FILE,
        low_memory=False,
    )

    required = CORE_FEATURES + [
        "job_name",
        "job_status",
        "has_telemetry",
        "has_execution_timing",
        "has_gpu_request",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Feature-engineered dataset is missing required columns:\n"
            + "\n".join(f"- {col}" for col in missing)
        )

    return df


def build_eligible_population(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the Objective-1 eligible population using the same
    complete-case numeric requirement used before clustering.
    """
    core = feature_df[CORE_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    values = core.to_numpy(dtype=np.float64)

    valid_numeric = np.isfinite(values).all(axis=1)
    complete_case = core.notna().all(axis=1)

    eligible_mask = complete_case & valid_numeric

    eligible = (
        feature_df.loc[eligible_mask]
        .copy()
        .reset_index(drop=True)
    )

    if eligible.empty:
        raise ValueError(
            "No eligible jobs remain after Objective-1 eligibility filtering."
        )

    if eligible["job_name"].isna().any():
        raise ValueError(
            "Eligible population contains missing job_name values."
        )

    eligible["job_name"] = eligible["job_name"].astype(str)

    duplicate_count = int(
        eligible["job_name"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Eligible population contains {duplicate_count} duplicate job_name values."
        )

    return eligible


def load_hdbscan_matrix() -> pd.DataFrame:
    """Load and validate the final 12-feature HDBSCAN matrix."""
    if not HDBSCAN_MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"HDBSCAN matrix not found:\n{HDBSCAN_MATRIX_FILE}"
        )

    matrix_df = pd.read_csv(
        HDBSCAN_MATRIX_FILE,
        low_memory=False,
    )

    if matrix_df.empty:
        raise ValueError("HDBSCAN matrix is empty.")

    # We expect exactly the final 12 feature columns.
    if matrix_df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} HDBSCAN features, "
            f"but found {matrix_df.shape[1]} columns."
        )

    if matrix_df.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} eligible jobs, "
            f"but HDBSCAN matrix contains {len(matrix_df):,} rows."
        )

    values = matrix_df.to_numpy(dtype=np.float64)

    if np.isnan(values).any():
        raise ValueError(
            "HDBSCAN matrix contains NaN values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "HDBSCAN matrix contains non-finite values."
        )

    return matrix_df


def validate_alignment(
    eligible_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the population used for final assignments.

    The HDBSCAN matrix itself contains only the 12 modeling features.
    The matrix was created from the Objective-1 eligible population in
    that population's row order. Therefore we reconstruct the same
    eligible population and verify the exact expected row count before
    attaching identifiers to model outputs.
    """
    eligible_rows = len(eligible_df)
    matrix_rows = len(matrix_df)

    if eligible_rows != matrix_rows:
        raise ValueError(
            "Final Objective-1 alignment failed:\n"
            f"- eligible population rows: {eligible_rows:,}\n"
            f"- HDBSCAN matrix rows:      {matrix_rows:,}"
        )

    audit = pd.DataFrame(
        [
            {
                "total_source_jobs": int(
                    len(pd.read_csv(
                        FEATURE_ENGINEERED_FILE,
                        usecols=["job_name"],
                    ))
                ),
                "eligible_jobs": int(eligible_rows),
                "matrix_jobs": int(matrix_rows),
                "unique_job_names": int(
                    eligible_df["job_name"].nunique()
                ),
                "duplicate_job_names": int(
                    eligible_df["job_name"].duplicated().sum()
                ),
                "alignment_status": "PASS",
            }
        ]
    )

    audit.to_csv(
        ALIGNMENT_AUDIT_FILE,
        index=False,
    )

    return audit


# =============================================================================
# MODEL
# =============================================================================

def fit_final_hdbscan(
    X: np.ndarray,
) -> hdbscan.HDBSCAN:
    """Fit the frozen final HDBSCAN configuration."""
    model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=True,
        core_dist_n_jobs=1,
    )

    model.fit(X)

    return model


# =============================================================================
# STRUCTURE + METRICS
# =============================================================================

def get_cluster_structure(
    labels: np.ndarray,
) -> dict[str, Any]:
    """Calculate cluster-size structure including noise."""
    total = len(labels)
    non_noise = labels != -1
    valid_labels = labels[non_noise]

    noise_count = int((~non_noise).sum())

    if len(valid_labels) == 0:
        return {
            "cluster_count": 0,
            "noise_count": noise_count,
            "noise_fraction": noise_count / total,
            "largest_cluster_size": 0,
            "largest_cluster_fraction": np.nan,
            "smallest_cluster_size": 0,
            "median_cluster_size": np.nan,
            "number_of_clusters_below_1_percent": 0,
        }

    _, counts = np.unique(
        valid_labels,
        return_counts=True,
    )

    return {
        "cluster_count": int(len(counts)),
        "noise_count": noise_count,
        "noise_fraction": float(noise_count / total),
        "largest_cluster_size": int(counts.max()),
        "largest_cluster_fraction": float(
            counts.max() / total
        ),
        "smallest_cluster_size": int(counts.min()),
        "median_cluster_size": float(
            np.median(counts)
        ),
        "number_of_clusters_below_1_percent": int(
            (counts / total < 0.01).sum()
        ),
    }


def bounded_metric_sample(
    X: np.ndarray,
    labels: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic non-noise evaluation sample."""
    mask = labels != -1

    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(X_valid) <= max_size:
        return X_valid, labels_valid

    rng = np.random.default_rng(RANDOM_STATE)

    indices = rng.choice(
        len(X_valid),
        size=max_size,
        replace=False,
    )

    return (
        X_valid[indices],
        labels_valid[indices],
    )


def calculate_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    model: hdbscan.HDBSCAN,
) -> dict[str, Any]:
    """Calculate final HDBSCAN metrics and persistence."""
    metrics: dict[str, Any] = get_cluster_structure(labels)

    non_noise = labels != -1
    valid_labels = labels[non_noise]

    metrics.update(
        {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "dbcv": np.nan,
            "dbcv_evaluation_sample_size": 0,
            "secondary_metric_evaluation_sample_size": 0,
            "dbcv_status": "unavailable",
        }
    )

    # Need at least 2 clusters for internal metrics.
    if len(np.unique(valid_labels)) >= 2:

        # -----------------------------
        # DBCV
        # -----------------------------
        X_dbcv, y_dbcv = bounded_metric_sample(
            X,
            labels,
            DBCV_EVAL_CAP,
        )

        metrics[
            "dbcv_evaluation_sample_size"
        ] = int(len(X_dbcv))

        if validity_index is not None:
            try:
                metrics["dbcv"] = float(
                    validity_index(
                        X_dbcv,
                        y_dbcv,
                        metric=METRIC,
                    )
                )
                metrics["dbcv_status"] = "available"
            except Exception as exc:
                metrics["dbcv_status"] = str(exc)
        else:
            metrics["dbcv_status"] = (
                "hdbscan.validity.validity_index unavailable"
            )

        # -----------------------------
        # Secondary metrics
        # -----------------------------
        X_eval, y_eval = bounded_metric_sample(
            X,
            labels,
            SECONDARY_METRIC_CAP,
        )

        metrics[
            "secondary_metric_evaluation_sample_size"
        ] = int(len(X_eval))

        try:
            metrics["silhouette"] = float(
                silhouette_score(
                    X_eval,
                    y_eval,
                )
            )
        except Exception:
            pass

        try:
            metrics["davies_bouldin"] = float(
                davies_bouldin_score(
                    X_eval,
                    y_eval,
                )
            )
        except Exception:
            pass

        try:
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(
                    X_eval,
                    y_eval,
                )
            )
        except Exception:
            pass

    # -----------------------------
    # Membership probability
    # -----------------------------
    probabilities = np.asarray(
        model.probabilities_,
        dtype=float,
    )

    non_noise_probabilities = probabilities[
        labels != -1
    ]

    metrics["mean_cluster_membership_probability"] = (
        float(np.mean(non_noise_probabilities))
        if len(non_noise_probabilities)
        else np.nan
    )

    # -----------------------------
    # Cluster persistence
    # -----------------------------
    persistence = np.asarray(
        model.cluster_persistence_,
        dtype=float,
    )

    metrics["minimum_cluster_persistence"] = (
        float(np.min(persistence))
        if len(persistence)
        else np.nan
    )

    metrics["median_cluster_persistence"] = (
        float(np.median(persistence))
        if len(persistence)
        else np.nan
    )

    metrics["maximum_cluster_persistence"] = (
        float(np.max(persistence))
        if len(persistence)
        else np.nan
    )

    return metrics


# =============================================================================
# ASSIGNMENTS
# =============================================================================

def create_assignments(
    eligible_df: pd.DataFrame,
    model: hdbscan.HDBSCAN,
) -> pd.DataFrame:
    """Create job-level HDBSCAN assignments."""
    labels = model.labels_
    probabilities = np.asarray(
        model.probabilities_,
        dtype=float,
    )

    if len(labels) != len(eligible_df):
        raise ValueError(
            "Assignment row count mismatch:\n"
            f"- model labels: {len(labels):,}\n"
            f"- eligible jobs: {len(eligible_df):,}"
        )

    assignments = pd.DataFrame(
        {
            "matrix_row_index": np.arange(
                len(labels),
                dtype=int,
            ),
            "job_name": eligible_df["job_name"].to_numpy(),
            "job_status": eligible_df["job_status"].to_numpy(),
            "cluster_id": labels.astype(int),
            "cluster_probability": probabilities,
        }
    )

    return assignments


# =============================================================================
# SUMMARY
# =============================================================================

def create_cluster_summary(
    assignments: pd.DataFrame,
    model: hdbscan.HDBSCAN,
) -> pd.DataFrame:
    """Create cluster-level summary including noise."""
    total_jobs = len(assignments)

    rows: list[dict[str, Any]] = []

    for cluster_id in sorted(
        assignments["cluster_id"].unique()
    ):
        cluster_mask = (
            assignments["cluster_id"] == cluster_id
        )

        cluster_jobs = int(cluster_mask.sum())

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": (
                    "NOISE"
                    if cluster_id == -1
                    else f"CLUSTER_{cluster_id}"
                ),
                "job_count": cluster_jobs,
                "percentage": float(
                    cluster_jobs / total_jobs * 100
                ),
                "mean_probability": float(
                    assignments.loc[
                        cluster_mask,
                        "cluster_probability",
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CLUSTER PROFILING
# =============================================================================

def create_cluster_profiles(
    eligible_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate mean/median profiles for each non-noise cluster.

    The cluster profile table is the basis for later semantic naming.
    """
    analysis_df = eligible_df[
        [
            "job_name",
            *CORE_FEATURES,
        ]
    ].copy()

    merged = analysis_df.merge(
        assignments[
            [
                "job_name",
                "cluster_id",
                "cluster_probability",
            ]
        ],
        on="job_name",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(eligible_df):
        raise ValueError(
            "Cluster profiling merge lost or duplicated jobs."
        )

    # We profile actual workload clusters only.
    non_noise = merged[
        merged["cluster_id"] != -1
    ].copy()

    if non_noise.empty:
        raise ValueError(
            "No non-noise clusters available for profiling."
        )

    grouped = non_noise.groupby(
        "cluster_id",
        sort=True,
    )

    mean_df = grouped[CORE_FEATURES].mean().add_suffix(
        "_mean"
    )

    median_df = grouped[CORE_FEATURES].median().add_suffix(
        "_median"
    )

    count_df = grouped.size().rename(
        "job_count"
    ).to_frame()

    percentage_df = (
        grouped.size()
        .div(len(merged))
        .mul(100)
        .rename("percentage")
        .to_frame()
    )

    probability_df = (
        grouped["cluster_probability"]
        .mean()
        .rename("mean_probability")
        .to_frame()
    )

    profiles = (
        count_df
        .join(percentage_df)
        .join(probability_df)
        .join(mean_df)
        .join(median_df)
        .reset_index()
    )

    # Add normalized rank columns for easier interpretation.
    for feature in [
        "gpu_utilization_intensity",
        "gpu_memory_intensity",
        "memory_intensity",
        "cpu_usage_transformed",
        "resource_efficiency_score",
        "io_intensity",
        "runtime_log",
        "gpu_demand_scale",
    ]:
        mean_col = f"{feature}_mean"

        if mean_col in profiles.columns:
            profiles[f"{feature}_rank"] = (
                profiles[mean_col]
                .rank(
                    method="average",
                    pct=True,
                )
            )

    return profiles


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = time.perf_counter()

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — FINAL HDBSCAN + CLUSTER PROFILING"
    )
    print("=" * 80)

    print(
        f"\nFrozen configuration:"
        f"\n  min_cluster_size = {MIN_CLUSTER_SIZE:,}"
        f"\n  min_samples      = {MIN_SAMPLES:,}"
        f"\n  metric           = {METRIC}"
        f"\n  selection        = {CLUSTER_SELECTION_METHOD}"
    )

    ensure_directories()

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading feature-engineered dataset...")
    feature_df = load_feature_engineered_data()

    print("[2/6] Reconstructing Objective-1 eligible population...")
    eligible_df = build_eligible_population(feature_df)

    print(f"Eligible jobs: {len(eligible_df):,}")

    # -------------------------------------------------------------------------
    # 2. Load matrix
    # -------------------------------------------------------------------------
    print("\n[3/6] Loading final HDBSCAN matrix...")
    matrix_df = load_hdbscan_matrix()

    print(
        f"Matrix shape: {matrix_df.shape[0]:,} x "
        f"{matrix_df.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # 3. Alignment
    # -------------------------------------------------------------------------
    print("\n[4/6] Validating final population alignment...")
    alignment_audit = validate_alignment(
        eligible_df,
        matrix_df,
    )

    print(
        "ALIGNMENT STATUS:",
        alignment_audit.iloc[0]["alignment_status"],
    )

    if alignment_audit.iloc[0]["alignment_status"] != "PASS":
        raise RuntimeError(
            "Final Objective-1 alignment validation failed."
        )

    # -------------------------------------------------------------------------
    # 4. Full HDBSCAN fit
    # -------------------------------------------------------------------------
    print("\n[5/6] Fitting final HDBSCAN on all eligible jobs...")
    print(
        f"Rows    : {matrix_df.shape[0]:,}"
        f"\nFeatures: {matrix_df.shape[1]}"
    )

    X = matrix_df.to_numpy(
        dtype=np.float64
    )

    fit_start = time.perf_counter()

    model = fit_final_hdbscan(X)

    fit_seconds = time.perf_counter() - fit_start

    labels = model.labels_

    print(
        f"Fit completed in {fit_seconds:.2f} seconds."
    )

    # -------------------------------------------------------------------------
    # 5. Metrics + outputs
    # -------------------------------------------------------------------------
    metrics = calculate_metrics(
        X,
        labels,
        model,
    )

    metrics_row = {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        "metric": METRIC,
        "rows": len(X),
        "features": X.shape[1],
        **metrics,
        "fit_seconds": fit_seconds,
        "total_runtime_seconds": (
            time.perf_counter() - start_time
        ),
    }

    metrics_df = pd.DataFrame([metrics_row])

    # Save metrics.
    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    # Save model.
    import joblib

    joblib.dump(
        model,
        MODEL_FILE,
    )

    # Save job-level assignments.
    assignments = create_assignments(
        eligible_df,
        model,
    )

    assignments.to_csv(
        ASSIGNMENTS_FILE,
        index=False,
    )

    # Save cluster summary.
    summary = create_cluster_summary(
        assignments,
        model,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # Save workload profiles.
    profiles = create_cluster_profiles(
        eligible_df,
        assignments,
    )

    profiles.to_csv(
        PROFILE_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    cluster_count = int(
        metrics_row["cluster_count"]
    )

    noise_fraction = float(
        metrics_row["noise_fraction"]
    )

    print("\n" + "=" * 80)
    print("FINAL HDBSCAN MODEL COMPLETED")
    print("=" * 80)

    print(
        f"\nConfiguration:"
        f"\n  min_cluster_size : {MIN_CLUSTER_SIZE:,}"
        f"\n  min_samples      : {MIN_SAMPLES:,}"
        f"\n  clusters         : {cluster_count}"
        f"\n  noise            : {noise_fraction * 100:.2f}%"
        f"\n  DBCV             : {metrics_row['dbcv']}"
        f"\n  Silhouette       : {metrics_row['silhouette']}"
        f"\n  Davies-Bouldin   : {metrics_row['davies_bouldin']}"
    )

    print("\nSaved outputs:")
    print(f"  Model      : {MODEL_FILE}")
    print(f"  Assignments: {ASSIGNMENTS_FILE}")
    print(f"  Metrics    : {METRICS_FILE}")
    print(f"  Summary    : {SUMMARY_FILE}")
    print(f"  Profiles   : {PROFILE_FILE}")
    print(f"  Alignment  : {ALIGNMENT_AUDIT_FILE}")

    print("\nCluster profile preview:")
    print(
        profiles[
            [
                "cluster_id",
                "job_count",
                "percentage",
                "mean_probability",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSTOP: Final Objective-1 HDBSCAN model and cluster "
        "profiles completed; no downstream analyses were performed."
    )

    # Cleanup.
    del X
    del model
    gc.collect()


if __name__ == "__main__":
    main()