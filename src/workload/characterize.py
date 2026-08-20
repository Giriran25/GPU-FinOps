"""
GPU-FinOps
----------
Objective 1: Workload Characterization

Primary model:
    HDBSCAN

Purpose:
    1. Tune HDBSCAN on a representative sample.
    2. Select a stable parameter configuration.
    3. Fit the selected configuration on the full eligible population.
    4. Save cluster assignments and diagnostics.
    5. Do NOT manually specify the final number of workload classes.

Important:
    - HDBSCAN is the primary clustering method.
    - K-Means and GMM are handled later as baselines.
    - No manual workload labels are used.
"""

from __future__ import annotations

import gc
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import hdbscan
try:
    from hdbscan.validity import validity_index
except ImportError:
    validity_index = None
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import train_test_split


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_matrix.csv"
)

FEATURE_ENGINEERED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gpu_finops_feature_engineered.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "objective1_hdbscan_model.joblib"
)

ASSIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_assignments.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "objective1_hdbscan_metrics.csv"
)

TUNING_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "objective1_hdbscan_tuning_stage1.csv"
)

TOP3_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_hdbscan_top3_candidates.csv"
)

FULL_CONFIRMATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "objective1_hdbscan_full_confirmation.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_hdbscan_summary.csv"
)


# ============================================================================
# SETTINGS
# ============================================================================

RANDOM_STATE = 42

# Tuning sample.
TUNING_SAMPLE_SIZE = 200_000

# Evaluation sample caps.
DBCV_EVAL_SAMPLE_SIZE = 5_000
SECONDARY_METRIC_SAMPLE_SIZE = 10_000

# HDBSCAN parameter ranges.
MIN_CLUSTER_SIZE_VALUES = [
    10000,
    20000,
    30000,
    40000,
]

MIN_SAMPLES_VALUES = [
    300,
    1000,
    2000,
]

CLUSTER_SELECTION_METHOD = "eom"

# Noise sanity thresholds.
MAX_ACCEPTABLE_NOISE = 0.40
MIN_ACCEPTABLE_NOISE = 0.01


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create required output directories."""

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ASSIGNMENT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TUNING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOAD MATRIX
# ============================================================================

def load_hdbscan_matrix() -> pd.DataFrame:
    """Load and validate the final HDBSCAN feature matrix."""

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"HDBSCAN matrix not found:\n{MATRIX_FILE}"
        )

    df = pd.read_csv(
        MATRIX_FILE,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            "HDBSCAN matrix is empty."
        )

    if df.isna().any().any():
        raise ValueError(
            "HDBSCAN matrix contains NaN values."
        )

    if np.isinf(
        df.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "HDBSCAN matrix contains infinite values."
        )

    return df


# ============================================================================
# SAMPLE
# ============================================================================

def load_feature_engineered_data() -> pd.DataFrame:
    """Load source metadata used only to construct sampling strata."""

    if not FEATURE_ENGINEERED_FILE.exists():
        raise FileNotFoundError(
            f"Feature-engineered data not found:\n{FEATURE_ENGINEERED_FILE}"
        )

    return pd.read_csv(
        FEATURE_ENGINEERED_FILE,
        usecols=[
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
            "job_status",
        ],
        low_memory=False,
    )


def make_random_tuning_sample(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Create the documented reproducible fallback sample."""

    return X.sample(
        n=min(TUNING_SAMPLE_SIZE, len(X)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def make_tuning_sample(
    X: pd.DataFrame,
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Create an exactly sized, reproducible stratified tuning sample."""

    if len(X) <= TUNING_SAMPLE_SIZE:
        return X.reset_index(drop=True).copy(), "full_population"

    try:
        numeric_features = X.apply(pd.to_numeric, errors="coerce")
        source_numeric = source_df.loc[:, X.columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        eligible_mask = np.isfinite(source_numeric.to_numpy()).all(axis=1)
        eligible_source = source_df.loc[eligible_mask].reset_index(drop=True)

        if len(eligible_source) != len(X):
            raise ValueError("Source and HDBSCAN matrix rows are not aligned.")

        gpu_bucket = pd.qcut(
            numeric_features["gpu_demand_scale"].rank(method="first"),
            q=5,
            labels=False,
            duplicates="drop",
        ).fillna(-1).astype(str)
        runtime_bucket = pd.qcut(
            numeric_features["runtime_log"].rank(method="first"),
            q=5,
            labels=False,
            duplicates="drop",
        ).fillna(-1).astype(str)
        status = eligible_source["job_status"].fillna("<missing>").astype(str)
        strata = status + "|gpu_" + gpu_bucket + "|runtime_" + runtime_bucket

        positions = np.arange(len(X))
        selected_positions, _ = train_test_split(
            positions,
            train_size=TUNING_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=strata,
        )
        return X.iloc[selected_positions].reset_index(drop=True), "stratified"
    except Exception as exc:
        print(
            "WARNING: stratified sampling unavailable; using reproducible "
            f"random fallback ({exc})."
        )
        return make_random_tuning_sample(X), "random_fallback"


# ============================================================================
# SAFE METRICS
# ============================================================================

def calculate_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    dbcv_sample_size: int = DBCV_EVAL_SAMPLE_SIZE,
    secondary_metric_sample_size: int = SECONDARY_METRIC_SAMPLE_SIZE,
) -> dict[str, float]:
    """Calculate bounded secondary metrics and HDBSCAN quality diagnostics."""

    non_noise = labels != -1
    X_valid = X[non_noise]
    labels_valid = labels[non_noise]

    unique_clusters, cluster_sizes = np.unique(labels_valid, return_counts=True)
    total_count = len(labels)
    cluster_count = len(unique_clusters)
    if cluster_count:
        largest_cluster_size = int(cluster_sizes.max())
        smallest_cluster_size = int(cluster_sizes.min())
        median_cluster_size = float(np.median(cluster_sizes))
        largest_cluster_fraction = largest_cluster_size / total_count
        clusters_below_one_percent = int(
            (cluster_sizes / total_count < 0.01).sum()
        )
    else:
        largest_cluster_size = 0
        smallest_cluster_size = 0
        median_cluster_size = np.nan
        largest_cluster_fraction = np.nan
        clusters_below_one_percent = 0

    result = {
        "cluster_count": int(cluster_count),
        "noise_count": int((~non_noise).sum()),
        "noise_fraction": float((~non_noise).mean()),
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": float(largest_cluster_fraction),
        "smallest_cluster_size": smallest_cluster_size,
        "median_cluster_size": median_cluster_size,
        "number_of_clusters_below_1_percent": clusters_below_one_percent,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "dbcv": np.nan,
        "dbcv_evaluation_sample_size": 0,
    }

    if validity_index is None:
        result["dbcv_status"] = "unavailable"

    if cluster_count < 2 or len(X_valid) < 3:
        return result

    rng = np.random.default_rng(RANDOM_STATE)
    dbcv_sample_size = min(len(X_valid), dbcv_sample_size)
    if len(X_valid) > dbcv_sample_size:
        dbcv_indices = rng.choice(
            len(X_valid),
            size=dbcv_sample_size,
            replace=False,
        )
        X_dbcv = X_valid[dbcv_indices]
        y_dbcv = labels_valid[dbcv_indices]
    else:
        X_dbcv, y_dbcv = X_valid, labels_valid

    result["dbcv_evaluation_sample_size"] = int(len(X_dbcv))

    if validity_index is not None:
        try:
            result["dbcv"] = float(
                validity_index(X_dbcv, y_dbcv, metric="euclidean")
            )
            result["dbcv_status"] = "available"
        except Exception as exc:
            result["dbcv"] = np.nan
            result["dbcv_status"] = f"unavailable: {exc}"

    secondary_sample_size = min(len(X_valid), secondary_metric_sample_size)
    if len(X_valid) > secondary_sample_size:
        secondary_indices = rng.choice(
            len(X_valid),
            size=secondary_sample_size,
            replace=False,
        )
        X_eval = X_valid[secondary_indices]
        y_eval = labels_valid[secondary_indices]
    else:
        X_eval, y_eval = X_valid, labels_valid

    if len(np.unique(y_eval)) < 2:
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            result["silhouette"] = float(silhouette_score(X_eval, y_eval))
        except Exception:
            pass

        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))
        except Exception:
            pass

        try:
            result["calinski_harabasz"] = float(
                calinski_harabasz_score(X_eval, y_eval)
            )
        except Exception:
            pass

    return result


# ============================================================================
# HDBSCAN FIT
# ============================================================================

def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> hdbscan.HDBSCAN:
    """Fit HDBSCAN with the project configuration."""

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=False,
        core_dist_n_jobs=1,
    )

    model.fit(
        X
    )

    return model


# ============================================================================
# TUNING
# ============================================================================

def rank_hdbscan_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates using DBCV, noise sanity, size quality, then silhouette/DB."""

    ranked = candidates.copy()
    ranked["dbcv_valid_rank"] = ranked["dbcv"].notna().astype(int)
    ranked["noise_sanity_rank"] = ranked["noise_fraction"].between(
        MIN_ACCEPTABLE_NOISE,
        MAX_ACCEPTABLE_NOISE,
    ).astype(int)
    ranked["cluster_size_quality_rank"] = (
        (~ranked["largest_cluster_over_70_percent"])
        & (~ranked["extreme_fragmentation"])
        & (ranked["smallest_cluster_size"] > 0)
        & (ranked["median_cluster_size"].notna())
    ).astype(int)
    ranked = ranked.sort_values(
        by=[
            "dbcv_valid_rank",
            "dbcv",
            "noise_sanity_rank",
            "cluster_size_quality_rank",
            "silhouette",
            "davies_bouldin",
            "min_cluster_size",
            "min_samples",
        ],
        ascending=[False, False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "candidate_rank", np.arange(1, len(ranked) + 1))
    return ranked


def tune_hdbscan(
    X: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune the required 12 configurations and return the top three."""

    results: list[dict] = []
    total_start = time.perf_counter()

    print("\nStarting HDBSCAN stage-one parameter tuning...")
    if validity_index is None:
        print(
            "WARNING: DBCV is unavailable because hdbscan.validity.validity_index "
            "could not be imported; DBCV will be recorded as unavailable."
        )

    total_runs = (
        len(MIN_CLUSTER_SIZE_VALUES)
        * len(MIN_SAMPLES_VALUES)
    )

    current = 0
    for min_cluster_size in MIN_CLUSTER_SIZE_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            current += 1
            run_start = time.perf_counter()
            print(
                f"[{current}/{total_runs}] "
                f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
            )

            try:
                model = fit_hdbscan(X, min_cluster_size, min_samples)
                labels = model.labels_
                metrics = calculate_cluster_metrics(X, labels)
                elapsed_seconds = time.perf_counter() - run_start
                metrics.update({
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "elapsed_seconds": elapsed_seconds,
                    "total_elapsed_seconds": time.perf_counter() - total_start,
                    "error": "",
                })
                results.append(metrics)
                del model
                del labels
                gc.collect()
            except Exception as exc:
                elapsed_seconds = time.perf_counter() - run_start
                results.append({
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_count": 0,
                    "noise_count": len(X),
                    "noise_fraction": 1.0,
                    "largest_cluster_size": 0,
                    "largest_cluster_fraction": np.nan,
                    "smallest_cluster_size": 0,
                    "median_cluster_size": np.nan,
                    "number_of_clusters_below_1_percent": 0,
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan,
                    "dbcv": np.nan,
                    "dbcv_evaluation_sample_size": 0,
                    "dbcv_status": "unavailable",
                    "elapsed_seconds": elapsed_seconds,
                    "total_elapsed_seconds": time.perf_counter() - total_start,
                    "error": str(exc),
                })
                gc.collect()

    results_df = pd.DataFrame(results)
    results_df = add_quality_flags(results_df)
    results_df.to_csv(TUNING_FILE, index=False)

    candidates = results_df[results_df["error"].eq("")].copy()
    candidates = candidates[candidates["cluster_count"] >= 2].copy()
    if candidates.empty:
        raise RuntimeError(
            "No successful HDBSCAN configuration produced at least two clusters."
        )

    ranked_candidates = rank_hdbscan_candidates(candidates)
    top3 = ranked_candidates.head(3).copy()
    top3.to_csv(TOP3_FILE, index=False)
    return results_df, top3


def add_quality_flags(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add noise, concentration, and fragmentation diagnostics."""

    result = results_df.copy()
    result["noise_over_40_percent"] = result["noise_fraction"] > MAX_ACCEPTABLE_NOISE
    result["largest_cluster_over_70_percent"] = (
        result["largest_cluster_fraction"] > 0.70
    )
    result["fewer_than_two_clusters"] = result["cluster_count"] < 2
    result["extreme_fragmentation"] = (
        (result["cluster_count"] >= 20)
        & (result["number_of_clusters_below_1_percent"] >= result["cluster_count"] * 0.75)
    )
    result["few_clusters"] = result["cluster_count"] < 2
    return result


# ============================================================================
# FULL FIT
# ============================================================================

def fit_full_model(
    X: np.ndarray,
    params: dict,
) -> hdbscan.HDBSCAN:
    """Fit the selected HDBSCAN configuration on all jobs."""

    print(
        "\nFitting selected HDBSCAN configuration on all "
        f"{len(X):,} eligible jobs..."
    )

    model = fit_hdbscan(
        X,
        params[
            "min_cluster_size"
        ],
        params[
            "min_samples"
        ],
    )

    return model


def add_persistence_metrics(
    metrics: dict[str, float],
    model: hdbscan.HDBSCAN,
) -> dict[str, float]:
    """Add native HDBSCAN membership and cluster persistence diagnostics."""

    persistence = np.asarray(
        getattr(model, "cluster_persistence_", np.array([], dtype=float)),
        dtype=float,
    )
    metrics["mean_cluster_membership_probability"] = float(
        np.mean(model.probabilities_)
    )
    metrics["minimum_cluster_persistence"] = (
        float(persistence.min()) if len(persistence) else np.nan
    )
    metrics["median_cluster_persistence"] = (
        float(np.median(persistence)) if len(persistence) else np.nan
    )
    metrics["maximum_cluster_persistence"] = (
        float(persistence.max()) if len(persistence) else np.nan
    )
    return metrics


def confirm_top3(
    X: np.ndarray,
    top3: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fit only the top three configurations on the full population."""

    confirmations: list[dict] = []
    models: dict[str, hdbscan.HDBSCAN] = {}

    for position, candidate in enumerate(top3.to_dict("records"), start=1):
        min_cluster_size = int(candidate["min_cluster_size"])
        min_samples = int(candidate["min_samples"])
        run_start = time.perf_counter()
        print(
            f"[{position}/3] full confirmation: "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
        )
        try:
            model = fit_hdbscan(X, min_cluster_size, min_samples)
            metrics = calculate_cluster_metrics(X, model.labels_)
            metrics.update({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "elapsed_seconds": time.perf_counter() - run_start,
                "dbcv_evaluation_sample_size": int(metrics.get("dbcv_evaluation_sample_size", 0)),
                "error": "",
            })
            confirmation = add_persistence_metrics(metrics, model)
            confirmations.append(confirmation)
            models[f"{min_cluster_size}|{min_samples}"] = model
            del model
            gc.collect()
        except Exception as exc:
            confirmations.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_count": 0,
                "noise_count": len(X),
                "noise_fraction": 1.0,
                "largest_cluster_size": 0,
                "largest_cluster_fraction": np.nan,
                "smallest_cluster_size": 0,
                "median_cluster_size": np.nan,
                "number_of_clusters_below_1_percent": 0,
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "dbcv": np.nan,
                "dbcv_evaluation_sample_size": 0,
                "dbcv_status": "unavailable",
                "mean_cluster_membership_probability": np.nan,
                "minimum_cluster_persistence": np.nan,
                "median_cluster_persistence": np.nan,
                "maximum_cluster_persistence": np.nan,
                "elapsed_seconds": time.perf_counter() - run_start,
                "error": str(exc),
            })
            gc.collect()

    confirmation_df = add_quality_flags(pd.DataFrame(confirmations))
    confirmation_df.to_csv(FULL_CONFIRMATION_FILE, index=False)
    models.clear()
    gc.collect()
    return confirmation_df, models


def select_final_configuration(
    confirmation_df: pd.DataFrame,
) -> dict[str, int]:
    """Select one reproducible final configuration from full confirmations."""

    candidates = confirmation_df[confirmation_df["error"].eq("")].copy()
    candidates = candidates[candidates["cluster_count"] >= 2].copy()
    if candidates.empty:
        raise RuntimeError("All top-three full-data confirmations failed.")

    candidates["dbcv_valid_rank"] = candidates["dbcv"].notna().astype(int)
    candidates["noise_sanity_rank"] = candidates["noise_fraction"].between(
        MIN_ACCEPTABLE_NOISE,
        MAX_ACCEPTABLE_NOISE,
    ).astype(int)
    candidates["cluster_size_quality_rank"] = (
        (~candidates["largest_cluster_over_70_percent"])
        & (~candidates["extreme_fragmentation"])
        & (candidates["smallest_cluster_size"] > 0)
        & (candidates["median_cluster_size"].notna())
    ).astype(int)
    candidates = candidates.sort_values(
        by=[
            "dbcv_valid_rank",
            "dbcv",
            "noise_sanity_rank",
            "cluster_size_quality_rank",
            "silhouette",
            "davies_bouldin",
            "min_cluster_size",
            "min_samples",
        ],
        ascending=[False, False, False, False, False, True, True, True],
        na_position="last",
    )
    selected = candidates.iloc[0]
    return {
        "min_cluster_size": int(selected["min_cluster_size"]),
        "min_samples": int(selected["min_samples"]),
    }


# ============================================================================
# ASSIGNMENTS
# ============================================================================

def save_assignments(
    model: hdbscan.HDBSCAN,
    n_rows: int,
) -> pd.DataFrame:
    """Save job-level HDBSCAN assignments."""

    labels = model.labels_
    probabilities = model.probabilities_

    assignments = pd.DataFrame(
        {
            "row_index": np.arange(
                n_rows
            ),
            "cluster_id": labels,
            "cluster_probability": probabilities,
        }
    )

    assignments.to_csv(
        ASSIGNMENT_FILE,
        index=False,
    )

    return assignments


# ============================================================================
# CLUSTER SUMMARY
# ============================================================================

def build_cluster_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build cluster-size and confidence summary."""

    total = len(
        labels
    )

    rows = []

    unique_labels = sorted(
        np.unique(
            labels
        )
    )

    for cluster_id in unique_labels:

        mask = (
            labels
            == cluster_id
        )

        count = int(
            mask.sum()
        )

        if cluster_id == -1:

            cluster_name = "NOISE"

        else:

            cluster_name = (
                f"CLUSTER_{cluster_id}"
            )

        rows.append(
            {
                "cluster_id": int(
                    cluster_id
                ),
                "cluster_name": cluster_name,
                "job_count": count,
                "percentage": (
                    count
                    / total
                    * 100
                ),
                "mean_probability": float(
                    probabilities[
                        mask
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the complete Objective-1 HDBSCAN stage."""

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — HDBSCAN WORKLOAD CHARACTERIZATION"
    )
    print("=" * 80)

    ensure_directories()

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    print("\nLoading final HDBSCAN matrix...")
    matrix_df = load_hdbscan_matrix()
    X_full = matrix_df.to_numpy(dtype=np.float64)
    print(f"Eligible jobs : {len(X_full):,}")
    print(f"Features      : {X_full.shape[1]:,}")

    try:
        source_df = load_feature_engineered_data()
    except Exception as exc:
        print(f"WARNING: metadata sampling fallback ({exc}).")
        source_df = pd.DataFrame()

    print("\nCreating representative tuning sample...")
    tuning_df, sampling_method = make_tuning_sample(matrix_df, source_df)
    X_tuning = tuning_df.to_numpy(dtype=np.float64)
    print(f"Tuning sample : {len(X_tuning):,} ({sampling_method})")

    _, top3 = tune_hdbscan(X_tuning)
    print("\nTOP 3 STAGE-ONE CANDIDATES")
    print(top3.to_string(index=False))

    print("\nSTAGE 2 — FULL DATA CONFIRMATION")
    confirmation_df, _ = confirm_top3(X_full, top3)
    print(confirmation_df.to_string(index=False))
    final_params = select_final_configuration(confirmation_df)

    print("\nFINAL FULL-DATA REFIT")
    model = fit_full_model(X_full, final_params)
    joblib.dump(model, MODEL_FILE)
    save_assignments(model, len(X_full))

    full_metrics = calculate_cluster_metrics(X_full, model.labels_)
    full_metrics = add_persistence_metrics(full_metrics, model)
    full_metrics.update(final_params)
    pd.DataFrame([full_metrics]).to_csv(METRICS_FILE, index=False)

    summary = build_cluster_summary(model.labels_, model.probabilities_)
    summary.to_csv(SUMMARY_FILE, index=False)
    del model
    gc.collect()

    selected_metrics = pd.Series(full_metrics)
    print("\nFINAL HDBSCAN CONFIGURATION")
    print(f"min_cluster_size = {final_params['min_cluster_size']}")
    print(f"min_samples      = {final_params['min_samples']}")
    print(f"clusters          = {int(selected_metrics['cluster_count'])}")
    print(f"noise %           = {selected_metrics['noise_fraction'] * 100:.2f}%")
    print(f"DBCV              = {selected_metrics['dbcv']}")
    print(f"Silhouette        = {selected_metrics['silhouette']}")
    print(f"Davies-Bouldin    = {selected_metrics['davies_bouldin']}")
    print("\nSaved:")
    print(f"  Model             : {MODEL_FILE}")
    print(f"  Assignments       : {ASSIGNMENT_FILE}")
    print(f"  Stage 1 tuning    : {TUNING_FILE}")
    print(f"  Top 3 candidates  : {TOP3_FILE}")
    print(f"  Full confirmation : {FULL_CONFIRMATION_FILE}")
    print(f"  Metrics           : {METRICS_FILE}")
    print(f"  Summary           : {SUMMARY_FILE}")
    print("\nHDBSCAN STAGE STATUS:")
    print("- TUNING COMPLETED")
    print("- TOP 3 CONFIRMATION COMPLETED")
    print("- FINAL FULL-DATA MODEL COMPLETED")
    print(
        "- FINAL CONFIGURATION = "
        f"min_cluster_size={final_params['min_cluster_size']}, "
        f"min_samples={final_params['min_samples']}"
    )
    print("- READY FOR STABILITY ANALYSIS = YES")


if __name__ == "__main__":
    main()