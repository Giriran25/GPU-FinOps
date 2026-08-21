
"""Objective 1 HDBSCAN screening and Stage-2 tuning workflow."""

from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

try:
    from hdbscan.validity import validity_index
except ImportError:  # pragma: no cover
    validity_index = None

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ENGINEERED_FILE = (
    PROJECT_ROOT / "data/processed/gpu_finops_feature_engineered.csv"
)
MATRIX_FILE = PROJECT_ROOT / "data/processed/objective1_hdbscan_matrix.csv"

PHASE1_FILE = PROJECT_ROOT / "results/metrics/objective1_hdbscan_densification.csv"
PHASE1_SUMMARY_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_densification_summary.csv"
)
SAMPLE_INDEX_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_stage1_sample_indices.csv"
)

ALIGNMENT_AUDIT_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_stage2_alignment_audit.csv"
)
STAGE2_SAMPLE_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_stage2_sample.csv"
)
STAGE2_TUNING_FILE = (
    PROJECT_ROOT / "results/metrics/objective1_hdbscan_stage2_tuning.csv"
)
STAGE2_SUMMARY_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_stage2_summary.csv"
)
STAGE2_TOP_CANDIDATES_FILE = (
    PROJECT_ROOT / "results/tables/objective1_hdbscan_stage2_top_candidates.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42

TUNING_SAMPLE_SIZE = 75_000
STAGE2_SAMPLE_SIZE = 120_000
DBCV_EVAL_SAMPLE_SIZE = 2_000
SECONDARY_METRIC_SAMPLE_SIZE = 5_000
STAGE2_DBCV_CAP = 3_000
STAGE2_SECONDARY_CAP = 10_000

REGION_GRID = {
    "A": {
        "min_cluster_size": 3000,
        "min_samples": [20, 25, 40, 50, 60, 75, 85, 90, 100],
    },
    "B": {
        "min_cluster_size": 2000,
        "min_samples": [40, 50, 60, 65, 75, 85, 90, 100],
    },
    "C": {
        "min_cluster_size": [2500, 2750],
        "min_samples": [25, 50, 100],
    },
}

STAGE2_REGION_GRID = [
    {"region": "B", "min_cluster_size": 1800, "min_samples": 50},
    {"region": "B", "min_cluster_size": 1800, "min_samples": 60},
    {"region": "B", "min_cluster_size": 1800, "min_samples": 75},
    {"region": "B", "min_cluster_size": 1800, "min_samples": 90},
    {"region": "B", "min_cluster_size": 1800, "min_samples": 100},
    {"region": "B", "min_cluster_size": 2000, "min_samples": 50},
    {"region": "B", "min_cluster_size": 2000, "min_samples": 60},
    {"region": "B", "min_cluster_size": 2000, "min_samples": 75},
    {"region": "B", "min_cluster_size": 2000, "min_samples": 90},
    {"region": "B", "min_cluster_size": 2000, "min_samples": 100},
    {"region": "B", "min_cluster_size": 2200, "min_samples": 50},
    {"region": "B", "min_cluster_size": 2200, "min_samples": 60},
    {"region": "B", "min_cluster_size": 2200, "min_samples": 75},
    {"region": "B", "min_cluster_size": 2200, "min_samples": 90},
    {"region": "B", "min_cluster_size": 2200, "min_samples": 100},
    {"region": "A", "min_cluster_size": 3000, "min_samples": 90},
    {"region": "A", "min_cluster_size": 3000, "min_samples": 100},
]

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

CLUSTER_SELECTION_METHOD = "eom"


# =============================================================================
# I/O
# =============================================================================

def ensure_directories() -> None:
    paths = [
        PHASE1_FILE,
        PHASE1_SUMMARY_FILE,
        SAMPLE_INDEX_FILE,
        ALIGNMENT_AUDIT_FILE,
        STAGE2_SAMPLE_FILE,
        STAGE2_TUNING_FILE,
        STAGE2_SUMMARY_FILE,
        STAGE2_TOP_CANDIDATES_FILE,
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAMPLE HANDLING
# =============================================================================

def load_or_create_stage1_sample(matrix_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if SAMPLE_INDEX_FILE.exists():
        try:
            idx_df = pd.read_csv(SAMPLE_INDEX_FILE)
            if "row_index" in idx_df.columns and not idx_df.empty:
                sample_idx = idx_df["row_index"].astype(int).to_numpy()
                if len(sample_idx) == TUNING_SAMPLE_SIZE:
                    return (
                        matrix_df.iloc[sample_idx].to_numpy(dtype=np.float64),
                        "reused_stage1_random_sample",
                    )
        except Exception:
            pass

    sample_idx = matrix_df.sample(
        n=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).index.to_numpy(dtype=int)
    pd.DataFrame({"row_index": sample_idx}).to_csv(
        SAMPLE_INDEX_FILE,
        index=False,
    )
    return (
        matrix_df.iloc[sample_idx].to_numpy(dtype=np.float64),
        "reused_stage1_random_sample" if SAMPLE_INDEX_FILE.exists() else "newly_generated_random_sample",
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_hdbscan_matrix() -> pd.DataFrame:
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(f"HDBSCAN matrix not found: {MATRIX_FILE}")

    df = pd.read_csv(MATRIX_FILE, low_memory=False)
    if df.empty:
        raise ValueError("HDBSCAN matrix is empty.")
    if df.isna().any().any():
        raise ValueError("HDBSCAN matrix contains NaN values.")
    values = df.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("HDBSCAN matrix contains non-finite values.")
    return df


def load_feature_engineered_data() -> pd.DataFrame:
    if not FEATURE_ENGINEERED_FILE.exists():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: {FEATURE_ENGINEERED_FILE}"
        )

    df = pd.read_csv(FEATURE_ENGINEERED_FILE, low_memory=False)
    required = CORE_FEATURES + [
        "job_name",
        "job_status",
        "has_telemetry",
        "has_execution_timing",
        "has_gpu_request",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in feature-engineered dataset: {missing}")
    return df


# =============================================================================
# ELIGIBILITY ALIGNMENT
# =============================================================================

def build_eligible_population(df: pd.DataFrame) -> pd.DataFrame:
    core = df[CORE_FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_numeric = np.isfinite(core.to_numpy()).all(axis=1)
    complete_case = core.notna().all(axis=1) & valid_numeric
    eligible = df.loc[complete_case].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No eligible jobs remain after applying Objective-1 feature filters.")
    if "job_name" not in eligible.columns:
        raise ValueError("job_name column missing from feature-engineered data.")
    if eligible["job_name"].isna().any():
        raise ValueError("At least one eligible job_name is missing.")
    return eligible


def align_stage2_population(matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered_df = load_feature_engineered_data()
    eligible_df = build_eligible_population(engineered_df)

    aligned_df = eligible_df[["job_name", "job_status"]].copy().reset_index(drop=True)
    aligned_df["row_index"] = np.arange(len(aligned_df), dtype=int)

    expected_rows = len(matrix_df)
    aligned_rows = len(aligned_df)
    if aligned_rows != expected_rows:
        raise ValueError(
            f"Eligibility alignment failed: eligible population rows={aligned_rows}, "
            f"HDBSCAN matrix rows={expected_rows}."
        )

    if aligned_df["job_name"].duplicated().any():
        raise ValueError("Alignment failed: duplicate job_name values exist in the eligible population.")

    if aligned_df["job_name"].nunique() != aligned_rows:
        raise ValueError("Alignment failed: job_name is not unique in the eligible population.")

    aligned_df["job_name"] = aligned_df["job_name"].astype(str)
    alignment_audit = pd.DataFrame([
        {
            "total_source_jobs": int(len(engineered_df)),
            "eligible_jobs": int(aligned_rows),
            "aligned_jobs": int(len(matrix_df)),
            "unique_identifier_count": int(aligned_df["job_name"].nunique()),
            "duplicate_identifier_count": int(aligned_df["job_name"].duplicated().sum()),
            "alignment_status": "PASS",
        }
    ])
    alignment_audit.to_csv(ALIGNMENT_AUDIT_FILE, index=False)

    return aligned_df, alignment_audit


# =============================================================================
# HDBSCAN FIT
# =============================================================================

def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    prediction_data: bool = False,
) -> hdbscan.HDBSCAN:
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=prediction_data,
        core_dist_n_jobs=1,
    )
    model.fit(X)
    return model


# =============================================================================
# METRICS
# =============================================================================

def cluster_structure(labels: np.ndarray) -> dict[str, float | int]:
    non_noise = labels != -1
    labels_valid = labels[non_noise]
    total_count = len(labels)

    if len(labels_valid) == 0:
        return {
            "cluster_count": 0,
            "noise_count": int((~non_noise).sum()),
            "noise_fraction": float((~non_noise).mean()),
            "largest_cluster_size": 0,
            "largest_cluster_fraction": np.nan,
            "smallest_cluster_size": 0,
            "median_cluster_size": np.nan,
            "number_of_clusters_below_1_percent": 0,
        }

    unique_clusters, cluster_sizes = np.unique(labels_valid, return_counts=True)
    return {
        "cluster_count": int(len(unique_clusters)),
        "noise_count": int((~non_noise).sum()),
        "noise_fraction": float((~non_noise).mean()),
        "largest_cluster_size": int(cluster_sizes.max()),
        "largest_cluster_fraction": float(cluster_sizes.max() / total_count),
        "smallest_cluster_size": int(cluster_sizes.min()),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "number_of_clusters_below_1_percent": int(
            (cluster_sizes / total_count < 0.01).sum()
        ),
    }


def evaluate_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    dbcv_cap: int = DBCV_EVAL_SAMPLE_SIZE,
    secondary_cap: int = SECONDARY_METRIC_SAMPLE_SIZE,
) -> dict[str, float | int | str]:
    result = cluster_structure(labels)
    non_noise = labels != -1
    X_valid = X[non_noise]
    labels_valid = labels[non_noise]

    result.update({
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "dbcv": np.nan,
        "dbcv_evaluation_sample_size": 0,
        "secondary_metric_evaluation_sample_size": 0,
        "dbcv_status": "unavailable",
    })

    if len(np.unique(labels_valid)) < 2 or len(X_valid) < 3:
        return result

    rng = np.random.default_rng(RANDOM_STATE)

    dbcv_limit = min(len(X_valid), dbcv_cap)
    if len(X_valid) > dbcv_limit:
        idx = rng.choice(len(X_valid), size=dbcv_limit, replace=False)
        X_dbcv = X_valid[idx]
        y_dbcv = labels_valid[idx]
    else:
        X_dbcv = X_valid
        y_dbcv = labels_valid
    result["dbcv_evaluation_sample_size"] = int(len(X_dbcv))

    if validity_index is not None:
        try:
            result["dbcv"] = float(validity_index(X_dbcv, y_dbcv, metric="euclidean"))
            result["dbcv_status"] = "available"
        except Exception as exc:
            result["dbcv"] = np.nan
            result["dbcv_status"] = str(exc)

    secondary_limit = min(len(X_valid), secondary_cap)
    if len(X_valid) > secondary_limit:
        idx = rng.choice(len(X_valid), size=secondary_limit, replace=False)
        X_eval = X_valid[idx]
        y_eval = labels_valid[idx]
    else:
        X_eval = X_valid
        y_eval = labels_valid
    result["secondary_metric_evaluation_sample_size"] = int(len(X_eval))

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
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
        except Exception:
            pass

    return result


# =============================================================================
# PHASE 1 AND PHASE 2 ANALYSIS
# =============================================================================

def compute_neighbor_flags(phase1_df: pd.DataFrame) -> pd.DataFrame:
    out = phase1_df.copy()
    out["plateau_flag"] = "NO"
    out["spike_flag"] = "NO"
    out["comparison_status"] = "VALID"
    out["edge_config"] = False
    out["largest_cluster_over_60_percent"] = out["largest_cluster_fraction"] > 0.60
    out["largest_cluster_over_70_percent"] = out["largest_cluster_fraction"] > 0.70
    out["excessive_noise"] = out["noise_fraction"] > 0.30
    out["extreme_fragmentation"] = out["number_of_clusters_below_1_percent"] >= 3
    out["interpretability_status"] = "PASS"
    out.loc[
        out["excessive_noise"] | out["largest_cluster_over_70_percent"] | out["extreme_fragmentation"],
        "interpretability_status",
    ] = "REVIEW"
    out.loc[
        out["largest_cluster_over_60_percent"] & (~out["largest_cluster_over_70_percent"]),
        "interpretability_status",
    ] = "CAUTION"

    for region_name in ["A", "B", "C"]:
        region_df = out[out["region"] == region_name].copy()
        region_df = region_df.sort_values(["min_cluster_size", "min_samples"]).reset_index(drop=True)

        for _, row in region_df.iterrows():
            original_index = row.name
            same_mcs = region_df[region_df["min_cluster_size"] == row["min_cluster_size"]]
            same_mcs = same_mcs.sort_values("min_samples").reset_index(drop=True)
            position = same_mcs.index[same_mcs["min_samples"] == row["min_samples"]][0]

            neighbor_rows = []
            if position > 0:
                neighbor_rows.append(same_mcs.iloc[position - 1])
            if position < len(same_mcs) - 1:
                neighbor_rows.append(same_mcs.iloc[position + 1])

            if len(neighbor_rows) == 0:
                out.loc[original_index, "edge_config"] = True
                out.loc[original_index, "plateau_flag"] = "INDETERMINATE"
                out.loc[original_index, "spike_flag"] = "INDETERMINATE"
                out.loc[original_index, "comparison_status"] = "INDETERMINATE"
                continue

            valid_neighbors = []
            invalid_required = False
            for neighbor in neighbor_rows:
                if neighbor["error"] != "" or pd.isna(neighbor["dbcv"]):
                    invalid_required = True
                else:
                    valid_neighbors.append(neighbor)

            if invalid_required:
                out.loc[original_index, "edge_config"] = len(neighbor_rows) == 1
                out.loc[original_index, "plateau_flag"] = "INDETERMINATE"
                out.loc[original_index, "spike_flag"] = "INDETERMINATE"
                out.loc[original_index, "comparison_status"] = "INDETERMINATE"
                continue

            if len(valid_neighbors) == 1:
                out.loc[original_index, "edge_config"] = True
                out.loc[original_index, "comparison_status"] = "ONE_SIDED"
                neighbor = valid_neighbors[0]
                diff_cluster = abs(int(row["cluster_count"]) - int(neighbor["cluster_count"]))
                diff_noise = abs(float(row["noise_fraction"]) - float(neighbor["noise_fraction"]))
                diff_dbcv = abs(float(row["dbcv"]) - float(neighbor["dbcv"]))
                plateau_ok = (
                    diff_cluster <= 1
                    and diff_noise <= 0.05
                    and diff_dbcv <= 0.15
                )
                out.loc[original_index, "plateau_flag"] = "YES" if plateau_ok else "NO"

                current_dbcv = float(row["dbcv"])
                neighbor_dbcv = float(neighbor["dbcv"])
                if current_dbcv > neighbor_dbcv + 0.25 and (
                    diff_noise >= 0.10 or diff_dbcv >= 0.25
                ):
                    out.loc[original_index, "spike_flag"] = "YES"
                else:
                    out.loc[original_index, "spike_flag"] = "NO"
                continue

            out.loc[original_index, "edge_config"] = False
            out.loc[original_index, "comparison_status"] = "VALID"
            plateau_pass = True
            for neighbor in valid_neighbors:
                diff_cluster = abs(int(row["cluster_count"]) - int(neighbor["cluster_count"]))
                diff_noise = abs(float(row["noise_fraction"]) - float(neighbor["noise_fraction"]))
                diff_dbcv = abs(float(row["dbcv"]) - float(neighbor["dbcv"]))
                if not (
                    diff_cluster <= 1
                    and diff_noise <= 0.05
                    and diff_dbcv <= 0.15
                ):
                    plateau_pass = False
                    break
            out.loc[original_index, "plateau_flag"] = "YES" if plateau_pass else "NO"

            spike_yes = False
            for neighbor in valid_neighbors:
                diff_noise = abs(float(row["noise_fraction"]) - float(neighbor["noise_fraction"]))
                diff_dbcv = abs(float(row["dbcv"]) - float(neighbor["dbcv"]))
                current_dbcv = float(row["dbcv"])
                neighbor_dbcv = float(neighbor["dbcv"])
                if current_dbcv > neighbor_dbcv + 0.25 and (
                    diff_noise >= 0.10 or diff_dbcv >= 0.25
                ):
                    spike_yes = True
                    break
            out.loc[original_index, "spike_flag"] = "YES" if spike_yes else "NO"

    return out


def rank_stable_candidates(summary_df: pd.DataFrame, region_filter: list[str]) -> pd.DataFrame:
    subset = summary_df[summary_df["region"].isin(region_filter)].copy()
    if subset.empty:
        return subset

    subset["plateau_yes"] = subset["plateau_flag"].eq("YES").astype(int)
    subset["interpretability_pass"] = subset["interpretability_status"].eq("PASS").astype(int)
    subset["reasonable_noise"] = subset["noise_fraction"].le(0.30).astype(int)
    subset["no_mega_cluster_warning"] = (~subset["largest_cluster_over_60_percent"]).astype(int)
    subset["no_fragmentation"] = (~subset["extreme_fragmentation"]).astype(int)
    subset = subset.sort_values(
        by=[
            "plateau_yes",
            "interpretability_pass",
            "reasonable_noise",
            "no_mega_cluster_warning",
            "no_fragmentation",
            "dbcv",
            "silhouette",
            "davies_bouldin",
        ],
        ascending=[False, False, False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return subset


# =============================================================================
# STAGE 1 RUN
# =============================================================================

def run_stage1_summary() -> pd.DataFrame:
    matrix_df = load_hdbscan_matrix()
    X_tuning, sample_source = load_or_create_stage1_sample(matrix_df)
    print(f"Sample source:\n- {sample_source}")
    print(f"Eligible jobs: {len(matrix_df):,}")
    print(f"Stage-1 sample: {len(X_tuning):,}")

    configs = []
    for region_name, region_cfg in REGION_GRID.items():
        cluster_values = region_cfg["min_cluster_size"]
        if not isinstance(cluster_values, list):
            cluster_values = [cluster_values]
        for min_cluster_size in cluster_values:
            for min_samples in region_cfg["min_samples"]:
                configs.append({
                    "region": region_name,
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                })

    phase1_rows = []
    print("\nPHASE 1 — RAW SWEEP")
    for idx, config in enumerate(configs, start=1):
        run_start = time.perf_counter()
        region_name = config["region"]
        min_cluster_size = config["min_cluster_size"]
        min_samples = config["min_samples"]

        print(
            f"[{idx}/{len(configs)}] region={region_name}, "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}",
            flush=True,
        )

        model = None
        labels = None
        try:
            model = fit_hdbscan(X_tuning, min_cluster_size, min_samples)
            labels = model.labels_
            metrics = evaluate_metrics(X_tuning, labels)
            row = {
                "region": region_name,
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "cluster_count": int(metrics.get("cluster_count", 0)),
                "noise_count": int(metrics.get("noise_count", 0)),
                "noise_fraction": float(metrics.get("noise_fraction", np.nan)),
                "largest_cluster_size": int(metrics.get("largest_cluster_size", 0)),
                "largest_cluster_fraction": float(metrics.get("largest_cluster_fraction", np.nan)),
                "smallest_cluster_size": int(metrics.get("smallest_cluster_size", 0)),
                "median_cluster_size": float(metrics.get("median_cluster_size", np.nan)),
                "number_of_clusters_below_1_percent": int(metrics.get("number_of_clusters_below_1_percent", 0)),
                "silhouette": metrics.get("silhouette"),
                "davies_bouldin": metrics.get("davies_bouldin"),
                "calinski_harabasz": metrics.get("calinski_harabasz"),
                "dbcv": metrics.get("dbcv"),
                "dbcv_evaluation_sample_size": int(metrics.get("dbcv_evaluation_sample_size", 0)),
                "secondary_metric_evaluation_sample_size": int(metrics.get("secondary_metric_evaluation_sample_size", 0)),
                "dbcv_status": metrics.get("dbcv_status", "available"),
                "elapsed_seconds": float(time.perf_counter() - run_start),
                "error": "",
            }
            phase1_rows.append(row)
            print(
                "  cluster_count="
                f"{row['cluster_count']}, noise%={row['noise_fraction'] * 100:.2f}, "
                f"DBCV={row['dbcv']}, elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            row = {
                "region": region_name,
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "cluster_count": 0,
                "noise_count": len(X_tuning),
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
                "secondary_metric_evaluation_sample_size": 0,
                "dbcv_status": str(exc),
                "elapsed_seconds": float(time.perf_counter() - run_start),
                "error": str(exc),
            }
            phase1_rows.append(row)
            print(
                "  cluster_count=0, noise%=100.00, DBCV=nan, "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )
        finally:
            del model
            del labels
            gc.collect()

    phase1_df = pd.DataFrame(phase1_rows)
    phase1_df.to_csv(PHASE1_FILE, index=False)

    print("\nPHASE 2 — NEIGHBOR ANALYSIS")
    summary_df = compute_neighbor_flags(phase1_df)
    summary_df.to_csv(PHASE1_SUMMARY_FILE, index=False)

    return summary_df


# =============================================================================
# STAGE 2 SAMPLE & GRID
# =============================================================================

def create_stage2_sample(aligned_df: pd.DataFrame) -> pd.DataFrame:
    eligible = aligned_df.copy().reset_index(drop=True)
    eligible["gpu_demand_bucket"] = pd.qcut(
        pd.to_numeric(eligible["job_status"].map({}), errors="coerce"),
        q=5,
        labels=False,
        duplicates="drop",
    )
    eligible["gpu_demand_bucket"] = pd.qcut(
        pd.to_numeric(eligible["job_status"].map({}), errors="coerce"),
        q=5,
        labels=False,
        duplicates="drop",
    )
    return eligible


def stage2_sampling_frame(matrix_df: pd.DataFrame, aligned_df: pd.DataFrame) -> pd.DataFrame:
    stage2_df = aligned_df[["row_index", "job_name", "job_status"]].copy().reset_index(drop=True)
    stage2_df["gpu_demand_scale"] = pd.to_numeric(
        matrix_df.iloc[stage2_df["row_index"].to_numpy()]["gpu_demand_scale"],
        errors="coerce",
    )
    stage2_df["runtime_log"] = pd.to_numeric(
        matrix_df.iloc[stage2_df["row_index"].to_numpy()]["runtime_log"],
        errors="coerce",
    )
    stage2_df["gpu_demand_bucket"] = pd.qcut(
        stage2_df["gpu_demand_scale"],
        q=5,
        labels=False,
        duplicates="drop",
    )
    stage2_df["runtime_bucket"] = pd.qcut(
        stage2_df["runtime_log"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    if stage2_df["gpu_demand_bucket"].isna().any() or stage2_df["runtime_bucket"].isna().any():
        stage2_df["gpu_demand_bucket"] = stage2_df["gpu_demand_bucket"].fillna(
            stage2_df["gpu_demand_scale"].rank(method="first").astype(int) % 5
        )
        stage2_df["runtime_bucket"] = stage2_df["runtime_bucket"].fillna(
            stage2_df["runtime_log"].rank(method="first").astype(int) % 5
        )

    strata = (
        stage2_df[["job_status", "gpu_demand_bucket", "runtime_bucket"]]
        .astype(str)
        .agg(lambda s: s.str.cat(sep="|"), axis=1)
    )
    sample = stage2_df.sample(
        n=STAGE2_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    ).sort_values("row_index").reset_index(drop=True)

    if len(sample) != STAGE2_SAMPLE_SIZE:
        raise ValueError(f"Stage-2 tuning sample size mismatch: expected {STAGE2_SAMPLE_SIZE}, got {len(sample)}")

    sample = sample[["row_index", "job_name", "job_status", "gpu_demand_bucket", "runtime_bucket"]].copy()
    sample.to_csv(STAGE2_SAMPLE_FILE, index=False)
    return sample


# =============================================================================
# STAGE 2 TUNING
# =============================================================================

def stage2_config_summary() -> pd.DataFrame:
    matrix_df = load_hdbscan_matrix()
    aligned_df, audit = align_stage2_population(matrix_df)
    if audit.iloc[0]["alignment_status"] != "PASS":
        print("STAGE 2 ALIGNMENT STATUS: FAIL")
        raise RuntimeError("Stage-2 alignment failed. No Stage-2 tuning run allowed.")
    print("STAGE 2 ALIGNMENT STATUS: PASS")

    sample_df = stage2_sampling_frame(matrix_df, aligned_df)
    print(f"STAGE 2 SAMPLE SIZE: {len(sample_df):,}")
    print("STAGE 2 SAMPLING METHOD: stratified")
    print("STAGE 2 SAMPLING METHOD: stratified diagnostic tuning sample")

    sample_index = sample_df["row_index"].to_numpy(dtype=int)
    X_stage2 = matrix_df.iloc[sample_index].to_numpy(dtype=np.float64)
    rows = []

    for config in STAGE2_REGION_GRID:
        region = config["region"]
        mcs = int(config["min_cluster_size"])
        ms = int(config["min_samples"])
        run_start = time.perf_counter()
        model = None
        labels = None
        try:
            model = fit_hdbscan(X_stage2, mcs, ms)
            labels = model.labels_
            metrics = evaluate_metrics(X_stage2, labels, dbcv_cap=STAGE2_DBCV_CAP, secondary_cap=STAGE2_SECONDARY_CAP)
            row = {
                "region": region,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "cluster_count": int(metrics.get("cluster_count", 0)),
                "noise_count": int(metrics.get("noise_count", 0)),
                "noise_fraction": float(metrics.get("noise_fraction", np.nan)),
                "largest_cluster_size": int(metrics.get("largest_cluster_size", 0)),
                "largest_cluster_fraction": float(metrics.get("largest_cluster_fraction", np.nan)),
                "smallest_cluster_size": int(metrics.get("smallest_cluster_size", 0)),
                "median_cluster_size": float(metrics.get("median_cluster_size", np.nan)),
                "number_of_clusters_below_1_percent": int(metrics.get("number_of_clusters_below_1_percent", 0)),
                "silhouette": metrics.get("silhouette"),
                "davies_bouldin": metrics.get("davies_bouldin"),
                "calinski_harabasz": metrics.get("calinski_harabasz"),
                "dbcv": metrics.get("dbcv"),
                "dbcv_evaluation_sample_size": int(metrics.get("dbcv_evaluation_sample_size", 0)),
                "secondary_metric_evaluation_sample_size": int(metrics.get("secondary_metric_evaluation_sample_size", 0)),
                "dbcv_status": metrics.get("dbcv_status", "available"),
                "elapsed_seconds": float(time.perf_counter() - run_start),
                "error": "",
            }
            rows.append(row)
        except Exception as exc:
            rows.append({
                "region": region,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "cluster_count": 0,
                "noise_count": len(X_stage2),
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
                "secondary_metric_evaluation_sample_size": 0,
                "dbcv_status": str(exc),
                "elapsed_seconds": float(time.perf_counter() - run_start),
                "error": str(exc),
            })
        finally:
            del model
            del labels
            gc.collect()

    stage2_df = pd.DataFrame(rows)
    stage2_df.to_csv(STAGE2_TUNING_FILE, index=False)

    summary_df = compute_neighbor_flags(stage2_df)
    summary_df["largest_cluster_over_60_percent"] = summary_df["largest_cluster_fraction"] > 0.60
    summary_df["largest_cluster_over_70_percent"] = summary_df["largest_cluster_fraction"] > 0.70
    summary_df["excessive_noise"] = summary_df["noise_fraction"] > 0.30
    summary_df["extreme_fragmentation"] = summary_df["number_of_clusters_below_1_percent"] >= 3
    summary_df["interpretability_status"] = "PASS"
    summary_df.loc[
        summary_df["excessive_noise"] | summary_df["largest_cluster_over_70_percent"] | summary_df["extreme_fragmentation"],
        "interpretability_status",
    ] = "REVIEW"
    summary_df.loc[
        summary_df["largest_cluster_over_60_percent"] & (~summary_df["largest_cluster_over_70_percent"]),
        "interpretability_status",
    ] = "CAUTION"
    summary_df.to_csv(STAGE2_SUMMARY_FILE, index=False)

    ranked = summary_df.copy()
    ranked["plateau_rank"] = ranked["plateau_flag"].eq("YES").astype(int)
    ranked["reasonable_noise"] = ranked["noise_fraction"].le(0.30).astype(int)
    ranked["pass_rank"] = ranked["interpretability_status"].eq("PASS").astype(int)
    ranked["mega_ok"] = (~ranked["largest_cluster_over_70_percent"]).astype(int)
    ranked = ranked.sort_values(
        by=[
            "plateau_rank",
            "reasonable_noise",
            "pass_rank",
            "mega_ok",
            "dbcv",
            "silhouette",
            "davies_bouldin",
        ],
        ascending=[False, False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    top_candidates = ranked[
        [
            "rank",
            "region",
            "min_cluster_size",
            "min_samples",
            "cluster_count",
            "noise_fraction",
            "dbcv",
            "silhouette",
            "davies_bouldin",
            "largest_cluster_fraction",
            "plateau_flag",
            "spike_flag",
            "interpretability_status",
        ]
    ].copy()
    top_candidates.to_csv(STAGE2_TOP_CANDIDATES_FILE, index=False)
    return top_candidates.head(3)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("GPU-FinOps | OBJECTIVE 1 — HDBSCAN DENSIFICATION ANALYSIS")
    print("=" * 80)

    ensure_directories()

    if PHASE1_FILE.exists() and PHASE1_SUMMARY_FILE.exists():
        print("Using existing Stage-1 densification outputs.")
        stage1_summary = pd.read_csv(PHASE1_SUMMARY_FILE)
    else:
        stage1_summary = run_stage1_summary()

    stable_coarse = rank_stable_candidates(stage1_summary, ["A", "C"])
    stable_moderate = rank_stable_candidates(stage1_summary, ["B", "C"])
    spike_count = int(stage1_summary["spike_flag"].eq("YES").sum())
    indeterminate_count = int(stage1_summary["comparison_status"].eq("INDETERMINATE").sum())

    print("\n============================================================")
    print("STAGE 1 DENSIFICATION COMPLETED")
    print("============================================================")
    print(f"Total configurations: {len(stage1_summary)}")
    print(f"Stable coarse candidates: {len(stable_coarse)}")
    print(f"Stable moderate candidates: {len(stable_moderate)}")
    print(f"Spike candidates: {spike_count}")
    print(f"Indeterminate comparisons: {indeterminate_count}")
    print("Region B is PRIMARY; Region A is SECONDARY only.")
    print("No final cluster count was chosen; no later-stage confirmation was run.")

    print("\nTOP STABLE COARSE CANDIDATES")
    print(
        stable_coarse[
            [
                "min_cluster_size",
                "min_samples",
                "cluster_count",
                "noise_fraction",
                "dbcv",
                "silhouette",
                "largest_cluster_fraction",
                "plateau_flag",
                "spike_flag",
                "interpretability_status",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTOP STABLE MODERATE CANDIDATES")
    print(
        stable_moderate[
            [
                "min_cluster_size",
                "min_samples",
                "cluster_count",
                "noise_fraction",
                "dbcv",
                "silhouette",
                "largest_cluster_fraction",
                "plateau_flag",
                "spike_flag",
                "interpretability_status",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSTOP: Stage 1 densification complete; no later-stage fits were performed.")
    return


if __name__ == "__main__":
    main()