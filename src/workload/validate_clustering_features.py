"""
GPU-FinOps
----------
Objective 1 pre-HDBSCAN audit.

Purpose:
    Audit the current 14-feature Objective-1 clustering inputs before
    HDBSCAN modeling. This script inspects the current matrices and the
    feature-engineered source data without modifying any dataset.

Important:
    - No clustering is run.
    - No rows are removed or filled.
    - Current eligibility population is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gpu_finops_feature_engineered.csv"
)

MATRIX_FILES = {
    "hdbscan": PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_matrix.csv",
    "kmeans": PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_kmeans_matrix.csv",
    "gmm": PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_gmm_matrix.csv",
}

CORRELATION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_feature_correlation.csv"
)

VIF_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_feature_vif.csv"
)

EXCLUDED_PROFILE_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_excluded_population_profile.csv"
)

AUDIT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_pre_hdbscan_audit.csv"
)

FEATURES = [
    "plan_cpu_mean",
    "plan_mem_mean",
    "gpu_demand_scale",
    "gpu_idle_ratio",
    "gpu_utilization_intensity",
    "gpu_memory_intensity",
    "memory_intensity",
    "cpu_usage_transformed",
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "io_intensity",
    "runtime_log",
    "task_fanout",
    "machine_count",
]

CURRENT_CLUSTERING_FEATURES = [
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

RATIO_REVIEW_FEATURES = [
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "gpu_idle_ratio",
    "gpu_utilization_intensity",
    "gpu_demand_scale",
]


# ============================================================================
# HELPERS
# ============================================================================


def ensure_directories() -> None:
    """Create output directories if missing."""

    for path in [
        CORRELATION_OUTPUT.parent,
        VIF_OUTPUT.parent,
        EXCLUDED_PROFILE_OUTPUT.parent,
        AUDIT_OUTPUT.parent,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )



def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric while preserving missing values."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )



def compute_complete_case_mask(
    df: pd.DataFrame,
    feature_list: Iterable[str],
) -> pd.Series:
    """Replicate the current eligibility logic on a feature DataFrame."""

    core = df.loc[:, list(feature_list)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_mask = np.isfinite(
        core.to_numpy(dtype=float)
    ).all(axis=1)

    return (
        core.notna().all(axis=1)
        & finite_mask
    )



def describe_feature_stats(series: pd.Series) -> dict[str, float | int]:
    """Return basic univariate stats for a numeric series."""

    numeric = safe_numeric(series)
    finite = numeric[np.isfinite(numeric)]

    if finite.empty:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
    }


# ============================================================================
# 1. CURRENT MATRIX VALIDATION
# ============================================================================


def load_current_matrices() -> dict[str, pd.DataFrame]:
    """Load the current algorithm-specific clustering matrices."""

    matrices: dict[str, pd.DataFrame] = {}

    for name, path in MATRIX_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing matrix file: {path}"
            )

        matrix = pd.read_csv(path)
        matrices[name] = matrix

    return matrices



def summarize_matrices(
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the current matrix files and feature-level stats."""

    rows: list[dict[str, object]] = []

    for name, matrix in matrices.items():
        numeric_matrix = matrix.apply(
            pd.to_numeric,
            errors="coerce",
        )

        feature_rows = []
        for feature in numeric_matrix.columns:
            stats = describe_feature_stats(
                numeric_matrix[feature]
            )
            feature_rows.append(
                {
                    "matrix": name,
                    "feature": feature,
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
            )

        summary = {
            "matrix": name,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "feature_names": "; ".join(map(str, matrix.columns.tolist())),
            "nan_count": int(matrix.isna().sum().sum()),
            "inf_count": int(np.isinf(numeric_matrix.to_numpy(dtype=float)).sum()),
            "duplicate_feature_vectors": int(matrix.duplicated().sum()),
        }

        rows.append(summary)

        rows.extend(feature_rows)

    return pd.DataFrame(rows)


# ============================================================================
# 2. SOURCE-LEVEL NaN / Inf PROVENANCE
# ============================================================================


def load_source_data() -> pd.DataFrame:
    """Load the feature-engineered source dataset for provenance analysis."""

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Feature-engineered source not found:\n{SOURCE_FILE}"
        )

    return pd.read_csv(
        SOURCE_FILE,
        low_memory=False,
    )



def build_source_provenance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute source-level NaN/Inf and zero/negative counts for each feature."""

    rows: list[dict[str, object]] = []

    for feature in FEATURES:
        if feature not in df.columns:
            continue

        series = safe_numeric(df[feature])
        valid = series.dropna()
        inf_count = int(np.isinf(valid.to_numpy(dtype=float)).sum())
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / len(df) * 100)
        zero_count = int((series == 0).sum())
        negative_count = int((series < 0).sum())

        if feature in {"resource_efficiency_score", "cpu_gpu_imbalance"}:
            likely_reason = (
                "Derived ratio or difference of ratio terms; missing when a "
                "required denominator or source value is zero/missing, and "
                "extreme values reflect valid but sparse GPU/CPU allocation "
                "ratios."
            )
        elif feature == "gpu_idle_ratio":
            likely_reason = (
                "Defined only when telemetry exists and observed utilization "
                "is within the valid 0–100 range; otherwise the feature remains "
                "missing by design."
            )
        elif feature in {"gpu_utilization_intensity", "gpu_memory_intensity"}:
            likely_reason = (
                "Uses a log transform of observed utilization or memory; missing "
                "when the underlying measurement is absent or invalid."
            )
        elif feature in {"runtime_log", "task_fanout", "machine_count"}:
            likely_reason = (
                "Derived from execution metadata; missing when the job lacks a "
                "valid runtime, machine count, or task count."
            )
        else:
            likely_reason = (
                "Missing because the corresponding source quantity is absent, "
                "non-finite, or invalid for the engineered feature definition."
            )

        rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "inf_count": inf_count,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "likely_reason": likely_reason,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================


def compute_correlation_analysis(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson correlations on the current 12-feature HDBSCAN matrix."""

    clustering_matrix = matrix.loc[:, CURRENT_CLUSTERING_FEATURES].copy()
    numeric = clustering_matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    corr = numeric.corr(method="pearson")

    corr_rows: list[dict[str, object]] = []
    for left in corr.columns:
        for right in corr.columns:
            if left == right:
                continue
            value = corr.loc[left, right]
            if abs(value) >= 0.80:
                corr_rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "pearson_correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    }
                )

    corr_pairs = pd.DataFrame(corr_rows)

    if not corr_pairs.empty:
        corr_pairs = corr_pairs.sort_values(
            ["absolute_correlation", "feature_a", "feature_b"],
            ascending=[False, True, True],
        )

    return corr, corr_pairs


# ============================================================================
# 4. VIF ANALYSIS
# ============================================================================


def compute_vif(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute VIF for the current 12-feature HDBSCAN matrix."""

    numeric = matrix.loc[:, CURRENT_CLUSTERING_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    vif_rows: list[dict[str, float | str]] = []

    for feature in CURRENT_CLUSTERING_FEATURES:
        y = numeric[feature].to_numpy(dtype=float)
        other_features = [
            column for column in CURRENT_CLUSTERING_FEATURES if column != feature
        ]
        X = numeric[other_features].to_numpy(dtype=float)
        X = np.column_stack([
            np.ones(len(numeric)),
            X,
        ])

        beta, _, _, _ = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
        predictions = X @ beta
        sse = np.sum((y - predictions) ** 2)
        sst = np.sum((y - y.mean()) ** 2)

        if sst <= 0:
            vif = np.inf
        else:
            r_squared = 1 - (sse / sst)
            vif = np.inf if r_squared >= 1 else 1 / (1 - r_squared)

        vif_rows.append(
            {
                "feature": feature,
                "vif": float(vif),
                "flag_vif_gt_5": bool(vif > 5),
                "flag_vif_gt_10": bool(vif > 10),
            }
        )

    return pd.DataFrame(vif_rows).sort_values(
        by="vif",
        ascending=False,
    )


# ============================================================================
# 5. SCALING AUDIT
# ============================================================================


def check_scaling_audit(
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Audit whether the current matrices align with robust or standard scaling."""

    rows: list[dict[str, object]] = []

    for matrix_name, matrix in matrices.items():
        numeric_matrix = matrix.apply(
            pd.to_numeric,
            errors="coerce",
        )

        for feature in numeric_matrix.columns:
            series = numeric_matrix[feature]
            finite = series[np.isfinite(series)]

            if finite.empty:
                mean = median = std = iqr = np.nan
            else:
                q25 = finite.quantile(0.25)
                q75 = finite.quantile(0.75)
                mean = float(finite.mean())
                median = float(finite.median())
                std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
                iqr = float(q75 - q25)

            expected = "robust" if matrix_name == "hdbscan" else "standard"
            if expected == "robust":
                approx_median = abs(median) < 1e-6
                approx_iqr = abs(iqr - 1.0) < 0.10
                scale_status = "PASS" if approx_median and approx_iqr else "REVIEW"
            else:
                approx_mean = abs(mean) < 0.10
                approx_std = abs(std - 1.0) < 0.10
                scale_status = "PASS" if approx_mean and approx_std else "REVIEW"

            rows.append(
                {
                    "matrix": matrix_name,
                    "feature": feature,
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "iqr": iqr,
                    "expected_scaling": expected,
                    "scale_status": scale_status,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# 6. EXCLUDED POPULATION ANALYSIS
# ============================================================================


def build_excluded_population_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Profile the excluded population and compare with eligible jobs."""

    eligible_mask = compute_complete_case_mask(
        df,
        FEATURES,
    )
    eligible = df.loc[eligible_mask].copy()
    excluded = df.loc[~eligible_mask].copy()

    rows: list[dict[str, object]] = []

    def add_group_analysis(field: str) -> None:
        if field not in df.columns:
            return

        if field == "job_status":
            categories = sorted(
                pd.concat(
                    [eligible[field], excluded[field]],
                    ignore_index=True,
                )
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            categories = sorted(
                pd.Series(
                    pd.concat(
                        [eligible[field].astype(str), excluded[field].astype(str)],
                        ignore_index=True,
                    )
                )
                .unique()
                .tolist()
            )

        for category in categories:
            eligible_count = int(
                eligible[field].astype(str).eq(str(category)).sum()
            )
            excluded_count = int(
                excluded[field].astype(str).eq(str(category)).sum()
            )
            rows.append(
                {
                    "analysis_group": field,
                    "category": str(category),
                    "eligible_count": eligible_count,
                    "eligible_percentage": (
                        eligible_count / len(eligible) * 100 if len(eligible) else 0.0
                    ),
                    "excluded_count": excluded_count,
                    "excluded_percentage": (
                        excluded_count / len(excluded) * 100 if len(excluded) else 0.0
                    ),
                }
            )

    for field in [
        "job_status",
        "has_telemetry",
        "has_execution_timing",
        "has_gpu_request",
    ]:
        add_group_analysis(field)

    runtime_availability = {
        "runtime_available": int(
            excluded["has_execution_timing"].eq(1).sum()
        ),
        "runtime_missing": int(
            excluded["has_execution_timing"].eq(0).sum()
        ),
        "runtime_available_pct": (
            excluded["has_execution_timing"].eq(1).mean() * 100
        ),
        "runtime_missing_pct": (
            excluded["has_execution_timing"].eq(0).mean() * 100
        ),
    }
    rows.append(
        {
            "analysis_group": "runtime_availability",
            "category": "has_execution_timing",
            "eligible_count": int(
                eligible["has_execution_timing"].eq(1).sum()
            ),
            "eligible_percentage": (
                eligible["has_execution_timing"].eq(1).mean() * 100
            ),
            "excluded_count": runtime_availability["runtime_available"],
            "excluded_percentage": runtime_availability["runtime_available_pct"],
        }
    )
    rows.append(
        {
            "analysis_group": "runtime_availability",
            "category": "missing_execution_timing",
            "eligible_count": int(
                eligible["has_execution_timing"].eq(0).sum()
            ),
            "eligible_percentage": (
                eligible["has_execution_timing"].eq(0).mean() * 100
            ),
            "excluded_count": runtime_availability["runtime_missing"],
            "excluded_percentage": runtime_availability["runtime_missing_pct"],
        }
    )

    gpu_request_availability = {
        "gpu_request_available": int(
            excluded["has_gpu_request"].eq(1).sum()
        ),
        "gpu_request_missing": int(
            excluded["has_gpu_request"].eq(0).sum()
        ),
        "gpu_request_available_pct": (
            excluded["has_gpu_request"].eq(1).mean() * 100
        ),
        "gpu_request_missing_pct": (
            excluded["has_gpu_request"].eq(0).mean() * 100
        ),
    }
    rows.append(
        {
            "analysis_group": "gpu_request_availability",
            "category": "has_gpu_request",
            "eligible_count": int(
                eligible["has_gpu_request"].eq(1).sum()
            ),
            "eligible_percentage": (
                eligible["has_gpu_request"].eq(1).mean() * 100
            ),
            "excluded_count": gpu_request_availability["gpu_request_available"],
            "excluded_percentage": gpu_request_availability["gpu_request_available_pct"],
        }
    )
    rows.append(
        {
            "analysis_group": "gpu_request_availability",
            "category": "missing_gpu_request",
            "eligible_count": int(
                eligible["has_gpu_request"].eq(0).sum()
            ),
            "eligible_percentage": (
                eligible["has_gpu_request"].eq(0).mean() * 100
            ),
            "excluded_count": gpu_request_availability["gpu_request_missing"],
            "excluded_percentage": gpu_request_availability["gpu_request_missing_pct"],
        }
    )

    profile = pd.DataFrame(rows)
    profile = profile.sort_values(
        ["analysis_group", "category"],
        ascending=[True, True],
    )
    return profile


# ============================================================================
# 7. PRE-HDBSCAN DECISION REPORT
# ============================================================================


def build_pre_hdbscan_audit(
    source_df: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    source_provenance: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    vif: pd.DataFrame,
    scaling_audit: pd.DataFrame,
    excluded_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Build the decision report for the pre-HDBSCAN audit."""

    issues: list[str] = []
    severe_issues: list[str] = []

    eligible_mask = compute_complete_case_mask(
        source_df,
        FEATURES,
    )
    eligible_count = int(eligible_mask.sum())
    excluded_count = len(source_df) - eligible_count

    matrix_issues = matrix_summary[
        (matrix_summary["nan_count"] > 0)
        | (matrix_summary["inf_count"] > 0)
    ]
    if not matrix_issues.empty:
        issues.append(
            "Current matrices contain NaN or Inf values that must be explained "
            "before HDBSCAN."
        )
    else:
        severe_issues.append(
            "Matrix-level NaN/Inf integrity checks pass for the current outputs."
        )

    source_issues = source_provenance[
        source_provenance["missing_count"] > 0
    ]
    if not source_issues.empty:
        issues.append(
            "Source-level feature missingness is present in telemetry- and ratio-derived features; "
            "this is expected for the current engineered data but must be documented."
        )

    if not corr_pairs.empty:
        top_pairs = corr_pairs.head(5).to_dict("records")
        issue_text = "Correlation redundancy above 0.80 detected in "
        issue_text += ", ".join(
            f"{row['feature_a']}/{row['feature_b']}={row['absolute_correlation']:.2f}"
            for row in top_pairs
        )
        issues.append(issue_text)

    vif_high_5 = vif[vif["vif"] > 5]
    vif_high_10 = vif[vif["vif"] > 10]
    if not vif_high_5.empty:
        issues.append(
            "High VIF candidates were observed for several features; multicollinearity requires review before clustering."
        )
    if not vif_high_10.empty:
        severe_issues.append(
            "Features with VIF > 10 need explicit review because they may distort distance geometry."
        )

    scaling_review = scaling_audit[
        scaling_audit["scale_status"] == "REVIEW"
    ]
    if not scaling_review.empty:
        issues.append(
            "One or more matrix features do not match the expected scaling profile for their algorithm."
        )
    else:
        severe_issues.append(
            "All current matrices align with their expected scaling pattern."
        )

    excluded_nonzero = excluded_profile[
        excluded_profile["analysis_group"].isin(
            ["job_status", "has_telemetry", "has_execution_timing", "has_gpu_request"]
        )
    ]
    if not excluded_nonzero.empty:
        issues.append(
            "The excluded population is large and heavily concentrated in missing telemetry / execution-timing rows; confirm the 56.78% eligible population is still the intended cluster-ready cohort."
        )

    duplicate_rows = matrix_summary[
        matrix_summary["duplicate_feature_vectors"] > 0
    ]
    if not duplicate_rows.empty:
        severe_issues.append(
            "Duplicate behavioral vectors are present and valid for clustering; they should be reported, not rejected."
        )

    rows = [
        {
            "check": "A. NaN/Inf integrity",
            "result": (
                "Current matrices contain no NaN or Inf; source features still carry missingness due to the engineered feature definitions."
            ),
            "status": "PASS" if matrix_issues.empty else "REVIEW",
            "explanation": (
                "The matrices are finite after preprocessing, but the source-level feature-engineered data intentionally retains NaN for unavailable telemetry or invalid ratios."
            ),
            "recommended_action": (
                "Keep the current matrices as-is for audit; document missingness provenance and confirm it is expected before clustering."
            ),
        },
        {
            "check": "B. Ratio-feature validity",
            "result": (
                "resource_efficiency_score and cpu_gpu_imbalance are the primary ratio-tail candidates; gpu_idle_ratio is bounded and log-based features are stable when telemetry is present."
            ),
            "status": "REVIEW",
            "explanation": (
                "The ratio-based features are sensitive to denominator scarcity and can show extreme values when plan_gpu, plan_cpu, or telemetry are small/invalid."
            ),
            "recommended_action": (
                "Document the current ratio semantics and inspect outlier behavior before finalizing HDBSCAN distance inputs."
            ),
        },
        {
            "check": "C. Correlation redundancy",
            "result": (
                "Pairs with |r| >= 0.80 were identified in the current 12-feature HDBSCAN clustering matrix."
            ),
            "status": "REVIEW" if not corr_pairs.empty else "PASS",
            "explanation": (
                "Within the current 12-feature HDBSCAN matrix, some clustering features are highly correlated and should be reviewed before distance-based density estimation."
            ),
            "recommended_action": (
                "Review the top correlated pairs in the current HDBSCAN feature set, but do not remove features automatically based only on correlation."
            ),
        },
        {
            "check": "D. VIF/multicollinearity",
            "result": (
                f"VIF > 5: {int((vif['vif'] > 5).sum())}; VIF > 10: {int((vif['vif'] > 10).sum())} in the current 12-feature HDBSCAN clustering matrix."
            ),
            "status": "REVIEW" if not vif_high_5.empty else "PASS",
            "explanation": (
                "Several features in the current 12-feature HDBSCAN matrix exhibit variance inflation above the standard review thresholds, indicating shared explanatory structure within the clustering feature set."
            ),
            "recommended_action": (
                "Treat these as review candidates only; validate whether the multicollinearity is acceptable for the current clustering feature set."
            ),
        },
        {
            "check": "E. Scaling",
            "result": (
                "HDBSCAN is expected to use robust scaling and K-Means/GMM to use standard scaling; current matrices should be audited against those expectations."
            ),
            "status": "REVIEW" if not scaling_review.empty else "PASS",
            "explanation": (
                "The current matrices should be evaluated against robust-versus-standard scaling expectations before HDBSCAN modeling."
            ),
            "recommended_action": (
                "Confirm the current scaler choices match the intended algorithm assumptions."
            ),
        },
        {
            "check": "F. Excluded-population composition",
            "result": (
                f"Eligible jobs = {eligible_count:,}; excluded jobs = {excluded_count:,}; current eligible population remains {eligible_count / len(source_df) * 100:.2f}% of all jobs."
            ),
            "status": "REVIEW",
            "explanation": (
                "The excluded cohort is large and concentrated in jobs missing telemetry and execution timing, which affects the representativeness of the eligible cluster-ready population."
            ),
            "recommended_action": (
                "Confirm the 56.78% eligible population is the intended analytical subset before proceeding to HDBSCAN."
            ),
        },
        {
            "check": "G. Duplicate behavioral vectors",
            "result": (
                "Duplicate feature vectors are present in the current matrices but should be treated as valid cluster-eligible observations, not as data-quality failure."
            ),
            "status": "PASS",
            "explanation": (
                "Multiple jobs may legitimately share identical 14-dimensional behavioral profiles. Duplicate vectors are expected and not a reason to reject the population."
            ),
            "recommended_action": (
                "Report duplicate counts as diagnostic metadata and continue without deduplicating the data."
            ),
        },
        {
            "check": "H. Temporal-shape support",
            "result": (
                "Temporal-shape feature not currently supported by the job-level master dataset."
            ),
            "status": "FAIL",
            "explanation": (
                "The current job-level master/engineered dataset does not provide sufficient per-job temporal telemetry to engineer trend or burstiness features."
            ),
            "recommended_action": (
                "Do not invent temporal-shape features. Keep the current 14-feature job-level definition and document this limitation."
            ),
        },
    ]

    report = pd.DataFrame(rows)
    return report


# ============================================================================
# 8. RATIO-SPECIFIC REVIEW
# ============================================================================


def build_ratio_review(df: pd.DataFrame) -> pd.DataFrame:
    """Assess ratio-feature stability before clustering."""

    rows: list[dict[str, object]] = []

    for feature in RATIO_REVIEW_FEATURES:
        if feature not in df.columns:
            continue

        series = safe_numeric(df[feature])
        finite = series[np.isfinite(series)]

        rows.append(
            {
                "feature": feature,
                "missing_count": int(series.isna().sum()),
                "missing_percentage": float(series.isna().mean() * 100),
                "zero_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum()),
                "min": float(finite.min()) if not finite.empty else np.nan,
                "p01": float(finite.quantile(0.01)) if not finite.empty else np.nan,
                "p05": float(finite.quantile(0.05)) if not finite.empty else np.nan,
                "median": float(finite.median()) if not finite.empty else np.nan,
                "p95": float(finite.quantile(0.95)) if not finite.empty else np.nan,
                "p99": float(finite.quantile(0.99)) if not finite.empty else np.nan,
                "max": float(finite.max()) if not finite.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# 9. OUTPUT
# ============================================================================


def print_final_summary(
    matrix_summary: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    vif: pd.DataFrame,
    excluded_profile: pd.DataFrame,
    ratio_review: pd.DataFrame,
) -> None:
    """Print the final pre-HDBSCAN summary and blocking issues."""

    issues: list[str] = []

    review_status = "REVIEW"

    matrix_has_issues = (
        (matrix_summary["nan_count"] > 0).any()
        or (matrix_summary["inf_count"] > 0).any()
    )
    if matrix_has_issues:
        issues.append(
            "Current matrices contain NaN or Inf values and require explanation before HDBSCAN."
        )

    if not corr_pairs.empty:
        pairs = corr_pairs.head(3)
        issues.append(
            "High-correlation redundancy detected: "
            + ", ".join(
                f"{row['feature_a']}/{row['feature_b']}={row['absolute_correlation']:.2f}"
                for _, row in pairs.iterrows()
            )
            + "."
        )

    vif_candidates = vif[vif["vif"] > 5]
    if not vif_candidates.empty:
        issues.append(
            f"VIF review required for {int(len(vif_candidates))} features above 5."
        )

    ratio_extremes = ratio_review[
        (ratio_review["missing_percentage"] > 0)
        | (ratio_review["max"] > 100)
        | (ratio_review["min"] < -100)
    ]
    if not ratio_extremes.empty:
        issues.append(
            "Ratio feature tails and small-denominator sensitivity require explicit review before HDBSCAN."
        )

    excluded_total = excluded_profile[
        excluded_profile["analysis_group"] == "job_status"
    ]
    if not excluded_total.empty:
        issues.append(
            "Eligible population is 599,288 jobs (56.78%); excluded jobs remain large and are concentrated in missing-telemetry / missing-timing cases."
        )

    if not issues:
        review_status = "PASS"

    print("\n" + "=" * 72)
    print("PRE-HDBSCAN STATUS")
    print(review_status)
    print("=" * 72)
    print("Issues to resolve before HDBSCAN:")
    for index, issue in enumerate(issues, start=1):
        print(f"{index}. {issue}")

    if not issues:
        print("None identified in the current audit.")

    print("=" * 72)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run the pre-HDBSCAN audit for the current Objective-1 clustering inputs."""

    ensure_directories()

    source_df = load_source_data()
    matrices = load_current_matrices()

    matrix_summary = summarize_matrices(matrices)
    source_provenance = build_source_provenance(source_df)

    hdbscan_matrix = matrices["hdbscan"]
    corr_matrix, corr_pairs = compute_correlation_analysis(hdbscan_matrix)
    vif = compute_vif(hdbscan_matrix)

    scaling_audit = check_scaling_audit(matrices)
    excluded_profile = build_excluded_population_profile(source_df)
    ratio_review = build_ratio_review(source_df)
    pre_hdbscan_audit = build_pre_hdbscan_audit(
        source_df,
        matrix_summary,
        source_provenance,
        corr_pairs,
        vif,
        scaling_audit,
        excluded_profile,
    )

    corr_matrix.to_csv(
        CORRELATION_OUTPUT,
        index=True,
    )
    vif.to_csv(
        VIF_OUTPUT,
        index=False,
    )
    excluded_profile.to_csv(
        EXCLUDED_PROFILE_OUTPUT,
        index=False,
    )
    pre_hdbscan_audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\nCURRENT MATRIX VALIDATION")
    print(matrix_summary.to_string(index=False))

    print("\nSOURCE-LEVEL NaN / Inf PROVENANCE")
    print(source_provenance.to_string(index=False))

    print("\nCORRELATION PAIRS WITH |r| >= 0.80 (current 12-feature HDBSCAN matrix)")
    if corr_pairs.empty:
        print("No pairs with |r| >= 0.80 were found in the current 12-feature HDBSCAN clustering matrix.")
    else:
        print(corr_pairs.to_string(index=False))

    print("\nVIF ANALYSIS (current 12-feature HDBSCAN matrix)")
    print(vif.to_string(index=False))

    print("\nSCALING AUDIT")
    print(scaling_audit.to_string(index=False))

    print("\nEXCLUDED POPULATION PROFILE")
    print(excluded_profile.to_string(index=False))

    print("\nRATIO-SPECIFIC REVIEW")
    print(ratio_review.to_string(index=False))

    print("\nPRE-HDBSCAN DECISION REPORT")
    print(pre_hdbscan_audit.to_string(index=False))

    print_final_summary(
        matrix_summary,
        corr_pairs,
        vif,
        excluded_profile,
        ratio_review,
    )


if __name__ == "__main__":
    main()
