"""
GPU-FinOps
----------
Stage 4: Objective 1 Feature Engineering

Base:
    Project Proposal

Evidence:
    Completed EDA on the validated master dataset

Feature decisions:
    Claude EDA interpretation

Purpose:
    Convert the validated job-level master dataset into a
    behavior-oriented feature set for Objective 1 workload
    characterization.

Objective 1 dimensions:
    1. Resource Requirements
    2. Utilization Behavior
    3. Execution Characteristics

IMPORTANT:
    - The original master dataset is never modified.
    - Raw telemetry/runtime missingness is preserved.
    - Missing telemetry is NOT interpreted as zero utilization.
    - No clustering is performed in this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gpu_finops_job_master.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gpu_finops_feature_engineered.csv"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "engineered_feature_audit.csv"
)

ELIGIBILITY_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_modeling_eligibility.csv"
)


# ============================================================================
# REQUIRED INPUT COLUMNS
# ============================================================================

REQUIRED_COLUMNS = [
    "job_name",
    "plan_cpu_mean",
    "plan_mem_mean",
    "plan_gpu_mean",
    "gpu_util_mean",
    "avg_gpu_mem_mean",
    "avg_mem_mean",
    "cpu_usage_mean",
    "total_read",
    "total_write",
    "instance_runtime_hours_sum",
    "task_runtime_hours_mean",
    "task_count",
    "machine_count",
    "has_telemetry",
    "has_gpu_request",
    "has_execution_timing",
    "telemetry_coverage",
]


# ============================================================================
# OBJECTIVE 1 FEATURE GROUPS
# ============================================================================

RESOURCE_FEATURES = [
    "plan_cpu_mean",
    "plan_mem_mean",
    "gpu_demand_scale",
]

UTILIZATION_FEATURES = [
    "gpu_idle_ratio",
    "gpu_utilization_intensity",
    "gpu_memory_intensity",
    "memory_intensity",
    "cpu_usage_transformed",
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "io_intensity",
]

EXECUTION_FEATURES = [
    "runtime_log",
    "task_fanout",
    "machine_count",
]

ENGINEERED_FEATURES = (
    RESOURCE_FEATURES
    + UTILIZATION_FEATURES
    + EXECUTION_FEATURES
    + ["gpu_active_flag"]
)


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create required output directories."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ELIGIBILITY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOAD MASTER DATASET
# ============================================================================

def load_master_dataset() -> pd.DataFrame:
    """Load and validate the current master dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from master dataset:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    return df


# ============================================================================
# SAFE NUMERIC HELPERS
# ============================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a Series to numeric safely."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_log1p(
    series: pd.Series,
) -> pd.Series:
    """
    Apply log1p safely.

    Negative values are treated as invalid and become NaN.
    """

    numeric = safe_numeric(series)

    numeric = numeric.mask(
        numeric < 0
    )

    return np.log1p(
        numeric
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safe element-wise ratio.

    Returns NaN when numerator/denominator are missing
    or denominator is zero.
    """

    numerator = safe_numeric(
        numerator
    )

    denominator = safe_numeric(
        denominator
    )

    valid = (
        numerator.notna()
        & denominator.notna()
        & denominator.ne(0)
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    result.loc[valid] = (
        numerator.loc[valid]
        / denominator.loc[valid]
    )

    return result


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Objective 1 behavioral features.

    Important:
        Ratios are calculated before standalone log transformations
        so that their interpretation remains explicit.
    """

    output = df.copy()

    # ========================================================================
    # RAW SOURCE VARIABLES
    # ========================================================================

    plan_cpu = safe_numeric(
        output["plan_cpu_mean"]
    )

    plan_mem = safe_numeric(
        output["plan_mem_mean"]
    )

    plan_gpu = safe_numeric(
        output["plan_gpu_mean"]
    )

    gpu_util = safe_numeric(
        output["gpu_util_mean"]
    )

    cpu_usage = safe_numeric(
        output["cpu_usage_mean"]
    )

    avg_gpu_mem = safe_numeric(
        output["avg_gpu_mem_mean"]
    )

    avg_mem = safe_numeric(
        output["avg_mem_mean"]
    )

    total_read = safe_numeric(
        output["total_read"]
    )

    total_write = safe_numeric(
        output["total_write"]
    )

    runtime = safe_numeric(
        output["instance_runtime_hours_sum"]
    )

    task_count = safe_numeric(
        output["task_count"]
    )

    machine_count = safe_numeric(
        output["machine_count"]
    )

    has_telemetry = (
        output["has_telemetry"]
        .eq(1)
    )

    # ========================================================================
    # 1. RESOURCE REQUIREMENTS
    # ========================================================================

    output["gpu_demand_scale"] = (
        safe_log1p(plan_gpu)
    )

    # ========================================================================
    # 2. GPU UTILIZATION
    # ========================================================================

    # IMPORTANT:
    #   1 = telemetry exists AND observed utilization > 0
    #   0 = telemetry exists AND observed utilization == 0
    #   NA = telemetry unavailable
    #
    # This preserves the distinction between:
    #   observed idleness
    # and
    #   unobserved utilization.
    output["gpu_active_flag"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="Int64",
    )

    observed_gpu_util = (
        has_telemetry
        & gpu_util.notna()
        & gpu_util.between(
            0,
            100,
            inclusive="both",
        )
    )

    output.loc[
        observed_gpu_util,
        "gpu_active_flag",
    ] = (
        gpu_util.loc[
            observed_gpu_util
        ]
        .gt(0)
        .astype("Int64")
    )

    # ------------------------------------------------------------------------
    # GPU idle ratio
    # ------------------------------------------------------------------------

    output["gpu_idle_ratio"] = np.nan

    output.loc[
        observed_gpu_util,
        "gpu_idle_ratio",
    ] = (
        1.0
        - (
            gpu_util.loc[
                observed_gpu_util
            ]
            / 100.0
        )
    )

    # ------------------------------------------------------------------------
    # GPU utilization intensity
    # ------------------------------------------------------------------------

    output["gpu_utilization_intensity"] = np.nan

    output.loc[
        observed_gpu_util,
        "gpu_utilization_intensity",
    ] = np.log1p(
        gpu_util.loc[
            observed_gpu_util
        ]
    )

    # ------------------------------------------------------------------------
    # GPU memory intensity
    # ------------------------------------------------------------------------

    output["gpu_memory_intensity"] = (
        safe_log1p(
            avg_gpu_mem
        )
    )

    # ------------------------------------------------------------------------
    # Host memory intensity
    # ------------------------------------------------------------------------

    output["memory_intensity"] = (
        safe_log1p(
            avg_mem
        )
    )

    # ------------------------------------------------------------------------
    # CPU usage transformed
    # ------------------------------------------------------------------------

    output["cpu_usage_transformed"] = (
        safe_log1p(
            cpu_usage
        )
    )

    # ========================================================================
    # 3. RESOURCE EFFICIENCY
    # ========================================================================

    # The source GPU-utilization metric is represented on a 0–100 scale.
    # The request value is also expressed in percentage-of-GPU units.
    #
    # Normalizing both by 100 makes the intended relationship explicit:
    #
    #     (gpu_util / 100) / (plan_gpu / 100)
    #
    # Algebraically this equals gpu_util / plan_gpu, but the normalization
    # makes the feature semantics explicit rather than silently assuming
    # comparable raw units.
    #
    # We intentionally DO NOT clip this feature here because the audit showed
    # values > 1 and the available project documents do not establish that
    # such values are invalid. We will inspect/treat the tail later in the
    # clustering-specific preprocessing stage.
    output["resource_efficiency_score"] = safe_ratio(
        gpu_util / 100.0,
        plan_gpu / 100.0,
    )

    # ========================================================================
    # 4. CPU-GPU IMBALANCE
    # ========================================================================

    # GPU side is normalized from its 0–100 utilization scale.
    gpu_utilization_ratio = safe_ratio(
        gpu_util / 100.0,
        plan_gpu / 100.0,
    )

    # CPU source semantics are retained without assuming an artificial
    # 0–1 percentage interpretation.
    cpu_utilization_ratio = safe_ratio(
        cpu_usage,
        plan_cpu,
    )

    # No clipping here.
    #
    # Positive value:
    #     comparatively more CPU-side utilization
    #
    # Negative value:
    #     comparatively more GPU-side utilization
    #
    # Exact treatment will be decided after the post-engineering audit.
    output["cpu_gpu_imbalance"] = (
        cpu_utilization_ratio
        - gpu_utilization_ratio
    )

    # ========================================================================
    # 5. I/O INTENSITY
    # ========================================================================

    total_io = (
        total_read
        + total_write
    )

    # One semantic definition only:
    #
    #     log1p(total I/O volume / execution runtime)
    #
    # If runtime is missing/zero, the intensity remains NaN.
    # We do NOT silently switch to raw I/O volume because that would give
    # the same feature different meanings across rows.
    valid_io_runtime = (
        total_io.notna()
        & total_io.ge(0)
        & runtime.notna()
        & runtime.gt(0)
    )

    output["io_intensity"] = np.nan

    output.loc[
        valid_io_runtime,
        "io_intensity",
    ] = np.log1p(
        (
            total_io.loc[
                valid_io_runtime
            ]
            / runtime.loc[
                valid_io_runtime
            ]
        ).clip(
            lower=0
        )
    )

    # ========================================================================
    # 6. EXECUTION CHARACTERISTICS
    # ========================================================================

    # Runtime
    output["runtime_log"] = (
        safe_log1p(
            runtime
        )
    )

    # Job parallelism / fan-out
    output["task_fanout"] = (
        safe_log1p(
            task_count
        )
    )

    # Machine footprint
    output["machine_count"] = (
        machine_count
    )

    return output


# ============================================================================
# MODELING ELIGIBILITY
# ============================================================================

def build_modeling_eligibility(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Record availability of Objective 1 inputs.

    This function does NOT remove rows.
    """

    behavioral_columns = [
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
        "plan_cpu_mean",
        "plan_mem_mean",
        "gpu_demand_scale",
    ]

    eligibility = {
        "total_jobs": len(df),
        "jobs_with_telemetry": int(
            df["has_telemetry"]
            .eq(1)
            .sum()
        ),
        "jobs_without_telemetry": int(
            df["has_telemetry"]
            .eq(0)
            .sum()
        ),
        "jobs_with_gpu_request": int(
            df["has_gpu_request"]
            .eq(1)
            .sum()
        ),
        "jobs_without_gpu_request": int(
            df["has_gpu_request"]
            .eq(0)
            .sum()
        ),
        "jobs_with_execution_timing": int(
            df["has_execution_timing"]
            .eq(1)
            .sum()
        ),
        "jobs_without_execution_timing": int(
            df["has_execution_timing"]
            .eq(0)
            .sum()
        ),
    }

    # Add missing counts for engineered/core features.
    for column in behavioral_columns:
        eligibility[
            f"missing_{column}"
        ] = int(
            df[column]
            .isna()
            .sum()
        )

    # Convert dictionary into one-row table.
    return pd.DataFrame(
        [eligibility]
    )


# ============================================================================
# FEATURE AUDIT
# ============================================================================

def build_feature_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a statistical audit for engineered features.
    """

    rows = []

    for column in ENGINEERED_FEATURES:

        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        non_null = series.dropna()

        inf_count = int(
            np.isinf(
                non_null
            ).sum()
        )

        finite_values = (
            non_null.loc[
                np.isfinite(
                    non_null
                )
            ]
        )

        if finite_values.empty:
            rows.append(
                {
                    "feature": column,
                    "dtype": str(series.dtype),
                    "count": 0,
                    "missing_count": int(
                        series.isna().sum()
                    ),
                    "missing_percentage": float(
                        series.isna().mean()
                        * 100
                    ),
                    "inf_count": inf_count,
                    "zero_count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                    "skewness": np.nan,
                    "negative_count": 0,
                }
            )
            continue

        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "count": int(
                    len(finite_values)
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_percentage": float(
                    series.isna().mean()
                    * 100
                ),
                "inf_count": inf_count,
                "zero_count": int(
                    finite_values.eq(0).sum()
                ),
                "mean": float(
                    finite_values.mean()
                ),
                "std": float(
                    finite_values.std()
                ),
                "min": float(
                    finite_values.min()
                ),
                "median": float(
                    finite_values.median()
                ),
                "max": float(
                    finite_values.max()
                ),
                "skewness": float(
                    finite_values.skew()
                ),
                "negative_count": int(
                    finite_values.lt(0).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_engineered_features(
    df: pd.DataFrame,
) -> None:
    """Validate mathematical sanity of engineered features."""

    problems: list[str] = []

    # ------------------------------------------------------------------------
    # Feature existence and infinite values
    # ------------------------------------------------------------------------

    for column in ENGINEERED_FEATURES:

        if column not in df.columns:
            problems.append(
                f"Missing engineered feature: {column}"
            )
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if np.isinf(
            values
        ).any():

            problems.append(
                f"Infinite values found: {column}"
            )

    # ------------------------------------------------------------------------
    # GPU idle ratio
    # ------------------------------------------------------------------------

    valid_idle = (
        df["gpu_idle_ratio"]
        .dropna()
    )

    if not valid_idle.between(
        0,
        1,
        inclusive="both",
    ).all():

        problems.append(
            "gpu_idle_ratio contains values outside [0,1]"
        )

    # ------------------------------------------------------------------------
    # GPU active flag
    # ------------------------------------------------------------------------

    valid_active = (
        df["gpu_active_flag"]
        .dropna()
        .astype(int)
    )

    if not set(
        valid_active.unique()
    ).issubset(
        {0, 1}
    ):

        problems.append(
            "gpu_active_flag contains values outside {0,1}"
        )

    # ------------------------------------------------------------------------
    # GPU utilization intensity
    # ------------------------------------------------------------------------

    valid_gpu_intensity = (
        df["gpu_utilization_intensity"]
        .dropna()
    )

    if (
        valid_gpu_intensity < 0
    ).any():

        problems.append(
            "gpu_utilization_intensity contains negative values"
        )

    # ------------------------------------------------------------------------
    # GPU demand scale
    # ------------------------------------------------------------------------

    valid_gpu_demand = (
        df["gpu_demand_scale"]
        .dropna()
    )

    if (
        valid_gpu_demand < 0
    ).any():

        problems.append(
            "gpu_demand_scale contains negative values"
        )

    # ------------------------------------------------------------------------
    # Runtime log
    # ------------------------------------------------------------------------

    valid_runtime = (
        df["runtime_log"]
        .dropna()
    )

    if (
        valid_runtime < 0
    ).any():

        problems.append(
            "runtime_log contains negative values"
        )

    # ------------------------------------------------------------------------
    # Resource efficiency
    # ------------------------------------------------------------------------

    valid_efficiency = (
        df["resource_efficiency_score"]
        .dropna()
    )

    if (
        valid_efficiency < 0
    ).any():

        problems.append(
            "resource_efficiency_score contains negative values"
        )

    # ------------------------------------------------------------------------
    # I/O intensity
    # ------------------------------------------------------------------------

    valid_io = (
        df["io_intensity"]
        .dropna()
    )

    if (
        valid_io < 0
    ).any():

        problems.append(
            "io_intensity contains negative values"
        )

    # ------------------------------------------------------------------------
    # Fail if anything is mathematically invalid
    # ------------------------------------------------------------------------

    if problems:
        raise ValueError(
            "\nFeature-engineering validation failed:\n"
            + "\n".join(
                f"  - {problem}"
                for problem in problems
            )
        )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    df: pd.DataFrame,
    audit: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> None:
    """Save all Objective-1 feature-engineering artifacts."""

    # Original master remains untouched.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    eligibility.to_csv(
        ELIGIBILITY_FILE,
        index=False,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run Objective-1 feature engineering."""

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — FEATURE ENGINEERING"
    )
    print("=" * 80)

    ensure_directories()

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    print(
        "\nLoading master dataset..."
    )

    df = load_master_dataset()

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    # ------------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------------

    print(
        "\nCreating Objective-1 behavioral features..."
    )

    df = create_engineered_features(
        df
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    print(
        "Validating engineered features..."
    )

    validate_engineered_features(
        df
    )

    # ------------------------------------------------------------------------
    # Eligibility report
    # ------------------------------------------------------------------------

    print(
        "Building modeling eligibility report..."
    )

    eligibility = (
        build_modeling_eligibility(
            df
        )
    )

    # ------------------------------------------------------------------------
    # Feature audit
    # ------------------------------------------------------------------------

    print(
        "Building engineered feature audit..."
    )

    audit = (
        build_feature_audit(
            df
        )
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    print(
        "Saving outputs..."
    )

    save_outputs(
        df=df,
        audit=audit,
        eligibility=eligibility,
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print(
        "\nFeature-engineered dataset:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\nFeature audit:"
    )

    print(
        f"  {AUDIT_FILE}"
    )

    print(
        "\nModeling eligibility:"
    )

    print(
        f"  {ELIGIBILITY_FILE}"
    )

    print(
        "\nEngineered Objective-1 features:"
    )

    for feature in (
        RESOURCE_FEATURES
        + UTILIZATION_FEATURES
        + EXECUTION_FEATURES
        + ["gpu_active_flag"]
    ):
        print(
            f"  - {feature}"
        )

    print(
        "\nNext stage:"
    )

    print(
        "  Feature selection"
        " → missingness strategy"
        " → transformation"
        " → scaling"
        " → clustering"
    )


if __name__ == "__main__":
    main()