"""
GPU-FinOps
----------
Objective 1: Prepare final clustering matrices.

Base:
    Project Proposal

Feature decisions:
    Project EDA + Claude EDA interpretation

Purpose:
    1. Load feature-engineered job-level data.
    2. Select the final Objective-1 behavioral feature set.
    3. Build the primary clustering population using complete required
       behavioral measurements.
    4. Preserve excluded/unusable jobs in an eligibility report.
    5. Apply extreme-tail winsorization ONLY to K-Means/GMM copies.
    6. Scale separately for HDBSCAN, K-Means and GMM.
    7. Save matrices and audit information.

Important:
    - Master dataset is never modified.
    - Feature-engineered dataset is never modified.
    - No clustering is performed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gpu_finops_feature_engineered.csv"
)

HDBSCAN_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_matrix.csv"
)

KMEANS_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_kmeans_matrix.csv"
)

GMM_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_gmm_matrix.csv"
)

ELIGIBILITY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_clustering_eligibility.csv"
)

AUDIT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_clustering_matrix_audit.csv"
)

RATIO_TAIL_CAPS_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_ratio_tail_caps.csv"
)

FEATURE_SELECTION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_final_feature_selection.csv"
)


# ============================================================================
# SETTINGS
# ============================================================================

WINSORIZE_PERCENTILE = 0.995


# ============================================================================
# FINAL OBJECTIVE-1 FEATURE SET
# ============================================================================

# Final 12-feature behavioral representation.
#
# gpu_idle_ratio:
#     Removed from clustering because it is highly redundant with
#     gpu_utilization_intensity. It remains in the engineered dataset
#     for post-cluster interpretation.
#
# machine_count:
#     Removed from clustering because it is highly redundant with
#     task_fanout. It remains in the engineered dataset for
#     post-cluster interpretation.

CORE_FEATURES = [
    # ------------------------------------------------------------------------
    # Resource requirements
    # ------------------------------------------------------------------------
    "plan_cpu_mean",
    "plan_mem_mean",
    "gpu_demand_scale",

    # ------------------------------------------------------------------------
    # Utilization behavior
    # ------------------------------------------------------------------------
    "gpu_utilization_intensity",
    "gpu_memory_intensity",
    "memory_intensity",
    "cpu_usage_transformed",
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "io_intensity",

    # ------------------------------------------------------------------------
    # Execution characteristics
    # ------------------------------------------------------------------------
    "runtime_log",
    "task_fanout",
]


# Features intentionally retained in the engineered dataset but
# excluded from the clustering distance matrix.
INTERPRETATION_ONLY_FEATURES = [
    "gpu_idle_ratio",
    "machine_count",
]


# Contextual features intentionally excluded from clustering.
CONTEXT_FEATURES = [
    "job_name",
    "job_status",
    "has_telemetry",
    "telemetry_coverage",
    "has_gpu_request",
    "has_execution_timing",
    "user",
]


# ============================================================================
# TAIL FEATURES
# ============================================================================

# Extreme-tail treatment for distance-based models.
#
# For the two ratio features that are numerically unstable in the tail,
# we apply a deterministic high-percentile winsorization on the eligible
# clustering population before scaling. This is clustering-specific tail
# handling and is NOT a data-cleaning step for the source dataset.
TAIL_FEATURES = [
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
    "io_intensity",
    "runtime_log",
]

RATIO_TAIL_FEATURES = [
    "resource_efficiency_score",
    "cpu_gpu_imbalance",
]


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def ensure_directories() -> None:
    """Create required output directories."""

    HDBSCAN_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ELIGIBILITY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_SELECTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RATIO_TAIL_CAPS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_feature_engineered_data() -> pd.DataFrame:
    """Load and validate the feature-engineered dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = (
        CORE_FEATURES
        + INTERPRETATION_ONLY_FEATURES
        + [
            "job_name",
            "has_telemetry",
            "telemetry_coverage",
            "has_gpu_request",
            "has_execution_timing",
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    return df


# ============================================================================
# FEATURE SELECTION DOCUMENTATION
# ============================================================================

def save_feature_selection_table() -> None:
    """
    Save explicit documentation of:
        - core clustering features
        - interpretation-only features
        - contextual features
    """

    rows = []

    # ------------------------------------------------------------------------
    # Resource requirements
    # ------------------------------------------------------------------------

    resource = {
        "plan_cpu_mean": (
            "Resource Requirement",
            "Primary CPU demand",
            "CORE",
        ),
        "plan_mem_mean": (
            "Resource Requirement",
            "Primary memory demand",
            "CORE",
        ),
        "gpu_demand_scale": (
            "Resource Requirement",
            "Log-scaled GPU allocation magnitude",
            "CORE",
        ),
    }

    # ------------------------------------------------------------------------
    # Utilization behavior
    # ------------------------------------------------------------------------

    utilization = {
        "gpu_utilization_intensity": (
            "Utilization Behavior",
            "GPU compute intensity",
            "CORE",
        ),
        "gpu_memory_intensity": (
            "Utilization Behavior",
            "GPU memory footprint",
            "CORE",
        ),
        "memory_intensity": (
            "Utilization Behavior",
            "Host-memory intensity",
            "CORE",
        ),
        "cpu_usage_transformed": (
            "Utilization Behavior",
            "CPU consumption",
            "CORE",
        ),
        "resource_efficiency_score": (
            "Utilization Behavior",
            "GPU allocation-utilization relationship",
            "CORE",
        ),
        "cpu_gpu_imbalance": (
            "Utilization Behavior",
            "Relative CPU-GPU utilization imbalance",
            "CORE",
        ),
        "io_intensity": (
            "Utilization Behavior",
            "I/O activity per runtime",
            "CORE",
        ),
        "gpu_idle_ratio": (
            "Utilization Behavior",
            "GPU idleness; redundant with transformed GPU utilization",
            "INTERPRETATION_ONLY",
        ),
    }

    # ------------------------------------------------------------------------
    # Execution characteristics
    # ------------------------------------------------------------------------

    execution = {
        "runtime_log": (
            "Execution Characteristics",
            "Execution duration",
            "CORE",
        ),
        "task_fanout": (
            "Execution Characteristics",
            "Task/job fan-out",
            "CORE",
        ),
        "machine_count": (
            "Execution Characteristics",
            "Machine execution footprint; redundant with task fan-out",
            "INTERPRETATION_ONLY",
        ),
    }

    mapping = {
        **resource,
        **utilization,
        **execution,
    }

    for feature in mapping:
        axis, rationale, status = mapping[
            feature
        ]

        rows.append(
            {
                "feature": feature,
                "objective1_axis": axis,
                "rationale": rationale,
                "clustering_status": status,
            }
        )

    # ------------------------------------------------------------------------
    # Contextual features
    # ------------------------------------------------------------------------

    context_mapping = {
        "job_name": (
            "Identifier",
            "Job identifier; not behavioral",
        ),
        "user": (
            "Context",
            "Post-cluster analysis only",
        ),
        "job_status": (
            "Context",
            "Outcome validation only",
        ),
        "has_telemetry": (
            "Context",
            "Observability guardrail",
        ),
        "telemetry_coverage": (
            "Context",
            "Observability guardrail",
        ),
        "has_gpu_request": (
            "Context",
            "Availability guardrail",
        ),
        "has_execution_timing": (
            "Context",
            "Availability guardrail",
        ),
    }

    for feature, (
        axis,
        rationale,
    ) in context_mapping.items():

        rows.append(
            {
                "feature": feature,
                "objective1_axis": axis,
                "rationale": rationale,
                "clustering_status": "EXCLUDED",
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        FEATURE_SELECTION_OUTPUT,
        index=False,
    )


# ============================================================================
# MODELING POPULATION
# ============================================================================

def build_modeling_population(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build complete-case population for the final 12-feature matrix.

    Missing behavioral measurements are not fabricated.
    Excluded jobs remain documented in the eligibility report.
    """

    core = (
        df[
            CORE_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    valid_numeric = np.isfinite(
        core.to_numpy()
    ).all(
        axis=1
    )

    complete_case = (
        core.notna()
        .all(
            axis=1
        )
        & valid_numeric
    )

    modeling_df = df.loc[
        complete_case
    ].copy()

    excluded_df = df.loc[
        ~complete_case
    ].copy()

    total_jobs = len(
        df
    )

    eligibility = pd.DataFrame(
        [
            {
                "total_jobs": total_jobs,
                "eligible_jobs": len(
                    modeling_df
                ),
                "excluded_jobs": len(
                    excluded_df
                ),
                "eligible_percentage": (
                    len(modeling_df)
                    / total_jobs
                    * 100
                ),
                "excluded_percentage": (
                    len(excluded_df)
                    / total_jobs
                    * 100
                ),
                "excluded_missing_telemetry": int(
                    (
                        excluded_df[
                            "has_telemetry"
                        ]
                        == 0
                    ).sum()
                ),
                "excluded_missing_execution_timing": int(
                    (
                        excluded_df[
                            "has_execution_timing"
                        ]
                        == 0
                    ).sum()
                ),
                "excluded_missing_gpu_request": int(
                    (
                        excluded_df[
                            "has_gpu_request"
                        ]
                        == 0
                    ).sum()
                ),
            }
        ]
    )

    eligibility.to_csv(
        ELIGIBILITY_OUTPUT,
        index=False,
    )

    return (
        modeling_df,
        eligibility,
    )


# ============================================================================
# EXTRACT CORE MATRIX
# ============================================================================

def extract_core_matrix(
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract the final 12-feature behavioral matrix."""

    matrix = (
        modeling_df[
            CORE_FEATURES
        ]
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    return matrix


# ============================================================================
# WINSORIZATION
# ============================================================================

def compute_ratio_tail_caps(
    matrix: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """
    Compute deterministic lower/upper caps for the two unstable ratio
    features using only the eligible clustering population.

    This is a clustering-specific tail-treatment step only.
    It does not mutate the feature-engineered source dataset.
    """

    caps: dict[str, dict[str, float]] = {}
    rows = []

    for feature in RATIO_TAIL_FEATURES:
        if feature not in matrix.columns:
            continue

        series = pd.to_numeric(
            matrix[feature],
            errors="coerce",
        ).dropna()

        lower = float(
            series.quantile(0.005)
        )
        upper = float(
            series.quantile(0.995)
        )

        caps[feature] = {
            "lower": lower,
            "upper": upper,
        }

        rows.append(
            {
                "feature": feature,
                "lower_cap": lower,
                "upper_cap": upper,
            }
        )

    return caps, pd.DataFrame(rows)



def apply_ratio_tail_caps(
    matrix: pd.DataFrame,
    caps: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply deterministic lower/upper clipping to the unstable ratio features."""

    output = matrix.copy()

    for feature in RATIO_TAIL_FEATURES:
        if feature not in output.columns:
            continue

        lower = caps[feature]["lower"]
        upper = caps[feature]["upper"]

        output[feature] = output[feature].clip(
            lower=lower,
            upper=upper,
        )

    return output



def winsorize_for_distance_models(
    matrix: pd.DataFrame,
    ratio_caps: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Winsorize selected extreme-tail features for K-Means/GMM only.

    HDBSCAN receives the robust-scaled version of the original values,
    but the two ratio features also use the same eligible-population
    cap values for deterministic clustering-specific tail handling.
    """

    output = matrix.copy()

    for feature in TAIL_FEATURES:

        if feature not in output.columns:
            continue

        series = output[
            feature
        ]

        if feature in RATIO_TAIL_FEATURES and ratio_caps is not None:
            lower = ratio_caps[feature]["lower"]
            upper = ratio_caps[feature]["upper"]
            output[feature] = series.clip(
                lower=lower,
                upper=upper,
            )
            continue

        upper = series.quantile(
            WINSORIZE_PERCENTILE
        )

        output[
            feature
        ] = series.clip(
            upper=upper
        )

    return output


# ============================================================================
# SCALE MATRICES
# ============================================================================

def scale_hdbscan_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Robust-scale the HDBSCAN matrix."""

    scaler = RobustScaler()

    scaled = scaler.fit_transform(
        matrix
    )

    return pd.DataFrame(
        scaled,
        columns=CORE_FEATURES,
        index=matrix.index,
    )


def scale_standard_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Standard-scale the K-Means/GMM matrices."""

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        matrix
    )

    return pd.DataFrame(
        scaled,
        columns=CORE_FEATURES,
        index=matrix.index,
    )


# ============================================================================
# MATRIX AUDIT
# ============================================================================

def build_matrix_audit(
    matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Generate audit statistics for all algorithm-specific matrices."""

    rows = []

    for matrix_name, matrix in matrices.items():

        array = matrix.to_numpy(
            dtype=float
        )

        rows.append(
            {
                "matrix": matrix_name,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "nan_count": int(
                    matrix.isna()
                    .sum()
                    .sum()
                ),
                "inf_count": int(
                    np.isinf(
                        array
                    ).sum()
                ),
                "duplicate_rows": int(
                    matrix.duplicated()
                    .sum()
                ),
                "global_min": float(
                    np.nanmin(
                        array
                    )
                ),
                "global_max": float(
                    np.nanmax(
                        array
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def validate_matrices(
    matrices: dict[str, pd.DataFrame],
) -> None:
    """Validate matrices before clustering."""

    problems: list[str] = []

    for name, matrix in matrices.items():

        if matrix.empty:
            problems.append(
                f"{name}: matrix is empty"
            )

        if matrix.shape[1] != len(
            CORE_FEATURES
        ):
            problems.append(
                f"{name}: incorrect feature count"
            )

        if matrix.isna().any().any():
            problems.append(
                f"{name}: NaN values remain"
            )

        array = matrix.to_numpy(
            dtype=float
        )

        if np.isinf(
            array
        ).any():

            problems.append(
                f"{name}: infinite values remain"
            )

        # Duplicate behavioral vectors are VALID.
        # They are reported only as a diagnostic.
        duplicate_rows = int(
            matrix.duplicated().sum()
        )

        label_map = {
            "hdbscan": "HDBSCAN",
            "kmeans": "K-Means",
            "gmm": "GMM",
        }

        print(
            f"{label_map.get(name, name.upper())}: "
            f"duplicate behavioral vectors = "
            f"{duplicate_rows:,}"
        )

    if problems:
        raise ValueError(
            "\nFinal clustering-matrix validation failed:\n"
            + "\n".join(
                f"  - {problem}"
                for problem in problems
            )
        )


# ============================================================================
# SAVE MATRICES
# ============================================================================

def save_matrices(
    hdbscan_matrix: pd.DataFrame,
    kmeans_matrix: pd.DataFrame,
    gmm_matrix: pd.DataFrame,
) -> None:
    """Save scaled clustering matrices."""

    hdbscan_matrix.to_csv(
        HDBSCAN_OUTPUT,
        index=False,
    )

    kmeans_matrix.to_csv(
        KMEANS_OUTPUT,
        index=False,
    )

    gmm_matrix.to_csv(
        GMM_OUTPUT,
        index=False,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Prepare the final Objective-1 clustering matrices."""

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — CLUSTERING DATA PREPARATION"
    )
    print("=" * 80)

    ensure_directories()

    # ------------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------------

    print(
        "\n[1/7] Loading feature-engineered dataset..."
    )

    df = load_feature_engineered_data()

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    # ------------------------------------------------------------------------
    # 2. Save feature-selection documentation
    # ------------------------------------------------------------------------

    print(
        "\n[2/7] Saving final feature-selection documentation..."
    )

    save_feature_selection_table()

    # ------------------------------------------------------------------------
    # 3. Build modeling population
    # ------------------------------------------------------------------------

    print(
        "\n[3/7] Building Objective-1 modeling population..."
    )

    (
        modeling_df,
        eligibility,
    ) = build_modeling_population(
        df
    )

    print(
        f"Eligible jobs : "
        f"{len(modeling_df):,}"
    )

    print(
        f"Excluded jobs : "
        f"{len(df) - len(modeling_df):,}"
    )

    print(
        f"Eligible %    : "
        f"{eligibility.iloc[0]['eligible_percentage']:.2f}%"
    )

    # ------------------------------------------------------------------------
    # 4. Extract final 12-feature matrix
    # ------------------------------------------------------------------------

    print(
        "\n[4/7] Extracting final 12 core features..."
    )

    core_matrix = extract_core_matrix(
        modeling_df
    )

    print(
        f"Core matrix shape: "
        f"{core_matrix.shape}"
    )

    print(
        "\nFinal clustering features:"
    )

    for feature in CORE_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        "\nInterpretation-only features:"
    )

    for feature in INTERPRETATION_ONLY_FEATURES:
        print(
            f"  - {feature}"
        )

    # ------------------------------------------------------------------------
    # 5. Create algorithm-specific copies
    # ------------------------------------------------------------------------

    print(
        "\n[5/7] Creating algorithm-specific matrices..."
    )

    # Compute deterministic caps using the eligible population only.
    # These caps are for clustering-tail treatment, not source-data repair.
    ratio_caps, ratio_cap_table = compute_ratio_tail_caps(
        core_matrix
    )

    ratio_cap_table.to_csv(
        RATIO_TAIL_CAPS_OUTPUT,
        index=False,
    )

    print(
        "\nRatio-tail caps (eligible population):"
    )
    print(
        ratio_cap_table.to_string(
            index=False
        )
    )

    # HDBSCAN:
    # Apply the deterministic ratio-tail caps before robust scaling.
    # This is intentionally limited to the two unstable ratio features.
    hdbscan_pre_scaling = apply_ratio_tail_caps(
        core_matrix.copy(),
        ratio_caps,
    )

    hdbscan_matrix = (
        scale_hdbscan_matrix(
            hdbscan_pre_scaling
        )
    )

    # K-Means/GMM:
    # Preserve the current upper-tail winsorization logic, while
    # applying the same deterministic caps to the two ratio features.
    distance_model_raw = (
        winsorize_for_distance_models(
            core_matrix,
            ratio_caps,
        )
    )

    kmeans_matrix = (
        scale_standard_matrix(
            distance_model_raw
        )
    )

    gmm_matrix = (
        scale_standard_matrix(
            distance_model_raw
        )
    )

    # ------------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------------

    print(
        "\n[6/7] Validating clustering matrices..."
    )

    matrices = {
        "hdbscan": hdbscan_matrix,
        "kmeans": kmeans_matrix,
        "gmm": gmm_matrix,
    }

    validate_matrices(
        matrices
    )

    audit = build_matrix_audit(
        matrices
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    duplicate_vector_count = int(
        sum(
            matrix.duplicated().sum()
            for matrix in matrices.values()
        )
    )

    print(
        "\nEligible jobs          : "
        f"{len(modeling_df):,}"
    )
    print(
        "12-feature matrix shape: "
        f"{core_matrix.shape}"
    )
    print(
        "Ratio tail caps        :"
    )
    print(
        ratio_cap_table.to_string(
            index=False
        )
    )
    print(
        "NaN count              : "
        f"{int(sum(matrix.isna().sum().sum() for matrix in matrices.values()))}"
    )
    print(
        "Inf count              : "
        f"{int(sum(np.isinf(matrix.to_numpy(dtype=float)).sum() for matrix in matrices.values()))}"
    )
    print(
        "Duplicate vector count : "
        f"{duplicate_vector_count:,}"
    )
    print(
        "\nPRE-HDBSCAN STATUS     : REVIEW"
    )

    # ------------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------------

    print(
        "\n[7/7] Saving final clustering matrices..."
    )

    save_matrices(
        hdbscan_matrix=hdbscan_matrix,
        kmeans_matrix=kmeans_matrix,
        gmm_matrix=gmm_matrix,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "OBJECTIVE 1 CLUSTERING DATA PREPARATION COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "\nOutput matrices:"
    )

    print(
        f"  HDBSCAN : {HDBSCAN_OUTPUT}"
    )

    print(
        f"  K-Means : {KMEANS_OUTPUT}"
    )

    print(
        f"  GMM     : {GMM_OUTPUT}"
    )

    print(
        "\nEligibility report:"
    )

    print(
        f"  {ELIGIBILITY_OUTPUT}"
    )

    print(
        "\nMatrix audit:"
    )

    print(
        f"  {AUDIT_OUTPUT}"
    )

    print(
        "\nFeature selection:"
    )

    print(
        f"  {FEATURE_SELECTION_OUTPUT}"
    )

    print(
        "\nNext:"
    )

    print(
        "  Re-run pre-HDBSCAN audit → HDBSCAN → "
        "K-Means → GMM → validation → stability"
    )


if __name__ == "__main__":
    main()