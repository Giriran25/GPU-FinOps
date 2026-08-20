"""
GPU-FinOps
==========
Stage 3: Professional Exploratory Data Analysis (EDA)

Input
-----
data/processed/gpu_finops_job_master.csv

Outputs
-------
results/
├── figures/
│   ├── distributions/
│   ├── boxplots/
│   ├── relationships/
│   └── categorical/
├── tables/
│   ├── descriptive_statistics.csv
│   ├── missing_values.csv
│   ├── skewness_kurtosis.csv
│   ├── outlier_analysis.csv
│   ├── correlation_matrix.csv
│   └── categorical_summary.csv
└── metrics/
    ├── eda_summary.txt
    └── feature_quality_report.csv

Important
---------
- Full numerical statistics use the complete analytical dataset.
- Plotting uses a representative sample to keep memory and rendering
  manageable.
- Raw data is never modified.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")


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

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"
DISTRIBUTIONS_DIR = FIGURES_DIR / "distributions"
BOXPLOTS_DIR = FIGURES_DIR / "boxplots"
RELATIONSHIPS_DIR = FIGURES_DIR / "relationships"
CATEGORICAL_DIR = FIGURES_DIR / "categorical"

TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"


# ============================================================================
# SETTINGS
# ============================================================================

PLOT_SAMPLE_SIZE = 100_000

RANDOM_STATE = 42

OUTLIER_MULTIPLIER = 1.5

MAX_HIST_BINS = 60

CORRELATION_SAMPLE_SIZE = 100_000

TOP_CATEGORY_LIMIT = 15


# ============================================================================
# FEATURE GROUPS
# ============================================================================

RESOURCE_FEATURES = [
    "plan_cpu_mean",
    "plan_cpu_max",
    "plan_mem_mean",
    "plan_mem_max",
    "plan_gpu_mean",
    "plan_gpu_max",
    "tasks_with_plan_gpu",
    "tasks_missing_plan_gpu",
    "gpu_type_count",
]

EXECUTION_FEATURES = [
    "task_count",
    "unique_task_count",
    "terminated_task_count",
    "failed_task_count",
    "running_task_count",
    "waiting_task_count",
    "task_runtime_hours_mean",
    "task_runtime_hours_max",
    "instance_count",
    "tasks_with_instances",
    "executable_instance_count",
    "instance_runtime_hours_mean",
    "instance_runtime_hours_max",
    "instance_runtime_hours_sum",
    "machine_count",
]

UTILIZATION_FEATURES = [
    "gpu_util_mean",
    "gpu_util_max",
    "cpu_usage_mean",
    "avg_mem_mean",
    "max_mem_max",
    "avg_gpu_mem_mean",
    "max_gpu_mem_max",
    "total_read",
    "total_write",
    "total_read_count",
    "total_write_count",
]

TELEMETRY_FEATURES = [
    "telemetry_instance_count",
    "telemetry_record_count",
    "gpu_util_valid_count",
    "gpu_util_outlier_count",
    "observed_gpu_type_count",
    "telemetry_coverage",
]

COVERAGE_FEATURES = [
    "group_tag_instance_count",
    "workload_tag_instance_count",
    "machine_spec_coverage",
    "telemetry_coverage",
    "has_telemetry",
    "has_gpu_request",
    "has_execution_timing",
]


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def create_directories() -> None:
    """Create all EDA output directories."""

    directories = [
        DISTRIBUTIONS_DIR,
        BOXPLOTS_DIR,
        RELATIONSHIPS_DIR,
        CATEGORICAL_DIR,
        TABLES_DIR,
        METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_dataset() -> pd.DataFrame:
    """Load the cleaned job-level analytical dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{INPUT_FILE}"
        )

    print("=" * 80)
    print("GPU-FinOps | STAGE 3 — PROFESSIONAL EDA")
    print("=" * 80)

    print("\nLoading:")
    print(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(
        f"\nDataset shape: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


# ============================================================================
# BASIC DATASET OVERVIEW
# ============================================================================

def dataset_overview(
    df: pd.DataFrame,
) -> None:
    """Generate basic dataset information."""

    print("\n" + "=" * 80)
    print("1. DATASET OVERVIEW")
    print("=" * 80)

    rows, columns = df.shape

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_jobs = 0

    if "job_name" in df.columns:
        duplicate_jobs = int(
            df["job_name"].duplicated().sum()
        )

    overview = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "numeric_columns",
                "categorical_columns",
                "duplicate_rows",
                "duplicate_job_names",
            ],
            "value": [
                rows,
                columns,
                len(numeric_columns),
                len(categorical_columns),
                duplicate_rows,
                duplicate_jobs,
            ],
        }
    )

    overview.to_csv(
        TABLES_DIR / "dataset_overview.csv",
        index=False,
    )

    print(
        f"\nRows               : {rows:,}"
    )
    print(
        f"Columns            : {columns:,}"
    )
    print(
        f"Numeric features   : {len(numeric_columns):,}"
    )
    print(
        f"Categorical fields : {len(categorical_columns):,}"
    )
    print(
        f"Duplicate rows     : {duplicate_rows:,}"
    )
    print(
        f"Duplicate job_name : {duplicate_jobs:,}"
    )


# ============================================================================
# DATA TYPES
# ============================================================================

def analyze_dtypes(
    df: pd.DataFrame,
) -> None:
    """Save data type information."""

    dtype_table = pd.DataFrame(
        {
            "feature": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )

    dtype_table.to_csv(
        TABLES_DIR / "data_types.csv",
        index=False,
    )


# ============================================================================
# MISSINGNESS
# ============================================================================

def analyze_missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze missing values."""

    print("\n" + "=" * 80)
    print("2. MISSING-VALUE ANALYSIS")
    print("=" * 80)

    missing_count = df.isna().sum()

    missing_percentage = (
        missing_count
        / len(df)
        * 100
    )

    result = pd.DataFrame(
        {
            "feature": df.columns,
            "missing_count": missing_count.values,
            "missing_percentage": missing_percentage.values,
            "non_missing_count": (
                len(df) - missing_count
            ).values,
        }
    )

    result = result.sort_values(
        "missing_percentage",
        ascending=False,
    )

    result.to_csv(
        TABLES_DIR / "missing_values.csv",
        index=False,
    )

    print(
        result[
            result["missing_count"] > 0
        ]
        .head(20)
        .to_string(index=False)
    )

    # Missingness plot
    plot_data = result[
        result["missing_percentage"] > 0
    ].head(25)

    if not plot_data.empty:

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            plot_data["feature"],
            plot_data["missing_percentage"],
        )

        plt.xlabel(
            "Missing values (%)"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Top Features by Missingness"
        )

        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(
            DISTRIBUTIONS_DIR
            / "missingness.png",
            dpi=300,
        )

        plt.close()

    return result


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate complete descriptive statistics."""

    print("\n" + "=" * 80)
    print("3. DESCRIPTIVE STATISTICS")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    statistics = (
        numeric_df
        .describe()
        .T
    )

    statistics["missing"] = (
        numeric_df.isna().sum()
    )

    statistics["missing_pct"] = (
        numeric_df.isna().mean() * 100
    )

    statistics["zeros"] = (
        numeric_df
        .eq(0)
        .sum()
    )

    statistics["zero_pct"] = (
        numeric_df.eq(0).mean() * 100
    )

    statistics.to_csv(
        TABLES_DIR
        / "descriptive_statistics.csv"
    )

    print(
        statistics.head(20).to_string()
    )

    return statistics


# ============================================================================
# SKEWNESS AND KURTOSIS
# ============================================================================

def skewness_kurtosis_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate skewness and excess kurtosis.

    Interpretation guide:
        |skew| < 0.5     approximately symmetric
        0.5–1.0         moderately skewed
        > 1.0           highly skewed
    """

    print("\n" + "=" * 80)
    print("4. SKEWNESS AND KURTOSIS")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    rows = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if len(series) < 3:
            continue

        skew_value = float(
            skew(
                series,
                bias=False,
            )
        )

        kurtosis_value = float(
            kurtosis(
                series,
                fisher=True,
                bias=False,
            )
        )

        abs_skew = abs(
            skew_value
        )

        if abs_skew < 0.5:
            skew_class = "Approximately symmetric"
        elif abs_skew <= 1.0:
            skew_class = "Moderately skewed"
        else:
            skew_class = "Highly skewed"

        rows.append(
            {
                "feature": column,
                "skewness": skew_value,
                "abs_skewness": abs_skew,
                "excess_kurtosis": kurtosis_value,
                "skewness_class": skew_class,
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "abs_skewness",
            ascending=False,
        )
    )

    result.to_csv(
        TABLES_DIR
        / "skewness_kurtosis.csv",
        index=False,
    )

    print(
        result.head(20).to_string(
            index=False
        )
    )

    return result


# ============================================================================
# IQR AND OUTLIER ANALYSIS
# ============================================================================

def iqr_outlier_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Q1, median, Q3, IQR and Tukey outlier counts.

    Outlier rule:
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

    Important:
    This is an EDA statistical flag, NOT an automatic deletion rule.
    """

    print("\n" + "=" * 80)
    print("5. IQR / OUTLIER ANALYSIS")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    rows = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if series.empty:
            continue

        q1 = float(
            series.quantile(0.25)
        )

        median = float(
            series.quantile(0.50)
        )

        q3 = float(
            series.quantile(0.75)
        )

        iqr = q3 - q1

        lower_bound = (
            q1
            - OUTLIER_MULTIPLIER * iqr
        )

        upper_bound = (
            q3
            + OUTLIER_MULTIPLIER * iqr
        )

        lower_outliers = int(
            (series < lower_bound).sum()
        )

        upper_outliers = int(
            (series > upper_bound).sum()
        )

        total_outliers = (
            lower_outliers
            + upper_outliers
        )

        rows.append(
            {
                "feature": column,
                "count": len(series),
                "q1": q1,
                "median": median,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "lower_outliers": lower_outliers,
                "upper_outliers": upper_outliers,
                "total_outliers": total_outliers,
                "outlier_percentage": (
                    total_outliers
                    / len(series)
                    * 100
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "outlier_percentage",
            ascending=False,
        )
    )

    result.to_csv(
        TABLES_DIR
        / "outlier_analysis.csv",
        index=False,
    )

    print(
        result.head(30).to_string(
            index=False
        )
    )

    return result


# ============================================================================
# SAMPLE FOR PLOTS
# ============================================================================

def create_plot_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a reproducible plotting sample.

    Statistics still use the full dataset.
    """

    if len(df) <= PLOT_SAMPLE_SIZE:
        return df.copy()

    return df.sample(
        n=PLOT_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )


# ============================================================================
# HISTOGRAMS
# ============================================================================

def generate_histograms(
    df: pd.DataFrame,
) -> None:
    """Generate histograms for all numerical features."""

    print("\n" + "=" * 80)
    print("6. HISTOGRAMS")
    print("=" * 80)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for index, column in enumerate(
        numeric_columns,
        start=1,
    ):

        series = df[column].dropna()

        if series.empty:
            continue

        # Avoid plotting enormous outliers on normal-scale
        # distributions.
        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            series,
            bins=MAX_HIST_BINS,
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.title(
            f"Distribution: {column}"
        )

        plt.tight_layout()

        plt.savefig(
            DISTRIBUTIONS_DIR
            / f"{index:02d}_{column}_histogram.png",
            dpi=250,
        )

        plt.close()

    print(
        f"Generated histograms: "
        f"{len(numeric_columns):,}"
    )


# ============================================================================
# LOG-SCALE HISTOGRAMS FOR HIGHLY SKEWED FEATURES
# ============================================================================

def generate_log_histograms(
    df: pd.DataFrame,
    skew_table: pd.DataFrame,
) -> None:
    """
    Generate log1p histograms for highly skewed non-negative features.
    """

    print(
        "\nGenerating log-scale histograms..."
    )

    numeric_columns = set(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    highly_skewed = skew_table[
        skew_table["abs_skewness"] > 1.0
    ]

    for _, row in highly_skewed.iterrows():

        column = row["feature"]

        if column not in numeric_columns:
            continue

        series = df[column].dropna()

        if series.empty:
            continue

        if (series < 0).any():
            continue

        transformed = np.log1p(
            series
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            transformed,
            bins=MAX_HIST_BINS,
        )

        plt.xlabel(
            f"log1p({column})"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            f"Log-Transformed Distribution: {column}"
        )

        plt.tight_layout()

        safe_name = (
            column.replace(
                "/",
                "_",
            )
        )

        plt.savefig(
            DISTRIBUTIONS_DIR
            / f"log1p_{safe_name}.png",
            dpi=250,
        )

        plt.close()


# ============================================================================
# BOXPLOTS
# ============================================================================

def generate_boxplots(
    df: pd.DataFrame,
) -> None:
    """
    Generate individual boxplots.

    Full series are used for IQR analysis;
    plotting is limited to the sample for rendering efficiency.
    """

    print("\n" + "=" * 80)
    print("7. BOXPLOTS")
    print("=" * 80)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    sample = create_plot_sample(
        df
    )

    for index, column in enumerate(
        numeric_columns,
        start=1,
    ):

        values = sample[column].dropna()

        if values.empty:
            continue

        plt.figure(
            figsize=(10, 5)
        )

        plt.boxplot(
            values,
            vert=False,
            showfliers=True,
        )

        plt.xlabel(column)

        plt.title(
            f"Boxplot: {column}"
        )

        plt.tight_layout()

        plt.savefig(
            BOXPLOTS_DIR
            / f"{index:02d}_{column}_boxplot.png",
            dpi=250,
        )

        plt.close()


# ============================================================================
# CORRELATION MATRIX
# ============================================================================

def correlation_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix.

    Correlation is computed on a representative sample to
    control memory and computation cost.
    """

    print("\n" + "=" * 80)
    print("8. CORRELATION ANALYSIS")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df) > CORRELATION_SAMPLE_SIZE:

        correlation_df = (
            numeric_df.sample(
                n=CORRELATION_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
        )

    else:

        correlation_df = numeric_df

    correlation = (
        correlation_df
        .corr(
            method="pearson"
        )
    )

    correlation.to_csv(
        TABLES_DIR
        / "correlation_matrix.csv"
    )

    # Full heatmap
    plt.figure(
        figsize=(22, 18)
    )

    image = plt.imshow(
        correlation.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label="Pearson correlation",
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
        fontsize=7,
    )

    plt.title(
        "Correlation Heatmap — Numerical Features"
    )

    plt.tight_layout()

    plt.savefig(
        RELATIONSHIPS_DIR
        / "correlation_heatmap.png",
        dpi=300,
    )

    plt.close()

    # Strong correlation pairs
    pairs = []

    columns = correlation.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation.iloc[
                i,
                j,
            ]

            if pd.isna(value):
                continue

            pairs.append(
                {
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "correlation": value,
                    "absolute_correlation": abs(
                        value
                    ),
                }
            )

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
    )

    pairs_df.to_csv(
        TABLES_DIR
        / "correlation_pairs.csv",
        index=False,
    )

    print(
        "\nTop absolute correlations:"
    )

    print(
        pairs_df.head(25).to_string(
            index=False
        )
    )

    return correlation


# ============================================================================
# CATEGORICAL ANALYSIS
# ============================================================================

def categorical_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze categorical features."""

    print("\n" + "=" * 80)
    print("9. CATEGORICAL ANALYSIS")
    print("=" * 80)

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    all_rows = []

    for column in categorical_columns:

        counts = (
            df[column]
            .value_counts(
                dropna=False
            )
            .head(TOP_CATEGORY_LIMIT)
        )

        percentage = (
            counts
            / len(df)
            * 100
        )

        summary = pd.DataFrame(
            {
                "feature": column,
                "category": counts.index.astype(str),
                "count": counts.values,
                "percentage": percentage.values,
            }
        )

        all_rows.append(
            summary
        )

        if column == "job_status":

            plt.figure(
                figsize=(10, 6)
            )

            plt.bar(
                counts.index.astype(str),
                counts.values,
            )

            plt.xlabel(
                column
            )

            plt.ylabel(
                "Count"
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.xticks(
                rotation=30,
                ha="right",
            )

            plt.tight_layout()

            plt.savefig(
                CATEGORICAL_DIR
                / f"{column}_distribution.png",
                dpi=300,
            )

            plt.close()

    if all_rows:

        categorical_summary = pd.concat(
            all_rows,
            ignore_index=True,
        )

        categorical_summary.to_csv(
            TABLES_DIR
            / "categorical_summary.csv",
            index=False,
        )


# ============================================================================
# RESOURCE VS UTILIZATION RELATIONSHIPS
# ============================================================================

def generate_relationship_plots(
    df: pd.DataFrame,
) -> None:
    """Generate research-oriented scatter plots."""

    print("\n" + "=" * 80)
    print("10. RESOURCE / UTILIZATION RELATIONSHIPS")
    print("=" * 80)

    sample = create_plot_sample(
        df
    )

    relationships = [
        (
            "plan_gpu_mean",
            "gpu_util_mean",
            "GPU Demand vs GPU Utilization",
        ),
        (
            "plan_cpu_mean",
            "gpu_util_mean",
            "CPU Demand vs GPU Utilization",
        ),
        (
            "plan_mem_mean",
            "gpu_util_mean",
            "Memory Demand vs GPU Utilization",
        ),
        (
            "task_runtime_hours_mean",
            "gpu_util_mean",
            "Runtime vs GPU Utilization",
        ),
        (
            "avg_gpu_mem_mean",
            "gpu_util_mean",
            "GPU Memory vs GPU Utilization",
        ),
        (
            "instance_runtime_hours_sum",
            "plan_gpu_mean",
            "Runtime vs GPU Demand",
        ),
    ]

    for x_column, y_column, title in relationships:

        if (
            x_column not in sample.columns
            or y_column not in sample.columns
        ):
            continue

        plot_df = sample[
            [
                x_column,
                y_column,
            ]
        ].dropna()

        if plot_df.empty:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        plt.scatter(
            plot_df[x_column],
            plot_df[y_column],
            alpha=0.25,
            s=10,
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.title(title)

        plt.tight_layout()

        filename = (
            f"{x_column}_vs_{y_column}.png"
        )

        plt.savefig(
            RELATIONSHIPS_DIR / filename,
            dpi=300,
        )

        plt.close()


# ============================================================================
# GPU UTILIZATION ANALYSIS
# ============================================================================

def gpu_utilization_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze GPU utilization specifically."""

    if "gpu_util_mean" not in df.columns:
        return

    print("\n" + "=" * 80)
    print("11. GPU UTILIZATION ANALYSIS")
    print("=" * 80)

    telemetry_df = df[
        df["has_telemetry"] == 1
    ].copy()

    if telemetry_df.empty:
        print(
            "No telemetry-covered jobs found."
        )
        return

    utilization_stats = (
        telemetry_df[
            [
                "gpu_util_mean",
                "gpu_util_max",
            ]
        ]
        .describe()
        .T
    )

    utilization_stats.to_csv(
        TABLES_DIR
        / "gpu_utilization_statistics.csv"
    )

    # Utilization histogram
    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        telemetry_df["gpu_util_mean"]
        .dropna(),
        bins=50,
    )

    plt.xlabel(
        "Mean GPU utilization"
    )

    plt.ylabel(
        "Number of jobs"
    )

    plt.title(
        "Distribution of Mean GPU Utilization"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR
        / "gpu_utilization_distribution.png",
        dpi=300,
    )

    plt.close()

    # Utilization boxplot
    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        telemetry_df[
            "gpu_util_mean"
        ].dropna(),
        vert=False,
    )

    plt.xlabel(
        "Mean GPU utilization"
    )

    plt.title(
        "GPU Utilization Boxplot"
    )

    plt.tight_layout()

    plt.savefig(
        BOXPLOTS_DIR
        / "gpu_utilization_boxplot.png",
        dpi=300,
    )

    plt.close()

    # Simple utilization bands for EDA only.
    # These are NOT our final ML labels.
    bins = [
        -np.inf,
        10,
        30,
        70,
        np.inf,
    ]

    labels = [
        "Very Low (<10)",
        "Low (10–30)",
        "Moderate (30–70)",
        "High (>70)",
    ]

    bands = pd.cut(
        telemetry_df[
            "gpu_util_mean"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    band_counts = (
        bands
        .value_counts()
        .sort_index()
    )

    band_table = pd.DataFrame(
        {
            "utilization_band": band_counts.index.astype(str),
            "job_count": band_counts.values,
            "percentage": (
                band_counts.values
                / len(telemetry_df)
                * 100
            ),
        }
    )

    band_table.to_csv(
        TABLES_DIR
        / "gpu_utilization_bands_eda.csv",
        index=False,
    )


# ============================================================================
# RUNTIME ANALYSIS
# ============================================================================

def runtime_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze execution runtime."""

    print("\n" + "=" * 80)
    print("12. RUNTIME ANALYSIS")
    print("=" * 80)

    if "instance_runtime_hours_sum" not in df.columns:
        return

    runtime = df[
        "instance_runtime_hours_sum"
    ].dropna()

    runtime = runtime[
        runtime >= 0
    ]

    if runtime.empty:
        return

    stats = runtime.describe()

    stats.to_frame(
        name="runtime_hours"
    ).to_csv(
        TABLES_DIR
        / "runtime_statistics.csv"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        runtime,
        bins=MAX_HIST_BINS,
    )

    plt.xlabel(
        "Runtime (hours)"
    )

    plt.ylabel(
        "Number of jobs"
    )

    plt.title(
        "Job Runtime Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR
        / "runtime_distribution.png",
        dpi=300,
    )

    plt.close()

    positive_runtime = runtime[
        runtime >= 0
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        np.log1p(
            positive_runtime
        ),
        bins=MAX_HIST_BINS,
    )

    plt.xlabel(
        "log1p(Runtime hours)"
    )

    plt.ylabel(
        "Number of jobs"
    )

    plt.title(
        "Log-Transformed Runtime Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR
        / "log_runtime_distribution.png",
        dpi=300,
    )

    plt.close()


# ============================================================================
# TELEMETRY COVERAGE ANALYSIS
# ============================================================================

def telemetry_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze telemetry availability."""

    print("\n" + "=" * 80)
    print("13. TELEMETRY COVERAGE ANALYSIS")
    print("=" * 80)

    coverage = (
        df["telemetry_coverage"]
        .clip(0, 1)
    )

    print(
        coverage.describe()
    )

    coverage.to_frame(
        name="telemetry_coverage"
    ).describe().to_csv(
        TABLES_DIR
        / "telemetry_coverage_statistics.csv"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        coverage,
        bins=30,
    )

    plt.xlabel(
        "Telemetry coverage"
    )

    plt.ylabel(
        "Number of jobs"
    )

    plt.title(
        "Telemetry Coverage Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTIONS_DIR
        / "telemetry_coverage_distribution.png",
        dpi=300,
    )

    plt.close()


# ============================================================================
# FEATURE QUALITY REPORT
# ============================================================================

def feature_quality_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine:
        missingness
        skewness
        IQR outlier percentage
        zero percentage
        unique count

    into a single feature quality report.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    rows = []

    for column in numeric_df.columns:

        series = numeric_df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (non_null < lower)
                | (non_null > upper)
            ).sum()
        )

        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_percentage": (
                    series.isna().mean()
                    * 100
                ),
                "zero_count": int(
                    non_null.eq(0).sum()
                ),
                "zero_percentage": (
                    non_null.eq(0).mean()
                    * 100
                ),
                "unique_values": int(
                    non_null.nunique()
                ),
                "q1": q1,
                "median": non_null.median(),
                "q3": q3,
                "iqr": iqr,
                "lower_iqr_bound": lower,
                "upper_iqr_bound": upper,
                "outlier_count": outlier_count,
                "outlier_percentage": (
                    outlier_count
                    / len(non_null)
                    * 100
                ),
                "skewness": float(
                    skew(
                        non_null,
                        bias=False,
                    )
                ),
                "excess_kurtosis": float(
                    kurtosis(
                        non_null,
                        fisher=True,
                        bias=False,
                    )
                ),
            }
        )

    report = (
        pd.DataFrame(rows)
        .sort_values(
            "outlier_percentage",
            ascending=False,
        )
    )

    report.to_csv(
        METRICS_DIR
        / "feature_quality_report.csv",
        index=False,
    )

    return report


# ============================================================================
# AUTOMATED EDA SUMMARY
# ============================================================================

def generate_eda_summary(
    df: pd.DataFrame,
    quality_report: pd.DataFrame,
    skew_table: pd.DataFrame,
    outlier_table: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Generate a concise machine-readable/text EDA summary."""

    summary_lines = []

    summary_lines.append(
        "GPU-FinOps — EDA SUMMARY"
    )

    summary_lines.append(
        "=" * 70
    )

    summary_lines.append(
        f"Rows: {len(df):,}"
    )

    summary_lines.append(
        f"Columns: {len(df.columns):,}"
    )

    summary_lines.append(
        f"Numeric features: "
        f"{len(df.select_dtypes(include=np.number).columns):,}"
    )

    summary_lines.append(
        f"Categorical features: "
        f"{len(df.select_dtypes(exclude=np.number).columns):,}"
    )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------------

    top_missing = (
        quality_report
        .sort_values(
            "missing_percentage",
            ascending=False,
        )
        .head(10)
    )

    summary_lines.append(
        "TOP MISSINGNESS FEATURES"
    )

    for _, row in top_missing.iterrows():

        summary_lines.append(
            f"{row['feature']}: "
            f"{row['missing_percentage']:.2f}%"
        )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # Highly skewed
    # ---------------------------------------------------------------

    highly_skewed = (
        skew_table[
            skew_table["abs_skewness"] > 1.0
        ]
        .head(15)
    )

    summary_lines.append(
        "HIGHLY SKEWED FEATURES"
    )

    for _, row in highly_skewed.iterrows():

        summary_lines.append(
            f"{row['feature']}: "
            f"skew={row['skewness']:.3f}"
        )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # Outliers
    # ---------------------------------------------------------------

    top_outliers = (
        outlier_table
        .head(15)
    )

    summary_lines.append(
        "TOP IQR OUTLIER FEATURES"
    )

    for _, row in top_outliers.iterrows():

        summary_lines.append(
            f"{row['feature']}: "
            f"{row['outlier_percentage']:.2f}%"
        )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # Strong correlations
    # ---------------------------------------------------------------

    strong_pairs = []

    columns = correlation.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation.iloc[
                i,
                j,
            ]

            if pd.isna(value):
                continue

            if abs(value) >= 0.80:

                strong_pairs.append(
                    (
                        columns[i],
                        columns[j],
                        value,
                    )
                )

    strong_pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    summary_lines.append(
        "STRONG CORRELATIONS |r| >= 0.80"
    )

    for feature_1, feature_2, value in (
        strong_pairs[:20]
    ):

        summary_lines.append(
            f"{feature_1} ↔ {feature_2}: "
            f"{value:.3f}"
        )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # GPU utilization
    # ---------------------------------------------------------------

    if "gpu_util_mean" in df.columns:

        gpu_util = df[
            "gpu_util_mean"
        ].dropna()

        summary_lines.append(
            "GPU UTILIZATION"
        )

        summary_lines.append(
            f"Mean: {gpu_util.mean():.4f}"
        )

        summary_lines.append(
            f"Median: {gpu_util.median():.4f}"
        )

        summary_lines.append(
            f"Maximum: {gpu_util.max():.4f}"
        )

    summary_lines.append(
        ""
    )

    # ---------------------------------------------------------------
    # Telemetry
    # ---------------------------------------------------------------

    if "has_telemetry" in df.columns:

        telemetry_rate = (
            df["has_telemetry"]
            .mean()
            * 100
        )

        summary_lines.append(
            "TELEMETRY AVAILABILITY"
        )

        summary_lines.append(
            f"Jobs with telemetry: "
            f"{telemetry_rate:.2f}%"
        )

    summary_lines.append(
        ""
    )

    summary_lines.append(
        "EDA interpretation:"
    )

    summary_lines.append(
        "- IQR outliers are statistical flags, "
        "not automatic deletion candidates."
    )

    summary_lines.append(
        "- Missing telemetry must not be interpreted "
        "as zero utilization."
    )

    summary_lines.append(
        "- Highly skewed positive variables may require "
        "log1p transformation during feature engineering."
    )

    summary_lines.append(
        "- Highly correlated variables should be reviewed "
        "for redundancy before clustering."
    )

    summary_lines.append(
        "- Final workload labels should be learned by "
        "clustering rather than manually assigned from EDA bands."
    )

    with open(
        METRICS_DIR / "eda_summary.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                summary_lines
            )
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the complete professional EDA pipeline."""

    create_directories()

    df = load_dataset()

    # ---------------------------------------------------------------
    # Core diagnostics
    # ---------------------------------------------------------------

    dataset_overview(df)

    analyze_dtypes(df)

    missingness = analyze_missingness(
        df
    )

    descriptive_statistics(
        df
    )

    skew_table = (
        skewness_kurtosis_analysis(
            df
        )
    )

    outlier_table = (
        iqr_outlier_analysis(
            df
        )
    )

    # ---------------------------------------------------------------
    # Visual analysis
    # ---------------------------------------------------------------

    generate_histograms(
        df
    )

    generate_log_histograms(
        df,
        skew_table,
    )

    generate_boxplots(
        df
    )

    correlation = (
        correlation_analysis(
            df
        )
    )

    categorical_analysis(
        df
    )

    # ---------------------------------------------------------------
    # Research-oriented analyses
    # ---------------------------------------------------------------

    generate_relationship_plots(
        df
    )

    gpu_utilization_analysis(
        df
    )

    runtime_analysis(
        df
    )

    telemetry_analysis(
        df
    )

    # ---------------------------------------------------------------
    # Final quality report
    # ---------------------------------------------------------------

    quality_report = (
        feature_quality_report(
            df
        )
    )

    generate_eda_summary(
        df=df,
        quality_report=quality_report,
        skew_table=skew_table,
        outlier_table=outlier_table,
        correlation=correlation,
    )

    print("\n" + "=" * 80)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated directories:")

    print(
        f"Figures      : {FIGURES_DIR}"
    )

    print(
        f"Tables       : {TABLES_DIR}"
    )

    print(
        f"Metrics      : {METRICS_DIR}"
    )

    print("\nImportant outputs:")

    print(
        "  - descriptive_statistics.csv"
    )

    print(
        "  - missing_values.csv"
    )

    print(
        "  - skewness_kurtosis.csv"
    )

    print(
        "  - outlier_analysis.csv"
    )

    print(
        "  - feature_quality_report.csv"
    )

    print(
        "  - correlation_matrix.csv"
    )

    print(
        "  - correlation_pairs.csv"
    )

    print(
        "  - eda_summary.txt"
    )


if __name__ == "__main__":
    main()