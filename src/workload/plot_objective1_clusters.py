"""
GPU-FinOps | OBJECTIVE 1 — CONFERENCE CLUSTER VISUALIZATIONS

Purpose
-------
Generate publication-ready visualizations for the final HDBSCAN workload
taxonomy.

Figures:
    1. PCA 2D cluster visualization
    2. Cluster profile heatmap

PCA is used ONLY for visualization.
It does not alter the clustering results.

The final HDBSCAN assignments remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_assignments.csv"
)

PROFILE_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_cluster_profiles.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42

PCA_SAMPLE_SIZE = 25_000

FEATURES = [
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
# HELPERS
# =============================================================================

def ensure_output_dir() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    matrix = pd.read_csv(
        MATRIX_FILE,
        low_memory=False,
    )

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        low_memory=False,
    )

    profiles = pd.read_csv(
        PROFILE_FILE,
        low_memory=False,
    )

    if len(matrix) != len(assignments):
        raise ValueError(
            "Matrix and assignment row counts differ."
        )

    return matrix, assignments, profiles


# =============================================================================
# FIGURE 1 — PCA CLUSTER MAP
# =============================================================================

def create_pca_plot(
    matrix: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    print("\n[1/2] Creating PCA cluster visualization...")

    X = matrix[
        FEATURES
    ].to_numpy(dtype=np.float64)

    labels = assignments[
        "cluster_id"
    ].to_numpy(dtype=int)

    # Exclude HDBSCAN noise for the primary cluster visualization.
    non_noise_mask = labels != -1

    X_non_noise = X[
        non_noise_mask
    ]

    labels_non_noise = labels[
        non_noise_mask
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(X_non_noise) > PCA_SAMPLE_SIZE:
        sample_idx = rng.choice(
            len(X_non_noise),
            size=PCA_SAMPLE_SIZE,
            replace=False,
        )
        X_plot = X_non_noise[
            sample_idx
        ]
        y_plot = labels_non_noise[
            sample_idx
        ]
    else:
        X_plot = X_non_noise
        y_plot = labels_non_noise

    # PCA is only for visualization.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_plot
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    explained = (
        pca.explained_variance_ratio_
        * 100
    )

    plt.figure(
        figsize=(10, 7),
        dpi=220,
    )

    # Plot one cluster at a time so that the legend remains interpretable.
    for cluster_id in sorted(
        np.unique(y_plot)
    ):
        mask = y_plot == cluster_id

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8,
            alpha=0.55,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel(
        f"PC1 ({explained[0]:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained[1]:.1f}% variance)"
    )

    plt.title(
        "GPU-FinOps — Final HDBSCAN Workload Profiles"
    )

    plt.legend(
        title="Workload Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.grid(
        alpha=0.20,
        linewidth=0.5,
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "objective1_hdbscan_pca_clusters.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# FIGURE 2 — PROFILE HEATMAP
# =============================================================================

def create_profile_heatmap(
    profiles: pd.DataFrame,
) -> None:
    print("\n[2/2] Creating workload-profile heatmap...")

    mean_columns = [
        f"{feature}_mean"
        for feature in FEATURES
        if f"{feature}_mean" in profiles.columns
    ]

    if not mean_columns:
        raise ValueError(
            "No feature mean columns found in cluster profiles."
        )

    heatmap_df = profiles[
        [
            "cluster_id",
            *mean_columns,
        ]
    ].copy()

    heatmap_df = heatmap_df.set_index(
        "cluster_id"
    )

    # Standardize each feature across clusters so the figure shows
    # relative behavioral signatures rather than raw magnitude.
    values = heatmap_df.to_numpy(
        dtype=np.float64
    )

    column_mean = np.nanmean(
        values,
        axis=0,
        keepdims=True,
    )

    column_std = np.nanstd(
        values,
        axis=0,
        keepdims=True,
    )

    column_std[
        column_std == 0
    ] = 1.0

    standardized = (
        values - column_mean
    ) / column_std

    plt.figure(
        figsize=(14, 7),
        dpi=220,
    )

    image = plt.imshow(
        standardized,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label="Relative cluster feature intensity (z-score)",
    )

    plt.xticks(
        np.arange(
            len(mean_columns)
        ),
        [
            column.replace(
                "_mean",
                "",
            ).replace(
                "_",
                " ",
            )
            for column in mean_columns
        ],
        rotation=70,
        ha="right",
    )

    plt.yticks(
        np.arange(
            len(heatmap_df)
        ),
        [
            f"Cluster {cluster_id}"
            for cluster_id in heatmap_df.index
        ],
    )

    plt.xlabel(
        "Behavioral Features"
    )

    plt.ylabel(
        "HDBSCAN Workload Cluster"
    )

    plt.title(
        "GPU-FinOps — Behavioral Signatures of Final Workload Profiles"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "objective1_cluster_profile_heatmap.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_output_dir()

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — CONFERENCE VISUALIZATIONS"
    )
    print("=" * 80)

    print("\nLoading final Objective-1 outputs...")

    matrix, assignments, profiles = (
        load_inputs()
    )

    print(
        f"Jobs    : {len(matrix):,}"
        f"\nFeatures: {len(FEATURES)}"
    )

    create_pca_plot(
        matrix,
        assignments,
    )

    create_profile_heatmap(
        profiles,
    )

    print("\nSaved conference figures:")
    print(
        FIGURE_DIR
        / "objective1_hdbscan_pca_clusters.png"
    )
    print(
        FIGURE_DIR
        / "objective1_cluster_profile_heatmap.png"
    )

    print(
        "\nSTOP: Objective-1 visualization generation completed."
    )


if __name__ == "__main__":
    main()