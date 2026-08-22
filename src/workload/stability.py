"""
GPU-FinOps | OBJECTIVE 1 — HDBSCAN STABILITY ANALYSIS

Purpose
-------
Evaluate stability of the frozen final HDBSCAN workload taxonomy.

Frozen configuration:
    min_cluster_size = 15000
    min_samples      = 300

Procedure:
    1. Load the final 12-feature Objective-1 matrix.
    2. Select a deterministic 100,000-job stability population.
    3. Perform 20 repeated 80% resamples.
    4. Fit HDBSCAN on each resample.
    5. Compare assignments on the shared overlapping evaluation jobs.
    6. Report ARI and AMI against the reference final-model assignments.
    7. Save detailed and summary stability results.

No K-Means.
No GMM.
No Objective 2/3.
No hyperparameter tuning.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


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

DETAIL_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "objective1_hdbscan_stability.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_hdbscan_stability_summary.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42

FINAL_MIN_CLUSTER_SIZE = 15_000
FINAL_MIN_SAMPLES = 300

FULL_POPULATION_SIZE = 599_288
STABILITY_POPULATION_SIZE = 100_000
POPULATION_PROPORTIONAL_MIN_CLUSTER_SIZE = round(
    FINAL_MIN_CLUSTER_SIZE
    * STABILITY_POPULATION_SIZE
    / FULL_POPULATION_SIZE
)
STABILITY_MIN_CLUSTER_SIZE = 2_500
RESAMPLE_FRACTION = 0.80
N_RESAMPLES = 20

EXPECTED_ROWS = FULL_POPULATION_SIZE
EXPECTED_FEATURES = 12

RESOLUTION_MATCHING_METHOD = (
    "population-proportional min_cluster_size"
)

# Use one CPU worker to keep laptop resource usage predictable.
CORE_DIST_N_JOBS = 1


# =============================================================================
# HELPERS
# =============================================================================

def ensure_directories() -> None:
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


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
            f"Expected matrix shape {(EXPECTED_ROWS, EXPECTED_FEATURES)}, "
            f"got {df.shape}."
        )

    values = df.to_numpy(dtype=np.float64)

    if np.isnan(values).any():
        raise ValueError("Matrix contains NaN values.")

    if not np.isfinite(values).all():
        raise ValueError("Matrix contains non-finite values.")

    return df


def load_final_assignments() -> np.ndarray:
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Final HDBSCAN assignments not found:\n{ASSIGNMENTS_FILE}"
        )

    df = pd.read_csv(
        ASSIGNMENTS_FILE,
        low_memory=False,
    )

    if "cluster_id" not in df.columns:
        raise ValueError(
            "Final assignments must contain cluster_id."
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} assignments, got {len(df):,}."
        )

    if "matrix_row_index" in df.columns:
        expected = np.arange(
            EXPECTED_ROWS,
            dtype=int,
        )

        actual = df["matrix_row_index"].to_numpy(
            dtype=int
        )

        if not np.array_equal(actual, expected):
            raise ValueError(
                "Final HDBSCAN assignment indices are not aligned "
                "with the objective1 matrix."
            )

    return df["cluster_id"].to_numpy(dtype=int)


def fit_hdbscan(
    X: np.ndarray,
) -> hdbscan.HDBSCAN:
    model = hdbscan.HDBSCAN(
        min_cluster_size=STABILITY_MIN_CLUSTER_SIZE,
        min_samples=FINAL_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=CORE_DIST_N_JOBS,
    )

    model.fit(X)

    return model


def summarize_labels(
    labels: np.ndarray,
) -> dict[str, Any]:
    non_noise = labels != -1

    if non_noise.any():
        cluster_count = int(
            len(np.unique(labels[non_noise]))
        )
    else:
        cluster_count = 0

    return {
        "cluster_count": cluster_count,
        "noise_fraction": float((~non_noise).mean()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("GPU-FinOps | OBJECTIVE 1 — HDBSCAN STABILITY ANALYSIS")
    print("=" * 80)

    print(
        "\nFrozen configuration:"
        f"\n  min_cluster_size = {FINAL_MIN_CLUSTER_SIZE:,}"
        f"\n  min_samples      = {FINAL_MIN_SAMPLES:,}"
    )

    print(
        "\nSTABILITY TYPE:\n"
        "resolution-matched subsample stability"
    )

    print(
        f"\nStability population : {STABILITY_POPULATION_SIZE:,}"
        f"\nStability min_cluster_size: "
        f"{STABILITY_MIN_CLUSTER_SIZE:,}"
        f"\nResamples            : {N_RESAMPLES}"
        f"\nResample fraction    : {RESAMPLE_FRACTION:.0%}"
        f"\nRandom state         : {RANDOM_STATE}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading matrix and final HDBSCAN assignments...")

    matrix_df = load_matrix()

    X_full = matrix_df.to_numpy(
        dtype=np.float64
    )

    reference_labels = load_final_assignments()

    # -------------------------------------------------------------------------
    # Build fixed stability population
    # -------------------------------------------------------------------------
    print(
        f"\n[2/4] Creating fixed {STABILITY_POPULATION_SIZE:,}-job "
        "stability population..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    population_indices = np.sort(
        rng.choice(
            EXPECTED_ROWS,
            size=STABILITY_POPULATION_SIZE,
            replace=False,
        )
    )

    X_population = X_full[
        population_indices
    ]

    reference_population_labels = reference_labels[
        population_indices
    ]

    print(
        f"Reference clusters: "
        f"{len(np.unique(reference_population_labels[reference_population_labels != -1]))}"
    )

    print("\n[3/4] Running stability resamples...")

    rows: list[dict[str, Any]] = []

    # We retain a fixed overlap evaluation population inside the
    # 100k stability population so ARI/AMI are always comparable.
    overlap_size = int(
        STABILITY_POPULATION_SIZE * 0.50
    )

    overlap_rng = np.random.default_rng(
        RANDOM_STATE + 1000
    )

    evaluation_population_positions = np.sort(
        overlap_rng.choice(
            STABILITY_POPULATION_SIZE,
            size=overlap_size,
            replace=False,
        )
    )

    resample_size = int(
        STABILITY_POPULATION_SIZE
        * RESAMPLE_FRACTION
    )

    for resample_id in range(
        1,
        N_RESAMPLES + 1,
    ):
        run_start = time.perf_counter()

        print(
            f"[{resample_id}/{N_RESAMPLES}] "
            f"80% resample",
            flush=True,
        )

        sample_rng = np.random.default_rng(
            RANDOM_STATE + resample_id
        )

        sample_positions = np.sort(
            sample_rng.choice(
                STABILITY_POPULATION_SIZE,
                size=resample_size,
                replace=False,
            )
        )

        X_sample = X_population[
            sample_positions
        ]

        model = None
        overlap_rows = 0
        comparison_rows = 0

        try:
            model = fit_hdbscan(
                X_sample
            )

            sample_labels = model.labels_

            # The comparison population is the intersection between
            # this resample and the fixed evaluation population.
            sample_position_set = set(
                sample_positions.tolist()
            )

            overlap_positions = np.array(
                [
                    p
                    for p in evaluation_population_positions
                    if p in sample_position_set
                ],
                dtype=int,
            )

            overlap_rows = len(overlap_positions)

            if overlap_rows < 2:
                raise RuntimeError(
                    "Insufficient overlap for stability comparison."
                )

            # Convert population positions to positions in sample_labels.
            sample_position_lookup = {
                position: idx
                for idx, position in enumerate(sample_positions)
            }

            sample_label_positions = np.array(
                [
                    sample_position_lookup[p]
                    for p in overlap_positions
                ],
                dtype=int,
            )

            predicted_labels = sample_labels[
                sample_label_positions
            ]

            reference_labels_overlap = (
                reference_population_labels[
                    overlap_positions
                ]
            )

            # Remove resample noise for the primary stability comparison.
            # This keeps ARI/AMI focused on workload clusters rather than
            # differences in HDBSCAN's treatment of boundary/noise jobs.
            comparison_mask = (
                (reference_labels_overlap != -1)
                & (predicted_labels != -1)
            )

            comparison_rows = int(comparison_mask.sum())

            if comparison_rows < 1000:
                raise RuntimeError(
                    "Quality gate failed: comparison_rows < 1000."
                )

            if comparison_rows < 2:
                raise RuntimeError(
                    "Insufficient non-noise overlap for ARI/AMI."
                )

            y_true = reference_labels_overlap[
                comparison_mask
            ]

            y_pred = predicted_labels[
                comparison_mask
            ]

            if len(np.unique(y_true)) < 2:
                raise RuntimeError(
                    "Quality gate failed: fewer than two reference labels."
                )

            if len(np.unique(y_pred)) < 2:
                raise RuntimeError(
                    "Quality gate failed: fewer than two resample labels."
                )

            ari = adjusted_rand_score(
                y_true,
                y_pred,
            )

            ami = adjusted_mutual_info_score(
                y_true,
                y_pred,
            )

            structure = summarize_labels(
                sample_labels
            )

            elapsed = (
                time.perf_counter()
                - run_start
            )

            rows.append(
                {
                    "resample_id": resample_id,
                    "population_size": STABILITY_POPULATION_SIZE,
                    "resample_size": resample_size,
                    "overlap_rows": int(
                        overlap_rows
                    ),
                    "comparison_rows": int(
                        comparison_rows
                    ),
                    "ari": float(ari),
                    "ami": float(ami),
                    "cluster_count": structure[
                        "cluster_count"
                    ],
                    "noise_fraction": structure[
                        "noise_fraction"
                    ],
                    "elapsed_seconds": float(
                        elapsed
                    ),
                    "error": "",
                }
            )

            print(
                f"  ARI={ari:.4f} | "
                f"AMI={ami:.4f} | "
                f"clusters={structure['cluster_count']} | "
                f"noise={structure['noise_fraction'] * 100:.2f}% | "
                f"time={elapsed:.1f}s",
                flush=True,
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - run_start
            )

            rows.append(
                {
                    "resample_id": resample_id,
                    "population_size": STABILITY_POPULATION_SIZE,
                    "resample_size": resample_size,
                    "overlap_rows": int(overlap_rows),
                    "comparison_rows": int(comparison_rows),
                    "ari": np.nan,
                    "ami": np.nan,
                    "cluster_count": 0,
                    "noise_fraction": np.nan,
                    "elapsed_seconds": float(
                        elapsed
                    ),
                    "error": str(exc),
                }
            )

            print(
                f"  ERROR: {exc}",
                flush=True,
            )

        finally:
            del model
            gc.collect()

    # -------------------------------------------------------------------------
    # Save detailed results
    # -------------------------------------------------------------------------
    detail_df = pd.DataFrame(rows)

    detail_df.to_csv(
        DETAIL_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n[4/4] Creating stability summary...")

    valid = detail_df[
        detail_df["error"].eq("")
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "configuration": (
                    "HDBSCAN resolution-matched stability fit "
                    "min_cluster_size=2500 min_samples=300"
                ),
                "full_population_size": FULL_POPULATION_SIZE,
                "stability_population_size": STABILITY_POPULATION_SIZE,
                "final_min_cluster_size": FINAL_MIN_CLUSTER_SIZE,
                "stability_min_cluster_size": STABILITY_MIN_CLUSTER_SIZE,
                "final_min_samples": FINAL_MIN_SAMPLES,
                "stability_min_samples": FINAL_MIN_SAMPLES,
                "resolution_matching_method": RESOLUTION_MATCHING_METHOD,
                "resamples_attempted": int(
                    len(detail_df)
                ),
                "resamples_valid": int(
                    len(valid)
                ),
                "valid_fraction": float(
                    len(valid) / len(detail_df)
                ) if len(detail_df) else np.nan,
                "mean_ari": float(
                    valid["ari"].mean()
                ) if not valid.empty else np.nan,
                "median_ari": float(
                    valid["ari"].median()
                ) if not valid.empty else np.nan,
                "std_ari": float(
                    valid["ari"].std()
                ) if len(valid) > 1 else np.nan,
                "min_ari": float(
                    valid["ari"].min()
                ) if not valid.empty else np.nan,
                "max_ari": float(
                    valid["ari"].max()
                ) if not valid.empty else np.nan,
                "mean_ami": float(
                    valid["ami"].mean()
                ) if not valid.empty else np.nan,
                "median_ami": float(
                    valid["ami"].median()
                ) if not valid.empty else np.nan,
                "mean_cluster_count": float(
                    valid["cluster_count"].mean()
                ) if not valid.empty else np.nan,
                "cluster_count_mode": (
                    int(
                        valid["cluster_count"]
                        .mode()
                        .iloc[0]
                    )
                    if not valid.empty
                    else np.nan
                ),
                "mean_noise_fraction": float(
                    valid["noise_fraction"].mean()
                ) if not valid.empty else np.nan,
                "final_model_cluster_count": int(
                    len(np.unique(reference_labels[reference_labels != -1]))
                ),
                "final_model_noise_fraction": float(
                    (reference_labels == -1).mean()
                ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("OBJECTIVE 1 — HDBSCAN STABILITY COMPLETED")
    print("=" * 80)

    print("\nDetailed results:")
    print(
        detail_df.to_string(
            index=False
        )
    )

    print("\nSummary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(f"  Detailed: {DETAIL_FILE}")
    print(f"  Summary : {SUMMARY_FILE}")

    print(
        "\nSTOP: Corrected Objective-1 stability evaluation completed; "
        "no downstream objectives were performed."
    )

    del X_full
    del X_population
    gc.collect()


if __name__ == "__main__":
    main()