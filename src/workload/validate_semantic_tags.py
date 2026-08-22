"""
GPU-FinOps | OBJECTIVE 1 — AUXILIARY SEMANTIC-TAG VALIDATION

Purpose
-------
Cross-reference the sparse semantic `workload` tag with the final
HDBSCAN workload clusters.

IMPORTANT:
- This is AUXILIARY / QUALITATIVE validation only.
- The semantic `workload` field is sparse (~9% coverage).
- It is NOT treated as ground-truth clustering labels.
- No clustering model is refit.
- Final HDBSCAN assignments are not modified.

Outputs
-------
results/tables/objective1_semantic_tag_cluster_crosstab.csv
results/tables/objective1_semantic_tag_validation_summary.csv
results/tables/objective1_semantic_tag_cluster_enrichment.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alibaba_gpu_v2020"

GROUP_TAG_FILE = RAW_DIR / "pai_group_tag_table.csv"
INSTANCE_FILE = RAW_DIR / "pai_instance_table.csv"

ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "objective1_hdbscan_assignments.csv"
)

CROSSTAB_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_semantic_tag_cluster_crosstab.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_semantic_tag_validation_summary.csv"
)

ENRICHMENT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_semantic_tag_cluster_enrichment.csv"
)

ALIGNMENT_AUDIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "objective1_semantic_tag_alignment_audit.csv"
)

RANDOM_STATE = 42
EXPECTED_CLUSTER_COUNT = 10


# =============================================================================
# HELPERS
# =============================================================================

def ensure_directories() -> None:
    CROSSTAB_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ALIGNMENT_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_semantic_workload_data() -> tuple[pd.DataFrame, dict[str, int | str]]:
    if not GROUP_TAG_FILE.exists() or not INSTANCE_FILE.exists():
        print("SEMANTIC TAG SOURCE NOT FOUND")
        print(f"Inspected: {GROUP_TAG_FILE}")
        print(f"Inspected: {INSTANCE_FILE}")
        raise FileNotFoundError("Semantic workload source is unavailable.")

    group_tags = pd.read_csv(
        GROUP_TAG_FILE,
        names=["inst_id", "user", "gpu_type_spec", "group", "workload"],
        header=None,
        usecols=["inst_id", "workload"],
        low_memory=False,
    )
    group_tags["inst_id"] = group_tags["inst_id"].astype("string").str.strip()
    group_tags["workload"] = group_tags["workload"].astype("string").str.strip()
    group_tags = clean_workload_tags(group_tags)
    group_tags = (
        group_tags[group_tags["inst_id"].notna()]
        .groupby("inst_id", as_index=False, dropna=False)["workload"]
        .max()
    )

    instances = pd.read_csv(
        INSTANCE_FILE,
        names=[
            "job_name", "task_name", "inst_name", "worker_name", "inst_id",
            "status", "start_time", "end_time", "machine",
        ],
        header=None,
        usecols=["job_name", "inst_id"],
        low_memory=False,
    )
    for column in ["job_name", "inst_id"]:
        instances[column] = instances[column].astype("string").str.strip()
    instances = instances.dropna(subset=["job_name", "inst_id"])

    conflicting_instances = int(
        (instances.groupby("inst_id")["job_name"].nunique() > 1).sum()
    )
    if conflicting_instances:
        raise ValueError(
            f"Instance-to-job mapping has {conflicting_instances:,} conflicting inst_id values."
        )

    instance_jobs = instances.drop_duplicates("inst_id")
    semantic_jobs = instance_jobs.merge(
        group_tags,
        on="inst_id",
        how="inner",
        validate="one_to_one",
    )
    semantic_jobs = (
        semantic_jobs.groupby("job_name", as_index=False, dropna=False)["workload"]
        .max()
    )

    duplicate_semantic_jobs = int(
        semantic_jobs["job_name"].duplicated().sum()
    )
    if duplicate_semantic_jobs:
        raise ValueError("Resolved semantic workload source is not unique by job_name.")

    return semantic_jobs, {
        "source_group_rows": int(len(group_tags)),
        "source_instance_rows": int(len(instance_jobs)),
        "source_job_rows": int(len(semantic_jobs)),
        "duplicate_key_count": duplicate_semantic_jobs,
        "conflicting_instance_count": conflicting_instances,
    }


def load_assignments() -> pd.DataFrame:
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"HDBSCAN assignments not found:\n{ASSIGNMENTS_FILE}"
        )

    df = pd.read_csv(
        ASSIGNMENTS_FILE,
        low_memory=False,
    )

    required = [
        "matrix_row_index",
        "job_name",
        "cluster_id",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing assignment columns: {missing}"
        )

    if df["job_name"].duplicated().any():
        raise ValueError(
            "HDBSCAN assignments contain duplicate job_name values."
        )

    cluster_ids = sorted(
        int(x)
        for x in df["cluster_id"].unique()
        if int(x) != -1
    )

    if len(cluster_ids) != EXPECTED_CLUSTER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLUSTER_COUNT} non-noise clusters, "
            f"found {len(cluster_ids)}."
        )

    return df


def clean_workload_tags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only meaningful semantic workload tags.

    Empty strings and common missing-value representations are removed.
    """
    out = df.copy()

    out["workload"] = (
        out["workload"]
        .astype("string")
        .str.strip()
    )

    invalid_values = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
    }

    out.loc[
        out["workload"].str.lower().isin(invalid_values),
        "workload",
    ] = pd.NA

    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print(
        "GPU-FinOps | OBJECTIVE 1 — AUXILIARY SEMANTIC-TAG VALIDATION"
    )
    print("=" * 80)

    print(
        "\nIMPORTANT:"
        "\n  This is qualitative auxiliary validation only."
        "\n  The semantic workload tag is NOT ground truth."
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading semantic workload tags from the raw source...")
    print(f"Semantic source file: {GROUP_TAG_FILE}")
    print("workload column found: YES")
    print("job_name column found: YES (via pai_instance_table.csv)")
    semantic_df, source_stats = load_semantic_workload_data()

    print("[2/6] Loading final HDBSCAN assignments...")
    assignment_df = load_assignments()

    print(
        f"Semantic job rows: {len(semantic_df):,}"
        f"\nAssignment rows : {len(assignment_df):,}"
    )

    # -------------------------------------------------------------------------
    # Job alignment
    # -------------------------------------------------------------------------
    print("\n[3/6] Joining workload tags to final HDBSCAN assignments...")

    merged = assignment_df.merge(
        semantic_df,
        on="job_name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(merged) != len(assignment_df):
        raise RuntimeError(
            "Semantic-tag join changed the assignment population."
        )

    matched_rows = int(merged["_merge"].eq("both").sum())
    unmatched_assignments = int(merged["_merge"].eq("left_only").sum())
    duplicate_assignment_keys = int(
        assignment_df["job_name"].duplicated().sum()
    )
    duplicate_semantic_keys = int(
        semantic_df["job_name"].duplicated().sum()
    )

    alignment_audit = pd.DataFrame(
        [
            {
                "semantic_source_file": str(GROUP_TAG_FILE),
                "instance_source_file": str(INSTANCE_FILE),
                "assignment_rows": len(assignment_df),
                "semantic_job_rows": len(semantic_df),
                "matched_rows": matched_rows,
                "unmatched_assignments": unmatched_assignments,
                "assignment_duplicate_key_count": duplicate_assignment_keys,
                "semantic_duplicate_key_count": duplicate_semantic_keys,
                "conflicting_instance_count": source_stats["conflicting_instance_count"],
                "join_key": "job_name",
                "duplicate_resolution": "MAX(workload) per inst_id, then MAX(workload) per job_name",
                "alignment_status": (
                    "PASS"
                    if duplicate_assignment_keys == 0
                    and duplicate_semantic_keys == 0
                    and len(merged) == len(assignment_df)
                    else "FAIL"
                ),
            }
        ]
    )
    alignment_audit.to_csv(ALIGNMENT_AUDIT_FILE, index=False)

    if (
        duplicate_assignment_keys
        or duplicate_semantic_keys
        or len(merged) != len(assignment_df)
    ):
        raise RuntimeError("Semantic-tag alignment validation failed.")

    print("SEMANTIC TAG ALIGNMENT STATUS: PASS")
    merged = merged.drop(columns="_merge")

    # Keep only non-noise jobs because semantic validation concerns
    # discovered workload profiles.
    non_noise = merged[
        merged["cluster_id"] != -1
    ].copy()

    non_noise = clean_workload_tags(
        non_noise
    )

    tagged = non_noise[
        non_noise["workload"].notna()
    ].copy()

    if tagged.empty:
        raise RuntimeError(
            "No usable semantic workload tags were found."
        )

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------
    total_jobs = len(assignment_df)
    non_noise_jobs = len(non_noise)
    tagged_non_noise_jobs = len(tagged)

    tagged_all_jobs = int(merged["workload"].notna().sum())

    coverage_all = tagged_all_jobs / total_jobs

    coverage_non_noise = (
        tagged_non_noise_jobs / non_noise_jobs
    )

    print(
        "\nSemantic-tag coverage:"
        f"\n  Total eligible jobs      : {total_jobs:,}"
        f"\n  Non-noise jobs           : {non_noise_jobs:,}"
        f"\n  Jobs with workload tag  : {tagged_all_jobs:,}"
        f"\n  Tagged non-noise jobs   : {tagged_non_noise_jobs:,}"
        f"\n  Coverage of all jobs    : {coverage_all * 100:.2f}%"
        f"\n  Coverage of non-noise   : {coverage_non_noise * 100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Crosstab
    # -------------------------------------------------------------------------
    print("\n[4/6] Building workload-tag × cluster crosstab...")

    crosstab_counts = pd.crosstab(
        tagged["workload"],
        tagged["cluster_id"],
    ).sort_index()

    crosstab_counts.to_csv(
        CROSSTAB_FILE
    )

    # Row-normalized proportions:
    # within each semantic workload tag, how are jobs distributed across
    # discovered clusters?
    row_percent = (
        crosstab_counts
        .div(
            crosstab_counts.sum(axis=1),
            axis=0,
        )
        .mul(100)
    )

    # -------------------------------------------------------------------------
    # Enrichment
    # -------------------------------------------------------------------------
    print("[5/6] Calculating qualitative cluster enrichment...")

    cluster_totals = tagged.groupby(
        "cluster_id"
    ).size()

    tag_totals = tagged.groupby(
        "workload"
    ).size()

    total_tagged = len(tagged)

    rows = []

    for workload_tag in crosstab_counts.index:
        for cluster_id in crosstab_counts.columns:
            observed = int(
                crosstab_counts.loc[
                    workload_tag,
                    cluster_id,
                ]
            )

            expected = (
                tag_totals.loc[workload_tag]
                * cluster_totals.loc[cluster_id]
                / total_tagged
            )

            enrichment = (
                observed / expected
                if expected > 0
                else np.nan
            )

            rows.append(
                {
                    "workload_tag": workload_tag,
                    "cluster_id": int(cluster_id),
                    "tag_count_in_cluster": observed,
                    "tag_total": int(
                        tag_totals.loc[workload_tag]
                    ),
                    "cluster_tagged_total": int(
                        cluster_totals.loc[cluster_id]
                    ),
                    "tag_within_cluster_percent": float(
                        observed
                        / cluster_totals.loc[cluster_id]
                        * 100
                    ),
                    "cluster_within_tag_percent": float(
                        row_percent.loc[
                            workload_tag,
                            cluster_id,
                        ]
                    ),
                    "enrichment_ratio": float(
                        enrichment
                    )
                    if np.isfinite(enrichment)
                    else np.nan,
                }
            )

    enrichment_df = pd.DataFrame(
        rows
    )

    enrichment_df.to_csv(
        ENRICHMENT_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Dominant association summary
    # -------------------------------------------------------------------------
    dominant_rows = []

    for cluster_id in sorted(
        tagged["cluster_id"].unique()
    ):
        cluster_subset = tagged[
            tagged["cluster_id"] == cluster_id
        ]

        tag_counts = (
            cluster_subset["workload"]
            .value_counts()
        )

        dominant_tag = (
            str(tag_counts.index[0])
            if len(tag_counts)
            else ""
        )

        dominant_count = (
            int(tag_counts.iloc[0])
            if len(tag_counts)
            else 0
        )

        cluster_size = len(
            cluster_subset
        )

        dominant_rows.append(
            {
                "cluster_id": int(cluster_id),
                "tagged_jobs_in_cluster": cluster_size,
                "dominant_workload_tag": dominant_tag,
                "dominant_tag_count": dominant_count,
                "dominant_tag_percentage": (
                    dominant_count
                    / cluster_size
                    * 100
                    if cluster_size
                    else np.nan
                ),
                "number_of_workload_tags": int(
                    tag_counts.nunique()
                ),
            }
        )

    dominant_df = pd.DataFrame(
        dominant_rows
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("[6/6] Writing validation summary...")

    summary_df = pd.DataFrame(
        [
            {
                "total_eligible_jobs": total_jobs,
                "non_noise_jobs": non_noise_jobs,
                "tagged_non_noise_jobs": tagged_non_noise_jobs,
                "jobs_with_workload_tag": tagged_all_jobs,
                "tag_coverage_all_jobs": coverage_all,
                "tag_coverage_non_noise_jobs": coverage_non_noise,
                "unique_workload_tags": int(
                    tagged["workload"].nunique()
                ),
                "hdbscan_cluster_count": EXPECTED_CLUSTER_COUNT,
                "validation_type": (
                    "qualitative_auxiliary_validation"
                ),
                "ground_truth_status": (
                    "NOT_GROUND_TRUTH"
                ),
                "semantic_source_file": str(GROUP_TAG_FILE),
                "semantic_tag_coverage_note": (
                    "Sparse qualitative auxiliary validation; not ground truth."
                ),
            }
        ]
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print(
        "OBJECTIVE 1 — SEMANTIC-TAG VALIDATION COMPLETED"
    )
    print("=" * 80)

    print("\nDominant workload-tag associations:")
    print(
        dominant_df.to_string(
            index=False
        )
    )

    print("\nCrosstab:")
    print(
        crosstab_counts.to_string()
    )

    print("\nSaved:")
    print(f"  Crosstab   : {CROSSTAB_FILE}")
    print(f"  Enrichment : {ENRICHMENT_FILE}")
    print(f"  Summary    : {SUMMARY_FILE}")

    print(
        "\nSTOP: Objective-1 auxiliary semantic validation completed; "
        "no clustering or downstream analyses were performed."
    )


if __name__ == "__main__":
    main()