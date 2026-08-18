"""
GPU-FinOps
==========

Stage 1: Raw Dataset Validation

This module validates the Alibaba PAI GPU Cluster Trace 2020 before
any feature engineering or ML is performed.

Stages implemented:
    1. Schema validation
    2. Data-quality validation
    3. Basic value/range validation for important fields

Important:
    - Raw files are NEVER modified.
    - Large CSV files are processed in chunks.
    - Official Alibaba .header files are used for schema.
    - No ML model is trained in this script.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alibaba_gpu_v2020"


# ============================================================================
# DATASET FILE DEFINITIONS
# ============================================================================

FILES = {
    "job": {
        "csv": "pai_job_table.csv",
        "header": "pai_job_table.header",
    },
    "task": {
        "csv": "pai_task_table.csv",
        "header": "pai_task_table.header",
    },
    "instance": {
        "csv": "pai_instance_table.csv",
        "header": "pai_instance_table.header",
    },
    "sensor": {
        "csv": "pai_sensor_table.csv",
        "header": "pai_sensor_table.header",
    },
    "group_tag": {
        "csv": "pai_group_tag_table.csv",
        "header": "pai_group_tag_table.header",
    },
    "machine_spec": {
        "csv": "pai_machine_spec.csv",
        "header": "pai_machine_spec.header",
    },
    "machine_metric": {
        "csv": "pai_machine_metric.csv",
        "header": "pai_machine_metric.header",
    },
}


# ============================================================================
# EXPECTED / IMPORTANT COLUMNS
# ============================================================================

EXPECTED_COLUMNS = {
    "job": [
        "job_name",
        "inst_id",
        "user",
        "status",
        "start_time",
        "end_time",
    ],
    "task": [
        "job_name",
        "task_name",
        "inst_num",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
        "plan_gpu",
        "gpu_type",
    ],
    "instance": [
        "job_name",
        "task_name",
        "inst_name",
        "worker_name",
        "inst_id",
        "status",
        "start_time",
        "end_time",
        "machine",
    ],
    "sensor": [
        "job_name",
        "task_name",
        "worker_name",
        "inst_id",
        "machine",
        "gpu_name",
        "cpu_usage",
        "gpu_wrk_util",
        "avg_mem",
        "max_mem",
        "avg_gpu_wrk_mem",
        "max_gpu_wrk_mem",
        "read",
        "write",
        "read_count",
        "write_count",
    ],
    "group_tag": [
        "inst_id",
        "user",
        "gpu_type_spec",
        "group",
        "workload",
    ],
    "machine_spec": [
        "machine",
        "gpu_type",
        "cap_cpu",
        "cap_mem",
        "cap_gpu",
    ],
    "machine_metric": [
        "worker_name",
        "machine",
        "start_time",
        "end_time",
        "machine_cpu_iowait",
        "machine_cpu_kernel",
        "machine_cpu_usr",
        "machine_gpu",
        "machine_load_1",
        "machine_net_receive",
        "machine_num_worker",
        "machine_cpu",
    ],
}


# ============================================================================
# GENERAL HELPER FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 78) -> None:
    """Print a consistent terminal section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def read_schema(header_path: Path) -> list[str]:
    """
    Read official column names from an Alibaba .header file.
    """
    if not header_path.exists():
        raise FileNotFoundError(
            f"Header file not found: {header_path}"
        )

    with header_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        columns = next(reader, None)

    if not columns:
        raise ValueError(
            f"Header file is empty: {header_path}"
        )

    columns = [column.strip() for column in columns if column.strip()]

    return columns


def read_first_data_row(csv_path: Path) -> list[str]:
    """
    Read only the first data row from a CSV.

    The Alibaba CSV files do not contain column names.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)

        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(
            f"CSV file is empty: {csv_path}"
        )

    return first_row


def validate_expected_columns(
    table_name: str,
    columns: list[str],
) -> None:
    """
    Compare the downloaded schema with the expected Alibaba schema.
    """
    expected = EXPECTED_COLUMNS[table_name]

    if columns != expected:
        print("\n[WARNING] Schema differs from expected definition.")

        print("Expected:")
        for index, column in enumerate(expected, start=1):
            print(f"  {index:02d}. {column}")

        print("\nFound:")
        for index, column in enumerate(columns, start=1):
            print(f"  {index:02d}. {column}")

        raise ValueError(
            f"Schema mismatch detected for table: {table_name}"
        )


# ============================================================================
# STAGE 1A — SCHEMA VALIDATION
# ============================================================================

def validate_table_schema(
    table_name: str,
    csv_filename: str,
    header_filename: str,
) -> list[str]:
    """
    Validate one table's:
        - CSV existence
        - header existence
        - schema
        - first-row column count

    Returns:
        List of validated column names.
    """

    csv_path = RAW_DIR / csv_filename
    header_path = RAW_DIR / header_filename

    print_section(f"SCHEMA VALIDATION: {table_name.upper()}", 70)

    # ------------------------------------------------------------------------
    # File existence
    # ------------------------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
        )

    if not header_path.exists():
        raise FileNotFoundError(
            f"Header file not found:\n{header_path}"
        )

    print(f"[OK] CSV    : {csv_filename}")
    print(f"[OK] HEADER : {header_filename}")

    # ------------------------------------------------------------------------
    # Read schema
    # ------------------------------------------------------------------------

    columns = read_schema(header_path)

    # ------------------------------------------------------------------------
    # Read one data row only
    # ------------------------------------------------------------------------

    first_row = read_first_data_row(csv_path)

    print(f"Schema columns   : {len(columns)}")
    print(f"First-row values : {len(first_row)}")

    # ------------------------------------------------------------------------
    # Validate counts
    # ------------------------------------------------------------------------

    if len(columns) != len(first_row):
        raise ValueError(
            f"\nColumn-count mismatch for {table_name}:\n"
            f"Schema columns = {len(columns)}\n"
            f"Data values    = {len(first_row)}"
        )

    print("[OK] Column count matches.")

    # ------------------------------------------------------------------------
    # Validate known schema
    # ------------------------------------------------------------------------

    validate_expected_columns(
        table_name=table_name,
        columns=columns,
    )

    print("[OK] Expected schema matches.")

    print("\nColumns:")
    for index, column in enumerate(columns, start=1):
        print(f"  {index:02d}. {column}")

    return columns


def validate_all_schemas() -> dict[str, list[str]]:
    """
    Validate all seven Alibaba tables.
    """
    print_section(
        "GPU-FinOps | STAGE 1A — SCHEMA VALIDATION"
    )

    schemas: dict[str, list[str]] = {}

    for table_name, files in FILES.items():
        schemas[table_name] = validate_table_schema(
            table_name=table_name,
            csv_filename=files["csv"],
            header_filename=files["header"],
        )

    print_section("SCHEMA VALIDATION COMPLETED")

    print("[OK] All seven tables passed schema validation.")

    return schemas


# ============================================================================
# STAGE 1B — CHUNKED DATA QUALITY VALIDATION
# ============================================================================

def inspect_csv_quality(
    table_name: str,
    csv_filename: str,
    columns: list[str],
    chunksize: int = 100_000,
) -> dict:
    """
    Scan one CSV in chunks and collect basic data-quality statistics.

    Returns:
        Dictionary containing:
            total_rows
            missing_counts
            status_counts
            blank_counts
    """

    csv_path = RAW_DIR / csv_filename

    print_section(
        f"DATA QUALITY CHECK: {table_name.upper()}",
        70,
    )

    total_rows = 0

    # Missing values represented as actual NaN by pandas.
    missing_counts: Counter[str] = Counter()

    # Empty-string / blank-field counts.
    blank_counts: Counter[str] = Counter()

    # Status distribution.
    status_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------------
    # Read the CSV in chunks.
    # ------------------------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            csv_path,
            names=columns,
            header=None,
            chunksize=chunksize,
            low_memory=False,
            encoding="utf-8",
        ),
        start=1,
    ):

        total_rows += len(chunk)

        # --------------------------------------------------------------------
        # Missing values
        # --------------------------------------------------------------------

        null_counts = chunk.isna().sum()

        for column, count in null_counts.items():
            missing_counts[column] += int(count)

        # --------------------------------------------------------------------
        # Blank strings
        # --------------------------------------------------------------------

        for column in columns:
            if chunk[column].dtype == "object":

                blank_count = (
                    chunk[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .eq("")
                    .sum()
                )

                blank_counts[column] += int(blank_count)

        # --------------------------------------------------------------------
        # Status values, if present
        # --------------------------------------------------------------------

        if "status" in chunk.columns:

            counts = chunk["status"].fillna("<MISSING>").value_counts()

            for status, count in counts.items():
                status_counts[str(status)] += int(count)

        # --------------------------------------------------------------------
        # Progress indicator
        # --------------------------------------------------------------------

        if chunk_number % 10 == 0:
            print(
                f"Processed {chunk_number:>5} chunks "
                f"| rows scanned: {total_rows:,}"
            )

    # ------------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------------

    print(f"\nTotal rows: {total_rows:,}")

    print("\nMissing values:")
    for column in columns:

        count = missing_counts.get(column, 0)

        percentage = (
            (count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        print(
            f"  {column:25s}"
            f"{count:12,}"
            f"  ({percentage:7.3f}%)"
        )

    print("\nBlank values:")
    for column in columns:

        count = blank_counts.get(column, 0)

        percentage = (
            (count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        print(
            f"  {column:25s}"
            f"{count:12,}"
            f"  ({percentage:7.3f}%)"
        )

    if status_counts:

        print("\nStatus distribution:")

        for status, count in sorted(
            status_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            percentage = (
                (count / total_rows) * 100
                if total_rows > 0
                else 0.0
            )

            print(
                f"  {status:20s}"
                f"{count:12,}"
                f"  ({percentage:7.3f}%)"
            )

    return {
        "total_rows": total_rows,
        "missing_counts": dict(missing_counts),
        "blank_counts": dict(blank_counts),
        "status_counts": dict(status_counts),
    }


# ============================================================================
# VALUE-RANGE VALIDATION
# ============================================================================

def validate_numeric_ranges(
    table_name: str,
    columns: list[str],
    chunksize: int = 100_000,
) -> None:
    """
    Validate basic numeric ranges for fields that are important to
    GPU-FinOps.

    This stage does NOT modify data.
    """

    csv_path = RAW_DIR / FILES[table_name]["csv"]

    print_section(
        f"VALUE/RANGE VALIDATION: {table_name.upper()}",
        70,
    )

    invalid_counts: Counter[str] = Counter()
    total_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        names=columns,
        header=None,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
    ):

        total_rows += len(chunk)

        # --------------------------------------------------------------------
        # TASK TABLE
        # --------------------------------------------------------------------

        if table_name == "task":

            numeric_columns = [
                "inst_num",
                "plan_cpu",
                "plan_mem",
                "plan_gpu",
            ]

            for column in numeric_columns:
                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

            invalid_counts["inst_num < 0"] += int(
                (chunk["inst_num"] < 0).sum()
            )

            invalid_counts["plan_cpu < 0"] += int(
                (chunk["plan_cpu"] < 0).sum()
            )

            invalid_counts["plan_mem < 0"] += int(
                (chunk["plan_mem"] < 0).sum()
            )

            invalid_counts["plan_gpu < 0"] += int(
                (chunk["plan_gpu"] < 0).sum()
            )

        # --------------------------------------------------------------------
        # SENSOR TABLE
        # --------------------------------------------------------------------

        elif table_name == "sensor":

            numeric_columns = [
                "cpu_usage",
                "gpu_wrk_util",
                "avg_mem",
                "max_mem",
                "avg_gpu_wrk_mem",
                "max_gpu_wrk_mem",
                "read",
                "write",
                "read_count",
                "write_count",
            ]

            for column in numeric_columns:
                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

            invalid_counts["gpu_wrk_util < 0"] += int(
                (chunk["gpu_wrk_util"] < 0).sum()
            )

            invalid_counts["gpu_wrk_util > 100"] += int(
                (chunk["gpu_wrk_util"] > 100).sum()
            )

            invalid_counts["cpu_usage < 0"] += int(
                (chunk["cpu_usage"] < 0).sum()
            )

            invalid_counts["avg_mem < 0"] += int(
                (chunk["avg_mem"] < 0).sum()
            )

            invalid_counts["max_mem < 0"] += int(
                (chunk["max_mem"] < 0).sum()
            )

            invalid_counts["avg_gpu_wrk_mem < 0"] += int(
                (chunk["avg_gpu_wrk_mem"] < 0).sum()
            )

            invalid_counts["max_gpu_wrk_mem < 0"] += int(
                (chunk["max_gpu_wrk_mem"] < 0).sum()
            )

        # --------------------------------------------------------------------
        # JOB TABLE
        # --------------------------------------------------------------------

        elif table_name == "job":

            start = pd.to_numeric(
                chunk["start_time"],
                errors="coerce",
            )

            end = pd.to_numeric(
                chunk["end_time"],
                errors="coerce",
            )

            invalid_counts["start_time missing"] += int(
                start.isna().sum()
            )

            invalid_counts["end_time missing"] += int(
                end.isna().sum()
            )

            invalid_counts["end_before_start"] += int(
                (end < start).sum()
            )

        # --------------------------------------------------------------------
        # INSTANCE TABLE
        # --------------------------------------------------------------------

        elif table_name == "instance":

            start = pd.to_numeric(
                chunk["start_time"],
                errors="coerce",
            )

            end = pd.to_numeric(
                chunk["end_time"],
                errors="coerce",
            )

            invalid_counts["start_time missing"] += int(
                start.isna().sum()
            )

            invalid_counts["end_time missing"] += int(
                end.isna().sum()
            )

            invalid_counts["end_before_start"] += int(
                (end < start).sum()
            )

        # --------------------------------------------------------------------
        # MACHINE METRIC TABLE
        # --------------------------------------------------------------------

        elif table_name == "machine_metric":

            start = pd.to_numeric(
                chunk["start_time"],
                errors="coerce",
            )

            end = pd.to_numeric(
                chunk["end_time"],
                errors="coerce",
            )

            invalid_counts["start_time missing"] += int(
                start.isna().sum()
            )

            invalid_counts["end_time missing"] += int(
                end.isna().sum()
            )

            invalid_counts["end_before_start"] += int(
                (end < start).sum()
            )

    print(f"Rows scanned: {total_rows:,}")

    print("\nPotential invalid values:")

    for description, count in invalid_counts.items():

        print(
            f"  {description:30s}: "
            f"{count:,}"
        )

    print(
        "\nNOTE: Potential invalid values are reported only. "
        "No rows are removed at this stage."
    )
def inspect_missingness_by_status(
    csv_filename: str,
    columns: list[str],
    chunksize: int = 100_000,
) -> None:
    """
    Analyze missing start/end times by job status.
    """

    csv_path = RAW_DIR / csv_filename

    print_section(
        "JOB MISSINGNESS BY STATUS",
        70,
    )

    results = []

    for chunk in pd.read_csv(
        csv_path,
        names=columns,
        header=None,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk["start_missing"] = chunk["start_time"].isna()
        chunk["end_missing"] = chunk["end_time"].isna()

        grouped = (
            chunk.groupby("status")[
                ["start_missing", "end_missing"]
            ]
            .sum()
            .reset_index()
        )

        results.append(grouped)

    result = (
        pd.concat(results)
        .groupby("status")
        .sum()
        .sort_values("end_missing", ascending=False)
    )

    print(result)


def inspect_task_missingness_by_status(
    csv_filename: str,
    columns: list[str],
    chunksize: int = 100_000,
) -> None:
    """
    Summarize missing task-column values by status.
    """

    csv_path = RAW_DIR / csv_filename

    print_section(
        "TASK MISSINGNESS BY STATUS",
        70,
    )

    missing_fields = [
        "inst_num",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
        "plan_gpu",
        "gpu_type",
    ]

    status_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {f"missing {field}": 0 for field in missing_fields}
    )

    for chunk in pd.read_csv(
        csv_path,
        names=columns,
        header=None,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk = chunk.copy()
        chunk["status"] = chunk["status"].fillna("<MISSING>")

        for field in missing_fields:
            grouped = (
                chunk.groupby("status", dropna=False)[field]
                .apply(lambda series: int(series.isna().sum()))
            )

            for status, count in grouped.items():
                status_totals[str(status)][f"missing {field}"] += int(count)

    if not status_totals:
        print("No status values found.")
        return

    rows = []
    for status in sorted(
        status_totals,
        key=lambda value: (value == "<MISSING>", str(value)),
    ):
        row = {"status": status}
        row.update(status_totals[status])
        rows.append(row)

    result = pd.DataFrame(
        rows,
        columns=["status"] + [f"missing {field}" for field in missing_fields],
    )

    print(result.to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run the complete Stage 1 validation pipeline.

    Current default:
        - Validate all schemas.
        - Run quality checks on the JOB table.
        - Run range validation on the JOB table.

    We deliberately begin with the smallest major table before
    scanning the multi-gigabyte instance and sensor tables.
    """

    print_section(
        "GPU-FinOps | STAGE 1 — RAW DATA VALIDATION"
    )

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print(f"\nRaw dataset directory:")
    print(f"  {RAW_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist:\n{RAW_DIR}"
        )

    # ------------------------------------------------------------------------
    # Stage 1A: Schema validation
    # ------------------------------------------------------------------------

    schemas = validate_all_schemas()

    # ------------------------------------------------------------------------
    # Stage 1B: Start with JOB table only.
    # ------------------------------------------------------------------------

    job_quality = inspect_csv_quality(
        table_name="job",
        csv_filename=FILES["job"]["csv"],
        columns=schemas["job"],
        chunksize=100_000,
    )

    inspect_missingness_by_status(
        csv_filename=FILES["job"]["csv"],
        columns=schemas["job"],
        chunksize=100_000,
    )

    # ------------------------------------------------------------------------
    # Basic range validation
    # ------------------------------------------------------------------------

    validate_numeric_ranges(
        table_name="job",
        columns=schemas["job"],
        chunksize=100_000,
    )
    # ------------------------------------------------------------------------
    # TASK data-quality validation
    # ------------------------------------------------------------------------

    task_quality = inspect_csv_quality(
        table_name="task",
        csv_filename=FILES["task"]["csv"],
        columns=schemas["task"],
        chunksize=100_000,
    )

    inspect_task_missingness_by_status(
        csv_filename=FILES["task"]["csv"],
        columns=schemas["task"],
        chunksize=100_000,
    )

    # TASK resource/timestamp validation
    validate_numeric_ranges(
        table_name="task",
        columns=schemas["task"],
        chunksize=100_000,
    )
        # ------------------------------------------------------------------------
    # INSTANCE data-quality validation
    # ------------------------------------------------------------------------

    inspect_csv_quality(
        table_name="instance",
        csv_filename=FILES["instance"]["csv"],
        columns=schemas["instance"],
        chunksize=100_000,
    )

    # INSTANCE timestamp validation
    validate_numeric_ranges(
        table_name="instance",
        columns=schemas["instance"],
        chunksize=100_000,
    )
    # ------------------------------------------------------------------------
    # SENSOR data-quality validation
    # ------------------------------------------------------------------------

    inspect_csv_quality(
        table_name="sensor",
        csv_filename=FILES["sensor"]["csv"],
        columns=schemas["sensor"],
        chunksize=100_000,
    )

    # SENSOR resource/utilization validation
    validate_numeric_ranges(
        table_name="sensor",
        columns=schemas["sensor"],
        chunksize=100_000,
    )
        # ------------------------------------------------------------------------
    # GROUP TAG validation
    # ------------------------------------------------------------------------

    inspect_csv_quality(
        table_name="group_tag",
        csv_filename=FILES["group_tag"]["csv"],
        columns=schemas["group_tag"],
        chunksize=100_000,
    )

    # ------------------------------------------------------------------------
    # MACHINE SPEC validation
    # ------------------------------------------------------------------------

    inspect_csv_quality(
        table_name="machine_spec",
        csv_filename=FILES["machine_spec"]["csv"],
        columns=schemas["machine_spec"],
        chunksize=100_000,
    )

    # ------------------------------------------------------------------------
    # MACHINE METRIC validation
    # ------------------------------------------------------------------------

    inspect_csv_quality(
        table_name="machine_metric",
        csv_filename=FILES["machine_metric"]["csv"],
        columns=schemas["machine_metric"],
        chunksize=100_000,
    )

    validate_numeric_ranges(
        table_name="machine_metric",
        columns=schemas["machine_metric"],
        chunksize=100_000,
    )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    print_section(
        "STAGE 1 INITIAL VALIDATION COMPLETED"
    )

    print("[OK] Schema validation completed.")
    print("[OK] JOB-table quality scan completed.")
    print("[OK] JOB-table range validation completed.")

    print(
        "\nNext step:"
        "\n  Review the JOB results before scanning the large "
        "INSTANCE and SENSOR tables."
    )


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()