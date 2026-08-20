"""
GPU-FinOps
==========

Stage 2: Data Cleaning and Master Analytical Dataset Construction

Purpose
-------
Convert the validated Alibaba GPU-2020 raw trace into a clean,
job-level analytical dataset suitable for:

    1. Exploratory Data Analysis
    2. Feature Engineering
    3. Workload Characterization
    4. Cost Forecasting
    5. Anomaly Detection

Design principles
-----------------
- Raw data is NEVER modified.
- Missing operational values are not blindly imputed.
- Large CSV files are processed in chunks.
- Relational integrity discovered during validation is respected.
- MACHINE_METRIC is treated as optional contextual data because
  its coverage is low.
- SENSOR utilization values above 100 are excluded from the
  primary utilization statistics but counted as quality flags.
- The final analytical dataset keeps jobs even when telemetry
  is unavailable, allowing downstream analyses to explicitly
  control for telemetry coverage.

Output
------
data/processed/gpu_finops_job_master.csv

Audit
-----
data/interim/cleaning_audit.csv

Temporary relational database
-----------------------------
data/interim/gpu_finops_clean.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alibaba_gpu_v2020"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DB_PATH = INTERIM_DIR / "gpu_finops_clean.db"

MASTER_OUTPUT = PROCESSED_DIR / "gpu_finops_job_master.csv"

AUDIT_OUTPUT = INTERIM_DIR / "cleaning_audit.csv"

CHUNKSIZE = 100_000


# ============================================================================
# RAW FILENAMES
# ============================================================================

JOB_FILE = "pai_job_table.csv"
TASK_FILE = "pai_task_table.csv"
INSTANCE_FILE = "pai_instance_table.csv"
SENSOR_FILE = "pai_sensor_table.csv"
GROUP_FILE = "pai_group_tag_table.csv"
MACHINE_SPEC_FILE = "pai_machine_spec.csv"


# ============================================================================
# SCHEMAS
# ============================================================================

JOB_COLUMNS = [
    "job_name",
    "inst_id",
    "user",
    "status",
    "start_time",
    "end_time",
]

TASK_COLUMNS = [
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
]

INSTANCE_COLUMNS = [
    "job_name",
    "task_name",
    "inst_name",
    "worker_name",
    "inst_id",
    "status",
    "start_time",
    "end_time",
    "machine",
]

SENSOR_COLUMNS = [
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
]

GROUP_COLUMNS = [
    "inst_id",
    "user",
    "gpu_type_spec",
    "group",
    "workload",
]

MACHINE_SPEC_COLUMNS = [
    "machine",
    "gpu_type",
    "cap_cpu",
    "cap_mem",
    "cap_gpu",
]


# ============================================================================
# AUDIT STORAGE
# ============================================================================

AUDIT_ROWS: list[dict] = []


def add_audit(
    stage: str,
    metric: str,
    value: float | int | str,
) -> None:
    """Store an auditable cleaning statistic."""
    AUDIT_ROWS.append(
        {
            "stage": stage,
            "metric": metric,
            "value": value,
        }
    )


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def ensure_directories() -> None:
    """Create required output directories."""

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize textual fields without inventing values."""

    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
            }
        )
    )


def numeric(series: pd.Series) -> pd.Series:
    """Safely convert a series to numeric values."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def read_chunks(
    filename: str,
    columns: list[str],
    usecols: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    """
    Read a raw Alibaba CSV in memory-safe chunks.
    """

    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required raw file does not exist:\n{path}"
        )

    return pd.read_csv(
        path,
        names=columns,
        header=None,
        usecols=usecols,
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    )


def create_connection() -> sqlite3.Connection:
    """Create the temporary relational processing database."""

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.execute(
        "PRAGMA journal_mode=WAL;"
    )

    connection.execute(
        "PRAGMA synchronous=OFF;"
    )

    connection.execute(
        "PRAGMA temp_store=MEMORY;"
    )

    connection.execute(
        "PRAGMA cache_size=-200000;"
    )

    return connection


def execute_script(
    connection: sqlite3.Connection,
    sql: str,
) -> None:
    """Execute a SQL script and commit."""
    connection.executescript(sql)
    connection.commit()


# ============================================================================
# JOB TABLE
# ============================================================================

def load_job_table(
    connection: sqlite3.Connection,
) -> None:
    """
    Load and clean JOB table.

    Rules:
    - Text fields are stripped.
    - Timestamps are numeric.
    - Missing timestamps are preserved.
    - No timestamp imputation is performed.
    """

    print("\n" + "=" * 80)
    print("CLEANING JOB TABLE")
    print("=" * 80)

    connection.execute(
        "DROP TABLE IF EXISTS job_raw"
    )

    total_rows = 0

    first_chunk = True

    for chunk in read_chunks(
        JOB_FILE,
        JOB_COLUMNS,
    ):

        total_rows += len(chunk)

        chunk["job_name"] = clean_text(
            chunk["job_name"]
        )

        chunk["inst_id"] = clean_text(
            chunk["inst_id"]
        )

        chunk["user"] = clean_text(
            chunk["user"]
        )

        chunk["status"] = clean_text(
            chunk["status"]
        )

        chunk["start_time"] = numeric(
            chunk["start_time"]
        )

        chunk["end_time"] = numeric(
            chunk["end_time"]
        )

        # Rows without job_name cannot participate in downstream
        # relational analysis.
        invalid_key_count = int(
            chunk["job_name"].isna().sum()
        )

        if invalid_key_count:
            add_audit(
                "job_cleaning",
                "rows_without_job_name",
                invalid_key_count,
            )

            chunk = chunk[
                chunk["job_name"].notna()
            ]

        chunk.to_sql(
            "job_raw",
            connection,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )

        first_chunk = False

    print(
        f"Raw JOB rows loaded: {total_rows:,}"
    )

    add_audit(
        "job_cleaning",
        "raw_rows",
        total_rows,
    )

    # Create an index for fast joins.
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_job_name
        ON job_raw(job_name)
        """
    )

    connection.commit()

    # ------------------------------------------------------------------------
    # Deduplicate job records at the logical job level.
    #
    # We select the latest available temporal record rather than filling
    # missing values.
    # ------------------------------------------------------------------------

    execute_script(
        connection,
        """
        DROP TABLE IF EXISTS job_clean;

        CREATE TABLE job_clean AS
        SELECT
            job_name,
            user,
            status,
            start_time,
            end_time
        FROM (
            SELECT
                j.*,
                ROW_NUMBER() OVER (
                    PARTITION BY job_name
                    ORDER BY
                        COALESCE(end_time, start_time, -1) DESC,
                        rowid DESC
                ) AS rn
            FROM job_raw AS j
        )
        WHERE rn = 1;
        """,
    )

    job_count = connection.execute(
        "SELECT COUNT(*) FROM job_clean"
    ).fetchone()[0]

    print(
        f"Clean logical JOB records: {job_count:,}"
    )

    add_audit(
        "job_cleaning",
        "clean_logical_jobs",
        job_count,
    )


# ============================================================================
# TASK TABLE
# ============================================================================

def load_task_table(
    connection: sqlite3.Connection,
) -> None:
    """
    Load and clean TASK table.

    Rules:
    - Text fields are normalized.
    - Resource fields are converted to numeric.
    - Negative resource values are treated as invalid and converted
      to missing values.
    - Missing plan_gpu and gpu_type are NOT imputed.
    - Missing timestamps are NOT imputed.
    """

    print("\n" + "=" * 80)
    print("CLEANING TASK TABLE")
    print("=" * 80)

    connection.execute(
        "DROP TABLE IF EXISTS task_raw"
    )

    total_rows = 0
    negative_resource_values = 0

    first_chunk = True

    for chunk in read_chunks(
        TASK_FILE,
        TASK_COLUMNS,
    ):

        total_rows += len(chunk)

        # ---------------------------------------------------------------
        # Text
        # ---------------------------------------------------------------

        chunk["job_name"] = clean_text(
            chunk["job_name"]
        )

        chunk["task_name"] = clean_text(
            chunk["task_name"]
        )

        chunk["status"] = clean_text(
            chunk["status"]
        )

        chunk["gpu_type"] = clean_text(
            chunk["gpu_type"]
        )

        # ---------------------------------------------------------------
        # Numeric
        # ---------------------------------------------------------------

        numeric_columns = [
            "inst_num",
            "start_time",
            "end_time",
            "plan_cpu",
            "plan_mem",
            "plan_gpu",
        ]

        for column in numeric_columns:
            chunk[column] = numeric(
                chunk[column]
            )

        # ---------------------------------------------------------------
        # Negative values are impossible resource values.
        # Convert them to missing instead of silently dropping rows.
        # ---------------------------------------------------------------

        for column in [
            "inst_num",
            "plan_cpu",
            "plan_mem",
            "plan_gpu",
        ]:

            negative_mask = (
                chunk[column] < 0
            )

            count = int(
                negative_mask.sum()
            )

            negative_resource_values += count

            if count:
                chunk.loc[
                    negative_mask,
                    column,
                ] = pd.NA

        # ---------------------------------------------------------------
        # Required relationship keys
        # ---------------------------------------------------------------

        invalid_key_mask = (
            chunk["job_name"].isna()
            | chunk["task_name"].isna()
        )

        invalid_key_count = int(
            invalid_key_mask.sum()
        )

        if invalid_key_count:
            add_audit(
                "task_cleaning",
                "rows_without_task_keys",
                invalid_key_count,
            )

            chunk = chunk[
                ~invalid_key_mask
            ]

        chunk.to_sql(
            "task_raw",
            connection,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )

        first_chunk = False

    print(
        f"Raw TASK rows loaded: {total_rows:,}"
    )

    print(
        "Negative resource values converted to missing: "
        f"{negative_resource_values:,}"
    )

    add_audit(
        "task_cleaning",
        "raw_rows",
        total_rows,
    )

    add_audit(
        "task_cleaning",
        "negative_resource_values",
        negative_resource_values,
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_task_job_task
        ON task_raw(job_name, task_name)
        """
    )

    connection.commit()

    # ------------------------------------------------------------------------
    # Logical TASK records.
    # ------------------------------------------------------------------------

    execute_script(
        connection,
        """
        DROP TABLE IF EXISTS task_clean;

        CREATE TABLE task_clean AS
        SELECT
            job_name,
            task_name,
            inst_num,
            status,
            start_time,
            end_time,
            plan_cpu,
            plan_mem,
            plan_gpu,
            gpu_type
        FROM (
            SELECT
                t.*,
                ROW_NUMBER() OVER (
                    PARTITION BY job_name, task_name
                    ORDER BY
                        COALESCE(end_time, start_time, -1) DESC,
                        rowid DESC
                ) AS rn
            FROM task_raw AS t
        )
        WHERE rn = 1;
        """,
    )

    task_count = connection.execute(
        "SELECT COUNT(*) FROM task_clean"
    ).fetchone()[0]

    print(
        f"Clean logical TASK records: {task_count:,}"
    )

    add_audit(
        "task_cleaning",
        "clean_logical_tasks",
        task_count,
    )


# ============================================================================
# INSTANCE AGGREGATION
# ============================================================================

def aggregate_instance_table(
    connection: sqlite3.Connection,
) -> None:
    """
    Aggregate INSTANCE records to one record per:

        (job_name, task_name, inst_id)

    This reduces the 7.5M-row instance trace to a compact execution layer.
    """

    print("\n" + "=" * 80)
    print("AGGREGATING INSTANCE TABLE")
    print("=" * 80)

    connection.execute(
        "DROP TABLE IF EXISTS instance_chunk_agg"
    )

    first_chunk = True

    total_rows = 0

    for chunk_number, chunk in enumerate(
        read_chunks(
            INSTANCE_FILE,
            INSTANCE_COLUMNS,
            usecols=[
                "job_name",
                "task_name",
                "worker_name",
                "inst_id",
                "status",
                "start_time",
                "end_time",
                "machine",
            ],
        ),
        start=1,
    ):

        total_rows += len(chunk)

        # ---------------------------------------------------------------
        # Normalize keys
        # ---------------------------------------------------------------

        for column in [
            "job_name",
            "task_name",
            "worker_name",
            "inst_id",
            "status",
            "machine",
        ]:
            chunk[column] = clean_text(
                chunk[column]
            )

        # ---------------------------------------------------------------
        # Numeric timestamps
        # ---------------------------------------------------------------

        chunk["start_time"] = numeric(
            chunk["start_time"]
        )

        chunk["end_time"] = numeric(
            chunk["end_time"]
        )

        # ---------------------------------------------------------------
        # Invalid key rows
        # ---------------------------------------------------------------

        chunk = chunk[
            chunk["job_name"].notna()
            & chunk["task_name"].notna()
            & chunk["inst_id"].notna()
        ]

        if chunk.empty:
            continue

        # ---------------------------------------------------------------
        # Derive valid row runtime.
        #
        # We do NOT impute missing times.
        # Only rows with both timestamps contribute runtime.
        # ---------------------------------------------------------------

        chunk["runtime_seconds"] = (
            chunk["end_time"]
            - chunk["start_time"]
        )

        chunk.loc[
            chunk["runtime_seconds"] < 0,
            "runtime_seconds",
        ] = pd.NA

        # ---------------------------------------------------------------
        # Aggregate per instance ID.
        # ---------------------------------------------------------------

        grouped = (
            chunk.groupby(
                [
                    "job_name",
                    "task_name",
                    "inst_id",
                ],
                dropna=False,
            )
            .agg(
                instance_rows=(
                    "inst_id",
                    "size",
                ),
                instance_start=(
                    "start_time",
                    "min",
                ),
                instance_end=(
                    "end_time",
                    "max",
                ),
                runtime_seconds=(
                    "runtime_seconds",
                    "sum",
                ),
                machine=(
                    "machine",
                    "first",
                ),
                worker_name=(
                    "worker_name",
                    "first",
                ),
            )
            .reset_index()
        )

        grouped.to_sql(
            "instance_chunk_agg",
            connection,
            if_exists="replace"
            if first_chunk
            else "append",
            index=False,
        )

        first_chunk = False

        if chunk_number % 10 == 0:
            print(
                f"Processed INSTANCE chunks: "
                f"{chunk_number:,}"
            )

    add_audit(
        "instance_cleaning",
        "raw_rows_scanned",
        total_rows,
    )

    # ------------------------------------------------------------------------
    # Merge partial aggregates.
    # ------------------------------------------------------------------------

    execute_script(
        connection,
        """
        DROP TABLE IF EXISTS instance_clean;

        CREATE TABLE instance_clean AS
        SELECT
            job_name,
            task_name,
            inst_id,
            MAX(worker_name) AS worker_name,
            MAX(machine) AS machine,
            MIN(instance_start) AS instance_start,
            MAX(instance_end) AS instance_end,
            SUM(instance_rows) AS instance_rows,
            SUM(
                CASE
                    WHEN runtime_seconds IS NOT NULL
                    THEN runtime_seconds
                    ELSE 0
                END
            ) AS runtime_seconds
        FROM instance_chunk_agg
        GROUP BY
            job_name,
            task_name,
            inst_id;
        """,
    )

    instance_count = connection.execute(
        "SELECT COUNT(*) FROM instance_clean"
    ).fetchone()[0]

    print(
        f"Logical instance records: {instance_count:,}"
    )

    add_audit(
        "instance_cleaning",
        "logical_instance_records",
        instance_count,
    )


# ============================================================================
# SENSOR AGGREGATION
# ============================================================================

def aggregate_sensor_table(
    connection: sqlite3.Connection,
) -> None:
    """
    Aggregate SENSOR records to one record per:

        (job_name, task_name, inst_id)

    Sensor GPU utilization values outside [0, 100] are not used in the
    primary utilization aggregate. Their occurrence count is retained.
    """

    print("\n" + "=" * 80)
    print("AGGREGATING SENSOR TABLE")
    print("=" * 80)

    connection.execute(
        "DROP TABLE IF EXISTS sensor_chunk_agg"
    )

    total_rows = 0
    outlier_rows = 0

    first_chunk = True

    sensor_numeric_columns = [
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

    for chunk_number, chunk in enumerate(
        read_chunks(
            SENSOR_FILE,
            SENSOR_COLUMNS,
            usecols=[
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
        ),
        start=1,
    ):

        total_rows += len(chunk)

        for column in [
            "job_name",
            "task_name",
            "worker_name",
            "inst_id",
            "machine",
            "gpu_name",
        ]:
            chunk[column] = clean_text(
                chunk[column]
            )

        for column in sensor_numeric_columns:
            chunk[column] = numeric(
                chunk[column]
            )

        # ---------------------------------------------------------------
        # Remove negative physical quantities from the primary aggregate
        # by turning them into missing values.
        # ---------------------------------------------------------------

        for column in [
            "cpu_usage",
            "avg_mem",
            "max_mem",
            "avg_gpu_wrk_mem",
            "max_gpu_wrk_mem",
            "read",
            "write",
            "read_count",
            "write_count",
        ]:

            negative_mask = (
                chunk[column] < 0
            )

            chunk.loc[
                negative_mask,
                column,
            ] = pd.NA

        # ---------------------------------------------------------------
        # GPU utilization anomaly handling
        # ---------------------------------------------------------------

        gpu_outlier_mask = (
            chunk["gpu_wrk_util"].notna()
            & (
                (chunk["gpu_wrk_util"] < 0)
                | (chunk["gpu_wrk_util"] > 100)
            )
        )

        current_outliers = int(
            gpu_outlier_mask.sum()
        )

        outlier_rows += current_outliers

        # Preserve the count as an audit field.
        chunk["gpu_util_outlier_count"] = (
            gpu_outlier_mask.astype(int)
        )

        # Do not let flagged values contaminate mean/std-type
        # utilization statistics.
        chunk.loc[
            gpu_outlier_mask,
            "gpu_wrk_util",
        ] = pd.NA

        # ---------------------------------------------------------------
        # Required relational keys.
        # ---------------------------------------------------------------

        chunk = chunk[
            chunk["job_name"].notna()
            & chunk["task_name"].notna()
            & chunk["inst_id"].notna()
        ]

        if chunk.empty:
            continue

        grouped = (
            chunk.groupby(
                [
                    "job_name",
                    "task_name",
                    "inst_id",
                ],
                dropna=False,
            )
            .agg(
                sensor_rows=(
                    "inst_id",
                    "size",
                ),
                gpu_util_sum=(
                    "gpu_wrk_util",
                    "sum",
                ),
                gpu_util_count=(
                    "gpu_wrk_util",
                    "count",
                ),
                gpu_util_max=(
                    "gpu_wrk_util",
                    "max",
                ),
                gpu_util_outlier_count=(
                    "gpu_util_outlier_count",
                    "sum",
                ),
                cpu_usage_sum=(
                    "cpu_usage",
                    "sum",
                ),
                cpu_usage_count=(
                    "cpu_usage",
                    "count",
                ),
                avg_mem_mean=(
                    "avg_mem",
                    "mean",
                ),
                max_mem_max=(
                    "max_mem",
                    "max",
                ),
                avg_gpu_mem_mean=(
                    "avg_gpu_wrk_mem",
                    "mean",
                ),
                max_gpu_mem_max=(
                    "max_gpu_wrk_mem",
                    "max",
                ),
                read_sum=(
                    "read",
                    "sum",
                ),
                write_sum=(
                    "write",
                    "sum",
                ),
                read_count_sum=(
                    "read_count",
                    "sum",
                ),
                write_count_sum=(
                    "write_count",
                    "sum",
                ),
                gpu_name=(
                    "gpu_name",
                    "first",
                ),
            )
            .reset_index()
        )

        grouped.to_sql(
            "sensor_chunk_agg",
            connection,
            if_exists="replace"
            if first_chunk
            else "append",
            index=False,
        )

        first_chunk = False

        if chunk_number % 10 == 0:
            print(
                f"Processed SENSOR chunks: "
                f"{chunk_number:,}"
            )

    add_audit(
        "sensor_cleaning",
        "raw_rows_scanned",
        total_rows,
    )

    add_audit(
        "sensor_cleaning",
        "gpu_util_outlier_rows",
        outlier_rows,
    )

    execute_script(
        connection,
        """
        DROP TABLE IF EXISTS sensor_clean;

        CREATE TABLE sensor_clean AS
        SELECT
            job_name,
            task_name,
            inst_id,

            SUM(sensor_rows) AS sensor_rows,

            SUM(gpu_util_sum) AS gpu_util_sum,
            SUM(gpu_util_count) AS gpu_util_count,
            MAX(gpu_util_max) AS gpu_util_max,

            SUM(gpu_util_outlier_count)
                AS gpu_util_outlier_count,

            SUM(cpu_usage_sum) AS cpu_usage_sum,
            SUM(cpu_usage_count) AS cpu_usage_count,

            AVG(avg_mem_mean) AS avg_mem_mean,
            MAX(max_mem_max) AS max_mem_max,

            AVG(avg_gpu_mem_mean)
                AS avg_gpu_mem_mean,

            MAX(max_gpu_mem_max)
                AS max_gpu_mem_max,

            SUM(read_sum) AS read_sum,
            SUM(write_sum) AS write_sum,

            SUM(read_count_sum)
                AS read_count_sum,

            SUM(write_count_sum)
                AS write_count_sum,

            MAX(gpu_name) AS gpu_name

        FROM sensor_chunk_agg
        GROUP BY
            job_name,
            task_name,
            inst_id;
        """,
    )

    sensor_count = connection.execute(
        "SELECT COUNT(*) FROM sensor_clean"
    ).fetchone()[0]

    print(
        f"Logical sensor records: {sensor_count:,}"
    )

    print(
        f"GPU utilization outlier rows flagged: "
        f"{outlier_rows:,}"
    )

    add_audit(
        "sensor_cleaning",
        "logical_sensor_records",
        sensor_count,
    )


# ============================================================================
# GROUP TAG
# ============================================================================

def load_group_tag(
    connection: sqlite3.Connection,
) -> None:
    """
    Create one GROUP_TAG record per inst_id.

    Missing workload labels remain missing.
    No synthetic workload labels are generated.
    """

    print("\n" + "=" * 80)
    print("CLEANING GROUP_TAG")
    print("=" * 80)

    connection.execute(
        "DROP TABLE IF EXISTS group_raw"
    )

    first_chunk = True
    total_rows = 0

    for chunk in read_chunks(
        GROUP_FILE,
        GROUP_COLUMNS,
    ):

        total_rows += len(chunk)

        for column in GROUP_COLUMNS:
            chunk[column] = clean_text(
                chunk[column]
            )

        chunk = chunk[
            chunk["inst_id"].notna()
        ]

        if chunk.empty:
            continue

        chunk.to_sql(
            "group_raw",
            connection,
            if_exists="replace"
            if first_chunk
            else "append",
            index=False,
        )

        first_chunk = False

    add_audit(
        "group_tag_cleaning",
        "raw_rows_scanned",
        total_rows,
    )

    execute_script(
        connection,
        """
        DROP TABLE IF EXISTS group_clean;

        CREATE TABLE group_clean AS
        SELECT
            inst_id,
            MAX(user) AS user,
            MAX(gpu_type_spec) AS gpu_type_spec,
            MAX("group") AS group_name,
            MAX(workload) AS workload
        FROM group_raw
        GROUP BY inst_id;
        """,
    )

    count = connection.execute(
        "SELECT COUNT(*) FROM group_clean"
    ).fetchone()[0]

    print(
        f"GROUP_TAG logical records: {count:,}"
    )

    add_audit(
        "group_tag_cleaning",
        "logical_group_records",
        count,
    )


# ============================================================================
# MACHINE SPECIFICATION
# ============================================================================

def load_machine_spec(
    connection: sqlite3.Connection,
) -> None:
    """
    Load the complete machine specification reference table.
    """

    print("\n" + "=" * 80)
    print("LOADING MACHINE SPECIFICATION")
    print("=" * 80)

    path = RAW_DIR / MACHINE_SPEC_FILE

    machine_spec = pd.read_csv(
        path,
        names=MACHINE_SPEC_COLUMNS,
        header=None,
        low_memory=False,
        encoding="utf-8",
    )

    for column in [
        "machine",
        "gpu_type",
    ]:
        machine_spec[column] = clean_text(
            machine_spec[column]
        )

    for column in [
        "cap_cpu",
        "cap_mem",
        "cap_gpu",
    ]:
        machine_spec[column] = numeric(
            machine_spec[column]
        )

    machine_spec.to_sql(
        "machine_spec",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_machine_spec_machine
        ON machine_spec(machine)
        """
    )

    connection.commit()

    count = len(machine_spec)

    print(
        f"MACHINE_SPEC records: {count:,}"
    )

    add_audit(
        "machine_spec",
        "records",
        count,
    )


# ============================================================================
# BUILD JOB-LEVEL MASTER DATASET
# ============================================================================

def build_master_dataset(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build the job-level analytical dataset from already-cleaned
    intermediate tables.

    Optimization strategy:
    - Aggregate each layer once.
    - Create explicit indexes before joins.
    - Avoid CAST() operations on join keys.
    - Keep joins on native TEXT keys.
    - Keep MACHINE_METRIC out of the core dataset because its
      validated coverage is low.
    """

    print("\n" + "=" * 80)
    print("BUILDING JOB-LEVEL MASTER DATASET")
    print("=" * 80)

    # ---------------------------------------------------------------------
    # 1. TASK → JOB SUMMARY
    # ---------------------------------------------------------------------

    print("\n[1/5] Building task summary...")

    connection.execute(
        "DROP TABLE IF EXISTS task_job_summary"
    )

    connection.execute(
        """
        CREATE TABLE task_job_summary AS
        SELECT
            job_name,

            COUNT(*) AS task_count,

            COUNT(DISTINCT task_name)
                AS unique_task_count,

            SUM(
                CASE
                    WHEN status = 'Terminated'
                    THEN 1 ELSE 0
                END
            ) AS terminated_task_count,

            SUM(
                CASE
                    WHEN status = 'Failed'
                    THEN 1 ELSE 0
                END
            ) AS failed_task_count,

            SUM(
                CASE
                    WHEN status = 'Running'
                    THEN 1 ELSE 0
                END
            ) AS running_task_count,

            SUM(
                CASE
                    WHEN status = 'Waiting'
                    THEN 1 ELSE 0
                END
            ) AS waiting_task_count,

            AVG(plan_cpu) AS plan_cpu_mean,
            MAX(plan_cpu) AS plan_cpu_max,

            AVG(plan_mem) AS plan_mem_mean,
            MAX(plan_mem) AS plan_mem_max,

            AVG(plan_gpu) AS plan_gpu_mean,
            MAX(plan_gpu) AS plan_gpu_max,

            COUNT(plan_gpu)
                AS tasks_with_plan_gpu,

            COUNT(gpu_type)
                AS tasks_with_gpu_type,

            COUNT(*) - COUNT(plan_gpu)
                AS tasks_missing_plan_gpu,

            COUNT(*) - COUNT(gpu_type)
                AS tasks_missing_gpu_type,

            COUNT(DISTINCT gpu_type)
                AS gpu_type_count,

            AVG(
                CASE
                    WHEN start_time IS NOT NULL
                     AND end_time IS NOT NULL
                     AND end_time >= start_time
                    THEN
                        (end_time - start_time) / 3600.0
                END
            ) AS task_runtime_hours_mean,

            MAX(
                CASE
                    WHEN start_time IS NOT NULL
                     AND end_time IS NOT NULL
                     AND end_time >= start_time
                    THEN
                        (end_time - start_time) / 3600.0
                END
            ) AS task_runtime_hours_max

        FROM task_clean
        GROUP BY job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_task_job_summary_job
        ON task_job_summary(job_name)
        """
    )

    connection.commit()

    # ---------------------------------------------------------------------
    # 2. INSTANCE → JOB SUMMARY
    # ---------------------------------------------------------------------

    print("[2/5] Building instance summary...")

    connection.execute(
        "DROP TABLE IF EXISTS instance_job_summary"
    )

    connection.execute(
        """
        CREATE TABLE instance_job_summary AS
        SELECT
            job_name,

            COUNT(*) AS instance_count,

            COUNT(DISTINCT task_name)
                AS tasks_with_instances,

            SUM(
                CASE
                    WHEN instance_start IS NOT NULL
                     AND instance_end IS NOT NULL
                     AND instance_end >= instance_start
                    THEN 1 ELSE 0
                END
            ) AS executable_instance_count,

            AVG(
                CASE
                    WHEN instance_start IS NOT NULL
                     AND instance_end IS NOT NULL
                     AND instance_end >= instance_start
                    THEN
                        (instance_end - instance_start) / 3600.0
                END
            ) AS instance_runtime_hours_mean,

            MAX(
                CASE
                    WHEN instance_start IS NOT NULL
                     AND instance_end IS NOT NULL
                     AND instance_end >= instance_start
                    THEN
                        (instance_end - instance_start) / 3600.0
                END
            ) AS instance_runtime_hours_max,

            SUM(
                CASE
                    WHEN instance_start IS NOT NULL
                     AND instance_end IS NOT NULL
                     AND instance_end >= instance_start
                    THEN
                        (instance_end - instance_start) / 3600.0
                    ELSE 0
                END
            ) AS instance_runtime_hours_sum,

            COUNT(DISTINCT machine)
                AS machine_count

        FROM instance_clean
        GROUP BY job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_instance_job_summary_job
        ON instance_job_summary(job_name)
        """
    )

    connection.commit()

    # ---------------------------------------------------------------------
    # 3. SENSOR → JOB SUMMARY
    # ---------------------------------------------------------------------

    print("[3/5] Building sensor summary...")

    connection.execute(
        "DROP TABLE IF EXISTS sensor_job_summary"
    )

    connection.execute(
        """
        CREATE TABLE sensor_job_summary AS
        SELECT
            job_name,

            COUNT(*) AS telemetry_instance_count,

            SUM(sensor_rows)
                AS telemetry_record_count,

            SUM(gpu_util_count)
                AS gpu_util_valid_count,

            CASE
                WHEN SUM(gpu_util_count) > 0
                THEN
                    SUM(gpu_util_sum)
                    / SUM(gpu_util_count)
            END AS gpu_util_mean,

            MAX(gpu_util_max)
                AS gpu_util_max,

            SUM(gpu_util_outlier_count)
                AS gpu_util_outlier_count,

            CASE
                WHEN SUM(cpu_usage_count) > 0
                THEN
                    SUM(cpu_usage_sum)
                    / SUM(cpu_usage_count)
            END AS cpu_usage_mean,

            AVG(avg_mem_mean)
                AS avg_mem_mean,

            MAX(max_mem_max)
                AS max_mem_max,

            AVG(avg_gpu_mem_mean)
                AS avg_gpu_mem_mean,

            MAX(max_gpu_mem_max)
                AS max_gpu_mem_max,

            SUM(read_sum)
                AS total_read,

            SUM(write_sum)
                AS total_write,

            SUM(read_count_sum)
                AS total_read_count,

            SUM(write_count_sum)
                AS total_write_count,

            COUNT(DISTINCT gpu_name)
                AS observed_gpu_type_count

        FROM sensor_clean
        GROUP BY job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_sensor_job_summary_job
        ON sensor_job_summary(job_name)
        """
    )

    connection.commit()

    # ---------------------------------------------------------------------
    # 4. GROUP TAG SUMMARY
    #
    # Keep GROUP_TAG as enrichment only.
    # workload itself remains auxiliary because of its high missingness.
    # ---------------------------------------------------------------------

    print("[4/5] Building GROUP_TAG and MACHINE_SPEC summaries...")

    connection.execute(
        "DROP TABLE IF EXISTS group_job_summary"
    )

    connection.execute(
        """
        CREATE TABLE group_job_summary AS
        SELECT
            i.job_name,

            COUNT(
                DISTINCT
                CASE
                    WHEN g.inst_id IS NOT NULL
                    THEN i.inst_id
                END
            ) AS group_tag_instance_count,

            COUNT(
                DISTINCT
                CASE
                    WHEN g.workload IS NOT NULL
                    THEN i.inst_id
                END
            ) AS workload_tag_instance_count

        FROM instance_clean AS i

        LEFT JOIN group_clean AS g
            ON i.inst_id = g.inst_id

        GROUP BY i.job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_group_job_summary_job
        ON group_job_summary(job_name)
        """
    )

    # ---------------------------------------------------------------------
    # MACHINE SPEC SUMMARY
    # ---------------------------------------------------------------------

    connection.execute(
        "DROP TABLE IF EXISTS machine_job_summary"
    )

    connection.execute(
        """
        CREATE TABLE machine_job_summary AS
        SELECT
            i.job_name,

            COUNT(*) AS instances_with_machine_reference,

            SUM(
                CASE
                    WHEN ms.machine IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS instances_with_machine_spec,

            AVG(
                CASE
                    WHEN ms.machine IS NOT NULL
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS machine_spec_coverage

        FROM instance_clean AS i

        LEFT JOIN machine_spec AS ms
            ON i.machine = ms.machine

        GROUP BY i.job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_machine_job_summary_job
        ON machine_job_summary(job_name)
        """
    )

    connection.commit()

    # ---------------------------------------------------------------------
    # 5. FINAL MASTER
    # ---------------------------------------------------------------------

    print("[5/5] Joining summary tables...")

    connection.execute(
        "DROP TABLE IF EXISTS gpu_finops_job_master"
    )

    connection.execute(
        """
        CREATE TABLE gpu_finops_job_master AS

        SELECT

            j.job_name,
            j.user,
            j.status AS job_status,
            j.start_time AS job_start_time,
            j.end_time AS job_end_time,

            /* ---------------------------------------------------------
               TASK / RESOURCE
               --------------------------------------------------------- */

            t.task_count,
            t.unique_task_count,
            t.terminated_task_count,
            t.failed_task_count,
            t.running_task_count,
            t.waiting_task_count,

            t.plan_cpu_mean,
            t.plan_cpu_max,

            t.plan_mem_mean,
            t.plan_mem_max,

            t.plan_gpu_mean,
            t.plan_gpu_max,

            t.tasks_with_plan_gpu,
            t.tasks_missing_plan_gpu,

            t.tasks_with_gpu_type,
            t.tasks_missing_gpu_type,

            t.gpu_type_count,

            t.task_runtime_hours_mean,
            t.task_runtime_hours_max,

            /* ---------------------------------------------------------
               INSTANCE / EXECUTION
               --------------------------------------------------------- */

            i.instance_count,
            i.tasks_with_instances,
            i.executable_instance_count,

            i.instance_runtime_hours_mean,
            i.instance_runtime_hours_max,
            i.instance_runtime_hours_sum,

            i.machine_count,

            /* ---------------------------------------------------------
               SENSOR / UTILIZATION
               --------------------------------------------------------- */

            s.telemetry_instance_count,
            s.telemetry_record_count,

            s.gpu_util_valid_count,
            s.gpu_util_mean,
            s.gpu_util_max,

            s.gpu_util_outlier_count,

            s.cpu_usage_mean,

            s.avg_mem_mean,
            s.max_mem_max,

            s.avg_gpu_mem_mean,
            s.max_gpu_mem_max,

            s.total_read,
            s.total_write,

            s.total_read_count,
            s.total_write_count,

            s.observed_gpu_type_count,

            /* ---------------------------------------------------------
               GROUP TAG
               --------------------------------------------------------- */

            COALESCE(
                g.group_tag_instance_count,
                0
            ) AS group_tag_instance_count,

            COALESCE(
                g.workload_tag_instance_count,
                0
            ) AS workload_tag_instance_count,

            /* ---------------------------------------------------------
               MACHINE SPEC
               --------------------------------------------------------- */

            COALESCE(
                m.machine_spec_coverage,
                0.0
            ) AS machine_spec_coverage,

            /* ---------------------------------------------------------
               TELEMETRY COVERAGE
               --------------------------------------------------------- */

            CASE
                WHEN COALESCE(i.instance_count, 0) > 0
                THEN
                    MIN(
                        1.0,
                        CAST(
                            COALESCE(
                                s.telemetry_instance_count,
                                0
                            )
                            AS REAL
                        )
                        / i.instance_count
                    )
                ELSE 0.0
            END AS telemetry_coverage,

            /* ---------------------------------------------------------
               AVAILABILITY FLAGS
               --------------------------------------------------------- */

            CASE
                WHEN COALESCE(
                    s.telemetry_instance_count,
                    0
                ) > 0
                THEN 1
                ELSE 0
            END AS has_telemetry,

            CASE
                WHEN COALESCE(
                    t.tasks_with_plan_gpu,
                    0
                ) > 0
                THEN 1
                ELSE 0
            END AS has_gpu_request,

            CASE
                WHEN COALESCE(
                    i.executable_instance_count,
                    0
                ) > 0
                THEN 1
                ELSE 0
            END AS has_execution_timing

        FROM job_clean AS j

        LEFT JOIN task_job_summary AS t
            ON j.job_name = t.job_name

        LEFT JOIN instance_job_summary AS i
            ON j.job_name = i.job_name

        LEFT JOIN sensor_job_summary AS s
            ON j.job_name = s.job_name

        LEFT JOIN group_job_summary AS g
            ON j.job_name = g.job_name

        LEFT JOIN machine_job_summary AS m
            ON j.job_name = m.job_name;
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_master_job
        ON gpu_finops_job_master(job_name)
        """
    )

    connection.commit()

    # ---------------------------------------------------------------------
    # Read compact job-level table.
    # ---------------------------------------------------------------------

    print("\nReading final job-level dataset...")

    master = pd.read_sql_query(
        """
        SELECT *
        FROM gpu_finops_job_master
        ORDER BY job_name
        """,
        connection,
    )

    print(
        f"\nFinal master dataset rows: "
        f"{len(master):,}"
    )

    print(
        f"Final master dataset columns: "
        f"{len(master.columns):,}"
    )

    add_audit(
        "master_dataset",
        "rows",
        len(master),
    )

    add_audit(
        "master_dataset",
        "columns",
        len(master.columns),
    )

    return master

# ============================================================================
# FINAL DATA QUALITY CHECK
# ============================================================================

def validate_master_dataset(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final validation before saving the analytical dataset.
    """

    print("\n" + "=" * 80)
    print("FINAL MASTER DATASET VALIDATION")
    print("=" * 80)

    print(
        f"Rows: {len(master):,}"
    )

    print(
        f"Columns: {len(master.columns):,}"
    )

    duplicate_jobs = int(
        master["job_name"].duplicated().sum()
    )

    print(
        f"Duplicate job_name values: "
        f"{duplicate_jobs:,}"
    )

    if duplicate_jobs > 0:
        raise ValueError(
            "Master dataset contains duplicate job_name values."
        )

    # ------------------------------------------------------------------------
    # Runtime sanity
    # ------------------------------------------------------------------------

    negative_runtime = int(
        (
            master["instance_runtime_hours_sum"]
            < 0
        ).sum()
    )

    print(
        f"Negative aggregate runtime: "
        f"{negative_runtime:,}"
    )

    if negative_runtime > 0:
        raise ValueError(
            "Negative runtime detected in final master dataset."
        )

    # ------------------------------------------------------------------------
    # Telemetry coverage
    # ------------------------------------------------------------------------

    telemetry_jobs = int(
        master["has_telemetry"].sum()
    )

    print(
        f"Jobs with telemetry: "
        f"{telemetry_jobs:,}"
    )

    # ------------------------------------------------------------------------
    # GPU request coverage
    # ------------------------------------------------------------------------

    gpu_request_jobs = int(
        master["has_gpu_request"].sum()
    )

    print(
        f"Jobs with GPU request information: "
        f"{gpu_request_jobs:,}"
    )

    # ------------------------------------------------------------------------
    # Execution timing coverage
    # ------------------------------------------------------------------------

    execution_jobs = int(
        master["has_execution_timing"].sum()
    )

    print(
        f"Jobs with execution timing: "
        f"{execution_jobs:,}"
    )

    return master


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    master: pd.DataFrame,
) -> None:
    """Save final master dataset and cleaning audit."""

    print("\n" + "=" * 80)
    print("SAVING CLEANED DATA")
    print("=" * 80)

    master.to_csv(
        MASTER_OUTPUT,
        index=False,
    )

    audit = pd.DataFrame(
        AUDIT_ROWS
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print(
        f"Master dataset saved to:\n"
        f"{MASTER_OUTPUT}"
    )

    print(
        f"\nCleaning audit saved to:\n"
        f"{AUDIT_OUTPUT}"
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Execute the complete cleaning pipeline."""

    print("=" * 80)
    print("GPU-FinOps | STAGE 2 — DATA CLEANING")
    print("=" * 80)

    ensure_directories()

    # ------------------------------------------------------------------------
    # Remove previous temporary processing database.
    # ------------------------------------------------------------------------

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = create_connection()

    try:

        # ---------------------------------------------------------------
        # 1. Clean JOB
        # ---------------------------------------------------------------

        load_job_table(
            connection
        )

        # ---------------------------------------------------------------
        # 2. Clean TASK
        # ---------------------------------------------------------------

        load_task_table(
            connection
        )

        # ---------------------------------------------------------------
        # 3. Aggregate INSTANCE
        # ---------------------------------------------------------------

        aggregate_instance_table(
            connection
        )

        # ---------------------------------------------------------------
        # 4. Aggregate SENSOR
        # ---------------------------------------------------------------

        aggregate_sensor_table(
            connection
        )

        # ---------------------------------------------------------------
        # 5. Clean GROUP_TAG
        # ---------------------------------------------------------------

        load_group_tag(
            connection
        )

        # ---------------------------------------------------------------
        # 6. Load MACHINE_SPEC
        # ---------------------------------------------------------------

        load_machine_spec(
            connection
        )

        # ---------------------------------------------------------------
        # 7. Build job-level master
        # ---------------------------------------------------------------

        master = build_master_dataset(
            connection
        )

        # ---------------------------------------------------------------
        # 8. Validate final dataset
        # ---------------------------------------------------------------

        master = validate_master_dataset(
            master
        )

        # ---------------------------------------------------------------
        # 9. Save outputs
        # ---------------------------------------------------------------

        save_outputs(
            master
        )

        print("\n" + "=" * 80)
        print("STAGE 2 DATA CLEANING COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(
            "\nNext stage:"
            "\n  Exploratory Data Analysis (EDA)"
            "\n  → Feature Engineering"
            "\n  → Objective 1 Workload Characterization"
        )

    finally:

        connection.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()