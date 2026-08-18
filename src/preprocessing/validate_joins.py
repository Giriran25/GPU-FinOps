from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alibaba_gpu_v2020"

CHUNKSIZE = 100_000


def read_job_names() -> set[str]:
    """
    Load unique job identifiers from the JOB table.
    Only the job_name column is required.
    """

    path = RAW_DIR / "pai_job_table.csv"

    job_names: set[str] = set()

    for chunk in pd.read_csv(
        path,
        names=["job_name"],
        usecols=["job_name"],
        header=None,
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        job_names.update(
            chunk["job_name"]
            .dropna()
            .astype(str)
        )

    return job_names


def validate_task_to_job() -> None:
    """
    Check how many TASK records reference a valid JOB.
    """

    job_names = read_job_names()

    print("=" * 70)
    print("JOIN VALIDATION: TASK → JOB")
    print("=" * 70)

    task_path = RAW_DIR / "pai_task_table.csv"

    total_tasks = 0
    matched_tasks = 0
    unmatched_tasks = 0

    task_columns = [
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

    for chunk in pd.read_csv(
        task_path,
        names=task_columns,
        header=None,
        usecols=["job_name"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):

        total_tasks += len(chunk)

        matched = chunk["job_name"].astype(str).isin(job_names)

        matched_tasks += int(matched.sum())
        unmatched_tasks += int((~matched).sum())

    match_rate = (
        matched_tasks / total_tasks * 100
        if total_tasks
        else 0
    )

    print(f"\nTotal TASK records     : {total_tasks:,}")
    print(f"Matched JOB records   : {matched_tasks:,}")
    print(f"Unmatched TASK records: {unmatched_tasks:,}")
    print(f"Join coverage         : {match_rate:.4f}%")

    print("\nIntegrity status:")

    if unmatched_tasks == 0:
        print("[OK] Every TASK record maps to a JOB.")
    else:
        print(
            "[WARNING] Some TASK records do not map to a JOB."
        )


def validate_instance_to_task() -> None:
    """
    Check how many INSTANCE records map to a valid TASK.
    """

    task_path = RAW_DIR / "pai_task_table.csv"
    instance_path = RAW_DIR / "pai_instance_table.csv"

    task_columns = [
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
    instance_columns = [
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

    task_keys: set[tuple[str, str]] = set()

    for chunk in pd.read_csv(
        task_path,
        names=task_columns,
        header=None,
        usecols=["job_name", "task_name"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        valid_task_rows = chunk[["job_name", "task_name"]].dropna()
        if valid_task_rows.empty:
            continue

        task_keys.update(
            valid_task_rows.astype(str).itertuples(index=False, name=None)
        )

    print("=" * 70)
    print("JOIN VALIDATION: INSTANCE → TASK")
    print("=" * 70)

    total_instances = 0
    matched_instances = 0
    unmatched_instances = 0

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=["job_name", "task_name"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        total_instances += len(chunk)

        valid_instance_rows = chunk[["job_name", "task_name"]].dropna()

        if valid_instance_rows.empty:
            unmatched_instances += len(chunk)
            continue

        matched_in_chunk = 0
        unmatched_in_chunk = 0

        for key in valid_instance_rows.astype(str).itertuples(index=False, name=None):
            if key in task_keys:
                matched_in_chunk += 1
            else:
                unmatched_in_chunk += 1

        matched_instances += matched_in_chunk
        unmatched_instances += unmatched_in_chunk

        if len(chunk) != len(valid_instance_rows):
            unmatched_instances += len(chunk) - len(valid_instance_rows)

    match_rate = (
        matched_instances / total_instances * 100
        if total_instances
        else 0
    )

    print(f"\nTotal INSTANCE records : {total_instances:,}")
    print(f"Matched TASK records   : {matched_instances:,}")
    print(f"Unmatched INSTANCE records: {unmatched_instances:,}")
    print(f"Join coverage         : {match_rate:.4f}%")

    print("\nIntegrity status:")

    if unmatched_instances == 0:
        print("[OK] Every INSTANCE record maps to a TASK.")
    else:
        print(
            "[WARNING] Some INSTANCE records do not map to a TASK."
        )
def validate_instance_to_sensor() -> None:
    """
    Check INSTANCE → SENSOR coverage using inst_id.
    """

    instance_path = RAW_DIR / "pai_instance_table.csv"
    sensor_path = RAW_DIR / "pai_sensor_table.csv"

    instance_columns = [
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

    sensor_columns = [
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

    # ---------------------------------------------------------------
    # Build unique instance IDs from INSTANCE table
    # ---------------------------------------------------------------

    instance_ids: set[str] = set()

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=["inst_id"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        instance_ids.update(
            chunk["inst_id"]
            .dropna()
            .astype(str)
        )

    print("=" * 70)
    print("JOIN VALIDATION: SENSOR → INSTANCE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Scan SENSOR table and check inst_id coverage
    # ---------------------------------------------------------------

    total_sensor_records = 0
    matched_sensor_records = 0
    unmatched_sensor_records = 0

    sensor_inst_ids: set[str] = set()

    for chunk in pd.read_csv(
        sensor_path,
        names=sensor_columns,
        header=None,
        usecols=["inst_id"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        total_sensor_records += len(chunk)

        valid = chunk["inst_id"].dropna().astype(str)

        sensor_inst_ids.update(valid)

        matched = valid.isin(instance_ids)

        matched_sensor_records += int(matched.sum())
        unmatched_sensor_records += int((~matched).sum())

    coverage = (
        matched_sensor_records / total_sensor_records * 100
        if total_sensor_records
        else 0
    )

    print(f"\nTotal SENSOR records     : {total_sensor_records:,}")
    print(f"Matched INSTANCE records : {matched_sensor_records:,}")
    print(
        f"Unmatched SENSOR records : "
        f"{unmatched_sensor_records:,}"
    )
    print(f"Join coverage             : {coverage:.4f}%")

    # ---------------------------------------------------------------
    # How many unique INSTANCE IDs have sensor telemetry?
    # ---------------------------------------------------------------

    matched_instance_ids = instance_ids.intersection(
        sensor_inst_ids
    )

    instance_sensor_coverage = (
        len(matched_instance_ids) / len(instance_ids) * 100
        if instance_ids
        else 0
    )

    print(
        f"\nUnique INSTANCE IDs       : {len(instance_ids):,}"
    )
    print(
        f"INSTANCE IDs with SENSOR : "
        f"{len(matched_instance_ids):,}"
    )
    print(
        f"INSTANCE coverage         : "
        f"{instance_sensor_coverage:.4f}%"
    )

    if unmatched_sensor_records == 0:
        print(
            "\n[OK] Every SENSOR record maps to a valid INSTANCE."
        )
    else:
        print(
            "\n[WARNING] Some SENSOR records do not map "
            "to an INSTANCE."
        )

def validate_instance_sensor_composite_key() -> None:
    """
    Validate INSTANCE ↔ SENSOR using a composite key:
    (job_name, task_name, inst_id, worker_name).
    """

    instance_path = RAW_DIR / "pai_instance_table.csv"
    sensor_path = RAW_DIR / "pai_sensor_table.csv"

    instance_columns = [
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

    sensor_columns = [
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

    # ---------------------------------------------------------------
    # Build unique composite keys from INSTANCE
    # ---------------------------------------------------------------

    instance_keys: set[tuple[str, str, str, str]] = set()

    # Map (job_name, task_name, inst_id) → worker_name values
    instance_worker_map: dict[
        tuple[str, str, str],
        set[str]
    ] = {}

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=[
            "job_name",
            "task_name",
            "inst_id",
            "worker_name",
        ],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):

        valid = chunk.dropna(
            subset=[
                "job_name",
                "task_name",
                "inst_id",
                "worker_name",
            ]
        )

        for row in valid.itertuples(index=False, name=None):
            job_name, task_name, inst_id, worker_name = map(
                str,
                row,
            )

            composite_key = (
                job_name,
                task_name,
                inst_id,
                worker_name,
            )

            instance_keys.add(composite_key)

            base_key = (
                job_name,
                task_name,
                inst_id,
            )

            instance_worker_map.setdefault(
                base_key,
                set(),
            ).add(worker_name)

    # ---------------------------------------------------------------
    # Scan SENSOR and compare composite keys
    # ---------------------------------------------------------------

    sensor_keys: set[tuple[str, str, str, str]] = set()

    matched_sensor_keys: set[tuple[str, str, str, str]] = set()

    unmatched_sensor_keys: set[tuple[str, str, str, str]] = set()

    sensor_worker_map: dict[
        tuple[str, str, str],
        set[str]
    ] = {}

    total_sensor_rows = 0

    for chunk in pd.read_csv(
        sensor_path,
        names=sensor_columns,
        header=None,
        usecols=[
            "job_name",
            "task_name",
            "inst_id",
            "worker_name",
        ],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):

        valid = chunk.dropna(
            subset=[
                "job_name",
                "task_name",
                "inst_id",
                "worker_name",
            ]
        )

        total_sensor_rows += len(chunk)

        for row in valid.itertuples(index=False, name=None):
            job_name, task_name, inst_id, worker_name = map(
                str,
                row,
            )

            composite_key = (
                job_name,
                task_name,
                inst_id,
                worker_name,
            )

            sensor_keys.add(composite_key)

            if composite_key in instance_keys:
                matched_sensor_keys.add(composite_key)
            else:
                unmatched_sensor_keys.add(composite_key)

            base_key = (
                job_name,
                task_name,
                inst_id,
            )

            sensor_worker_map.setdefault(
                base_key,
                set(),
            ).add(worker_name)

    # ---------------------------------------------------------------
    # Calculate coverage
    # ---------------------------------------------------------------

    total_unique_sensor_keys = len(sensor_keys)
    matched_unique_sensor_keys = len(matched_sensor_keys)
    unmatched_unique_sensor_keys = len(unmatched_sensor_keys)

    coverage = (
        matched_unique_sensor_keys
        / total_unique_sensor_keys
        * 100
        if total_unique_sensor_keys
        else 0
    )

    # ---------------------------------------------------------------
    # Check worker-name consistency
    # ---------------------------------------------------------------

    inconsistent_worker_keys = 0

    common_base_keys = (
        set(instance_worker_map.keys())
        & set(sensor_worker_map.keys())
    )

    for key in common_base_keys:

        instance_workers = instance_worker_map[key]
        sensor_workers = sensor_worker_map[key]

        if not sensor_workers.issubset(instance_workers):
            inconsistent_worker_keys += 1

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print("=" * 70)
    print("JOIN VALIDATION: INSTANCE ↔ SENSOR COMPOSITE KEY")
    print("=" * 70)

    print(
        f"\nUnique INSTANCE composite keys : "
        f"{len(instance_keys):,}"
    )

    print(
        f"Unique SENSOR composite keys   : "
        f"{total_unique_sensor_keys:,}"
    )

    print(
        f"Matched SENSOR composite keys  : "
        f"{matched_unique_sensor_keys:,}"
    )

    print(
        f"Unmatched SENSOR composite keys: "
        f"{unmatched_unique_sensor_keys:,}"
    )

    print(
        f"Composite-key coverage         : "
        f"{coverage:.4f}%"
    )

    print(
        f"\nWorker-name inconsistent keys  : "
        f"{inconsistent_worker_keys:,}"
    )

    print("\nIntegrity status:")

    if unmatched_unique_sensor_keys == 0:
        print(
            "[OK] Every SENSOR composite key maps "
            "to an INSTANCE composite key."
        )
    else:
        print(
            "[WARNING] Some SENSOR composite keys do not "
            "map to INSTANCE composite keys."
        )

    if inconsistent_worker_keys == 0:
        print(
            "[OK] No worker-name inconsistencies detected."
        )
    else:
        print(
            "[WARNING] Worker-name inconsistencies detected."
        )
def validate_instance_to_group_tag() -> None:
    """
    Validate INSTANCE -> GROUP_TAG coverage using inst_id.
    """

    instance_path = RAW_DIR / "pai_instance_table.csv"
    group_path = RAW_DIR / "pai_group_tag_table.csv"

    instance_columns = [
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

    group_columns = [
        "inst_id",
        "user",
        "gpu_type_spec",
        "group",
        "workload",
    ]

    group_inst_ids: set[str] = set()

    for chunk in pd.read_csv(
        group_path,
        names=group_columns,
        header=None,
        usecols=["inst_id"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        group_inst_ids.update(
            chunk["inst_id"].dropna().astype(str)
        )

    total_instances = 0
    matched_instance_rows = 0
    instance_ids: set[str] = set()

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=["inst_id"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        valid = chunk["inst_id"].dropna().astype(str)

        total_instances += len(chunk)
        instance_ids.update(valid)

        matched_instance_rows += int(
            valid.isin(group_inst_ids).sum()
        )

    row_coverage = (
        matched_instance_rows / total_instances * 100
        if total_instances
        else 0
    )

    matched_instance_ids = instance_ids & group_inst_ids

    unique_coverage = (
        len(matched_instance_ids) / len(instance_ids) * 100
        if instance_ids
        else 0
    )

    print("=" * 70)
    print("JOIN VALIDATION: INSTANCE → GROUP_TAG")
    print("=" * 70)

    print(f"\nTotal INSTANCE rows       : {total_instances:,}")
    print(f"Matched INSTANCE rows    : {matched_instance_rows:,}")
    print(f"Row-level coverage       : {row_coverage:.4f}%")

    print(f"\nUnique INSTANCE IDs       : {len(instance_ids):,}")
    print(f"IDs with GROUP_TAG        : {len(matched_instance_ids):,}")
    print(f"Unique-ID coverage        : {unique_coverage:.4f}%")
def validate_instance_to_machine_spec() -> None:
    """
    Validate INSTANCE -> MACHINE_SPEC coverage using machine.
    """

    instance_path = RAW_DIR / "pai_instance_table.csv"
    machine_spec_path = RAW_DIR / "pai_machine_spec.csv"

    instance_columns = [
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

    machine_columns = [
        "machine",
        "gpu_type",
        "cap_cpu",
        "cap_mem",
        "cap_gpu",
    ]

    machine_ids: set[str] = set()

    for chunk in pd.read_csv(
    machine_spec_path,
    names=machine_columns,
    header=None,
    usecols=[0],
    chunksize=CHUNKSIZE,
    low_memory=False,
    encoding="utf-8",
):
     machine_values = chunk.iloc[:, 0].dropna().astype(str)
    machine_ids.update(machine_values)

    total_instances = 0
    matched_instances = 0
    instance_machine_ids: set[str] = set()

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=["machine"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        valid = chunk["machine"].dropna().astype(str)

        total_instances += len(chunk)
        instance_machine_ids.update(valid)

        matched_instances += int(
            valid.isin(machine_ids).sum()
        )

    row_coverage = (
        matched_instances / total_instances * 100
        if total_instances
        else 0
    )

    matched_machine_ids = instance_machine_ids & machine_ids

    unique_coverage = (
        len(matched_machine_ids)
        / len(instance_machine_ids)
        * 100
        if instance_machine_ids
        else 0
    )

    print("=" * 70)
    print("JOIN VALIDATION: INSTANCE → MACHINE_SPEC")
    print("=" * 70)

    print(f"\nTotal INSTANCE rows       : {total_instances:,}")
    print(f"Matched INSTANCE rows    : {matched_instances:,}")
    print(f"Row-level coverage       : {row_coverage:.4f}%")

    print(
        f"\nUnique INSTANCE machines : "
        f"{len(instance_machine_ids):,}"
    )
    print(
        f"Machines in MACHINE_SPEC : "
        f"{len(matched_machine_ids):,}"
    )
    print(
        f"Machine-ID coverage      : "
        f"{unique_coverage:.4f}%"
    )

def validate_instance_to_machine_metric() -> None:
    """
    Validate INSTANCE/WORKER -> MACHINE_METRIC coverage using:
    (worker_name, machine)
    """

    instance_path = RAW_DIR / "pai_instance_table.csv"
    metric_path = RAW_DIR / "pai_machine_metric.csv"

    instance_columns = [
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

    metric_columns = [
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
    ]

    metric_keys: set[tuple[str, str]] = set()

    for chunk in pd.read_csv(
        metric_path,
        names=metric_columns,
        header=None,
        usecols=["worker_name", "machine"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        valid = chunk.dropna(
            subset=["worker_name", "machine"]
        )

        metric_keys.update(
            valid.astype(str)
            .itertuples(index=False, name=None)
        )

    total_instance_rows = 0
    matched_instance_rows = 0
    instance_keys: set[tuple[str, str]] = set()

    for chunk in pd.read_csv(
        instance_path,
        names=instance_columns,
        header=None,
        usecols=["worker_name", "machine"],
        chunksize=CHUNKSIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        valid = chunk.dropna(
            subset=["worker_name", "machine"]
        )

        total_instance_rows += len(chunk)

        instance_keys.update(
            valid.astype(str)
            .itertuples(index=False, name=None)
        )

        matched_instance_rows += sum(
            key in metric_keys
            for key in valid.astype(str)
            .itertuples(index=False, name=None)
        )

    row_coverage = (
        matched_instance_rows / total_instance_rows * 100
        if total_instance_rows
        else 0
    )

    matched_keys = instance_keys & metric_keys

    unique_coverage = (
        len(matched_keys) / len(instance_keys) * 100
        if instance_keys
        else 0
    )

    print("=" * 70)
    print("JOIN VALIDATION: INSTANCE/WORKER → MACHINE_METRIC")
    print("=" * 70)

    print(
        f"\nTotal INSTANCE rows          : "
        f"{total_instance_rows:,}"
    )

    print(
        f"Matched INSTANCE rows       : "
        f"{matched_instance_rows:,}"
    )

    print(
        f"Row-level coverage          : "
        f"{row_coverage:.4f}%"
    )

    print(
        f"\nUnique INSTANCE worker/machine keys : "
        f"{len(instance_keys):,}"
    )

    print(
        f"Keys with MACHINE_METRIC            : "
        f"{len(matched_keys):,}"
    )

    print(
        f"Unique-key coverage                  : "
        f"{unique_coverage:.4f}%"
    )

if __name__ == "__main__":
    validate_task_to_job()
    validate_instance_to_task()
    validate_instance_to_sensor()
    validate_instance_sensor_composite_key()
    validate_instance_to_group_tag()
    validate_instance_to_machine_spec()
    validate_instance_to_machine_metric()