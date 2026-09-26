"""Explicit research/reproduction DAG; the historical holdout is never scheduled."""

import logging
import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

ROOT = os.getenv("PROJECT_ROOT", "/app")
PYTHON = os.getenv("CORE_PYTHON", "/usr/local/bin/python")


def log_failure(context: dict) -> None:
    """Emit an actionable alert record; a real deployment routes it to its pager."""
    logging.error(
        "PIPELINE_FAILURE dag=%s task=%s run=%s log=%s",
        context["dag"].dag_id,
        context["task_instance"].task_id,
        context["run_id"],
        context["task_instance"].log_url,
    )


def require_final_confirmation(**context) -> None:
    """Avoid opening a historical holdout through an unattended scheduled run."""
    if not context["params"]["confirm_final"]:
        raise AirflowSkipException(
            "Final test remains locked; explicitly set confirm_final=true to reproduce"
        )


with DAG(
    "steel_energy_research",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={"confirm_final": Param(False, type="boolean")},
    default_args={
        "owner": "25MS13293",
        "retries": 1,
        "retry_delay": timedelta(seconds=15),
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": log_failure,
    },
    tags=["assignment2", "reproducible", "historical-benchmark"],
) as research_dag:

    def stage(task_id: str, arguments: str) -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f'cd "{ROOT}" && "{PYTHON}" -m steel_energy.cli {arguments}',
        )

    prepare = stage("ingest_validate_features", "prepare")
    develop = stage("ten_configuration_development", "develop")
    sensitivity = stage("timestamp_sensitivity", "develop --policy literal")
    calibrate = stage("freeze_model_and_advisory", "calibrate")
    approval = PythonOperator(
        task_id="confirm_locked_test", python_callable=require_final_confirmation
    )
    final = stage("locked_test_evaluation", "final --confirm-final")
    plots = stage("export_figures", "plots")
    verify = stage("verify_registry_and_serving", "verify")
    (
        prepare
        >> develop
        >> sensitivity
        >> calibrate
        >> approval
        >> final
        >> plots
        >> verify
    )


with DAG(
    "steel_energy_evidence_monitor",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="0 6 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=15),
        "on_failure_callback": log_failure,
    },
    tags=["assignment2", "monitoring"],
) as monitor_dag:
    BashOperator(
        task_id="verify_frozen_evidence",
        execution_timeout=timedelta(minutes=5),
        bash_command=f'cd "{ROOT}" && "{PYTHON}" -m steel_energy.cli verify',
    )
