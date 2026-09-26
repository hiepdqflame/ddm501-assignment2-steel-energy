"""Load executable DAGs and verify the default locked-test confirmation boundary."""

from airflow.exceptions import AirflowSkipException
from airflow.models import DagBag

bag = DagBag(dag_folder="/app/dags", include_examples=False)
assert not bag.import_errors, bag.import_errors
research = bag.dags.get("steel_energy_research")
monitor = bag.dags.get("steel_energy_evidence_monitor")
assert research is not None and monitor is not None
assert len(research.tasks) == 8
assert research.max_active_runs == 1
assert research.schedule_interval is None
confirmation = research.get_task("confirm_locked_test")
try:
    confirmation.python_callable(params={"confirm_final": False})
except AirflowSkipException:
    pass
else:
    raise AssertionError("Default confirmation must keep the test locked")
confirmation.python_callable(params={"confirm_final": True})
print("Both DAGs imported; eight research tasks; false skips and true confirms.")
