# Assignment 2 implementation plan

**Goal:** Deliver a working, reproducible forecasting project and an English PDF
report backed by measured results for Do Quang Hiep (25MS13293).

**Spec:** `../../assignment1-steel-energy/report/report.md` and
`../../INDIVIDUAL ASSIGNMENT 2.pdf`.

**Architecture:** Pure, shared validation/features/metrics modules underpin a
staged CLI. Development and calibration precede a frozen final evaluation.
MLflow records all experiments; Airflow invokes the same commands in its own
runtime. A Docker Compose stack is the primary lecturer interface; a replay API demonstrates the saved model contract.

**Stack:** Python 3.12, pandas, scikit-learn, MLflow, Airflow, pytest,
ReportLab and Docker configuration. Local SQLite supports the single-user demo.

## Constraints

- Preserve Assignment 1 and the raw UCI file; cite CC BY 4.0.
- Predict target row j from data no later than j-2.
- Three expanding folds in July-September; calibrate alerts in October.
- Lock November-December until experiment, model and policy selection is frozen.
- Same eligible rows for all ten configurations; one CPU thread and seed 42.
- No invented metrics, business savings, production deployment or passing gates.
- Document the midnight interval-end assumption and run literal-time sensitivity.

## Task 1: Temporal data contract

Files: `steel_energy/data.py`, `steel_energy/metrics.py`, `tests/test_contract.py`.
Interfaces: `load_meter(path, policy) -> DataFrame`,
`build_features(frame) -> DataFrame`, `temporal_split(frame, start, end)`.

- [x] Write tests using hand-checked timestamps, lag values, future perturbations,
  negative/duplicate/missing rows and zero metric denominators.
- [x] Run `python -m pytest tests/test_contract.py` and confirm failure first.
- [x] Implement validation, source-order normalisation, lagged rolling features
  and training-label availability cutoffs. Verify all contract tests pass.

## Task 2: Measured, tracked experiments

Files: `config/experiment.yaml`, `steel_energy/experiments.py`,
`steel_energy/cli.py`, `steel_energy/config.py`, `steel_energy/tracking.py`.

- [x] Write selection and release-gate tests with fixed metric examples.
- [x] Run ten configurations across three folds; log parameters, fold metrics,
  predictions and source/environment provenance to real MLflow SQLite storage.
- [x] Freeze best baseline and candidate on pooled development MAE, refit through
  September with boundary purging, calibrate October margin and freeze manifest.
- [x] Evaluate the locked test once. Save all point predictions, paired daily
  bootstrap intervals, subgroups and honest release-gate outcomes.
- [x] Compare timestamp conventions on development only and export plots.

## Task 3: Operational verification

Files: `dags/steel_energy_training.py`, `steel_energy/serving.py`,
`Dockerfile`, `Dockerfile.airflow`, `compose.yaml`, `README.md`.

- [x] Implement CLI prediction with schema validation and model/policy identity.
- [x] Test serialization, real registry loading and future-input rejection.
- [x] Execute an Airflow verification DAG against frozen evidence, with separate
  manual research DAG for experiments and an explicit final-test confirmation.
- [x] Pin resolved environment versions; validate Compose and Docker build if
  the local daemon is available. Record any unverified deployment boundaries.

## Task 4: Report and submission package

Files: `report/`, `artifacts/`, `docs/verification.md`, final PDF in `../output/pdf/`.

- [x] Write report covering all five weighted sections using generated evidence,
  actual implementation snippets, architecture/DAG figures and citations.
- [x] Run full tests, lint and reproducibility checks; verify MLflow records.
- [x] Render and inspect every PDF page; verify identity, citations and numbers.
- [x] Deliver report, project, data provenance and run instructions.
