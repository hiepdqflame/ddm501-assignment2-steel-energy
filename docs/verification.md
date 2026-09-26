# Verification record - 25 September 2026

## Engineering upgrade verification

- Expanded suite: **30 tests passed** in Linux ARM64 Docker; Ruff passed. The
  original 17-test AMD64 check below predates this upgrade and is not presented as
  an AMD64 run of the expanded suite.
- Explicit legacy adoption recomputed feature content, all 20 pooled development
  scores across both timestamp policies, October calibration and final prediction
  parity. The separate ledger records the runtime and completed artifact hashes.
- The full staged CLI subsequently completed using sealed cached results, including
  MLflow verification and supplementary analysis. Original source identity,
  frozen model and 5,856 final predictions remain unchanged.
- HTTP integration: 24 concurrent fixture requests share one cached model load;
  malformed input returns 422; corruption returns 503; a valid new bundle recovers
  with a second load. An unreachable MLflow endpoint raises, followed by successful
  registration/download and prediction parity against the running registry.
- HTTP benchmark: 10 warm-ups, then 100 requests at each concurrency (1, 4, 8).
  Observed p95: **6.10 / 70.45 / 131.20 ms**, respectively. Each request carries
  672 readings (53,407 bytes), and all predictions match the saved replay. These
  are warm Compose-network measurements on this machine, not a production SLA.
- Descriptive missed-peak analysis: 116 intervals, 81 consecutive episodes, 108
  above the last available reading, median rise 25.47 kWh. October-only grouped
  block permutation identifies recent energy as the largest sensitivity. There is
  no refit, threshold change or feature selection from these diagnostics.
- Both DAGs import and the eight-task/confirmation checks pass in the isolated
  Airflow runtime. This adds to the earlier executed full-DAG checks below.
- Chrome/Playwright: desktop 1440x1000 and mobile 390x844; all 61 day choices,
  date change and API replay verified; zero JavaScript errors and no page overflow.
  The wide time-series chart scrolls within its own container on mobile.
- Fresh ZIP checking exposed rapid-write metadata collisions in the cache. A
  deterministic regression test now forces identical metadata; comparing the small
  manifest by content fixes version detection. The recorded benchmark uses this fix.
- The upgraded English PDF has 20 pages and also builds inside Linux Docker.
- Extracted the upgraded ZIP into a fresh directory: all 30 tests and Ruff pass
  using the extracted source in Docker. `prepare` succeeds twice, producing and
  then verifying identical 34,368-row features and a fresh integrity ledger.
  This upgrade check is separate from the original full clean-room training run.
- The CI workflow includes readiness, HTTP/registry recovery and DAG checks.
  Equivalent local checks were executed; no hosted GitHub Actions run is claimed.

Supplementary machine-readable artifacts and dashboard screenshots are under
`reference-results/engineering/`; they do not replace the original experiment.

## Executed successfully

- Docker Engine 29.4.3, Compose v2 on Apple Silicon; application runtime reports
  Linux aarch64, Python 3.12.14 and the pinned package versions in `frozen.json`.
- Built application and isolated Airflow images from the delivered Dockerfile.
- Full Docker study: ten configurations x three folds x two timestamp conventions;
  primary calibration, locked-test evaluation, figures and registry verification.
- 17 pytest contract/serving tests passed in the ARM64 image.
- Built `linux/amd64` application image and ran the same 17 tests successfully
  under Docker emulation. The full AMD64 training benchmark was not executed.
- Ruff checks passed for application, tests, DAG and utility scripts.
- Actual MLflow inventory: 42 primary runs and 40 sensitivity runs, including
  nested folds; registered model version 1 in the author's fresh registry.
- Loaded the registered model and matched 20 predictions against the local bundle.
- Started the API in Docker; HTTP `/predict` matched the saved replay prediction.
- Airflow parsed both DAGs with no import errors.
- `airflow dags test steel_energy_research 2026-09-24 --conf '{"confirm_final": true}'`
  completed all eight tasks successfully, reusing the hash-verified study outputs.
- The same DAG on 2026-09-25 with default confirmation skipped final evaluation and
  downstream tasks; the DAG completed without opening a new test computation.
- `steel_energy_evidence_monitor` integration test completed successfully.
- Extracted the submission ZIP into a clean directory, started an independent
  Compose project and empty MLflow volume, and executed the complete benchmark
  again. All 5,856 final predictions matched the reference run exactly on this
  machine (`max_prediction_difference_kwh=0.0`).
- Regenerated the report inside the Linux application container successfully.
- Final Airflow webserver responds; metadata database, scheduler and triggerer
  report healthy after the standalone PATH correction. Both DAGs import cleanly.

## Measured primary result

Selected model: E06 random forest, 200 trees, no maximum depth,
minimum samples per leaf 2. Selected baseline: E01 target lag 2.

Locked-test rows: 5,856. MAE: 4.703003712 kWh versus 7.515040984 kWh for the
baseline (37.4188% relative improvement). RMSE: 9.765632324 kWh. WAPE: 18.9083%.

Forecast gate passes. Advisory calibration fails the joint predeclared recall
and precision constraints, and test recall is 67.87%, below the 80% target.
The candidate is not approved for production and the API keeps advice disabled.

## Evidence files

See `reference-results/` for all configuration/fold metrics, predictions, manifests,
figures, HTTP parity, MLflow run inventory, AMD64 test output and Airflow logs.
The final report uses the ARM64 Docker study, not the earlier native development
diagnostic. No native-only file path is required by the submitted run commands.

The first API check exposed a startup race when requests were sent immediately
after `up -d`. The final README uses `up -d --wait api`; the check passed after
waiting for the Compose healthcheck. The original failed check is not counted as
successful verification.

The initial standalone Airflow process could run explicit CLI DAG tests but its
background webserver/scheduler could not find `airflow` on PATH. The final Docker
stage prepends `/opt/airflow-venv/bin`; the healthcheck verifies both metadata
database and scheduler status instead of merely accepting an HTTP response.
This deployment fix does not alter the model or experiment identity.

## Interpretation and remaining limits

- Airflow's `dags test` output displays an execution-date-derived `run_duration`;
  do not interpret that field as measured wall-clock training time.
- A missing-Git warning in the slim image is expected: manifests explicitly
  record that no commit exists and use content hashes. No Git revision is fabricated.
- Original warm model timing excludes network, ingestion and scheduling. The
  supplemental HTTP measurements below include transport and serving work; no
  production SLA or live-meter integration is claimed.
- The source meter's midnight convention and arrival delays remain assumptions.
- MLflow/SQLite and Airflow standalone are a classroom stack, not highly available
  infrastructure. Initial builds need network access to image/package registries.
- CI configuration is supplied; a hosted GitHub Actions run has not been claimed.
