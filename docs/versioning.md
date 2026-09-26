# Reproducibility and release strategy

## Code

Use protected `main`, short-lived `feature/*` branches and reviewed pull requests.
Run the Docker test/lint job before merging. Tag a reviewed submission `v1.0.0`;
later fixes receive patch versions and a new evidence folder. Illustrative commands
(execute only in a repository created by the student):

```sh
git switch -c feature/energy-pipeline
git add steel_energy tests config dags Dockerfile compose.yaml requirements.lock
git commit -m "Implement reproducible steel energy pipeline"
git tag -a v1.0.0 -m "Assignment 2 reproducible submission"
git rev-parse HEAD
```

The reference evidence predates publication of this Git repository and does not
pretend these illustrative commits or tags were executed. The
pipeline records the actual Git revision if available and always hashes the
model-relevant source modules and configuration. Report prose and plots do not
change the experiment identity; data, transforms, model settings and selection
logic do. Model bundles are validated against their saved SHA-256 before loading.

## Data

Keep the unmodified UCI CSV and licence attribution. Its checksum is pinned in
`config/experiment.yaml`. A new source extract gets a new versioned snapshot and
checksum, never an in-place replacement. Record collection/arrival times for a
future live feed. The derived feature Parquet has its own SHA-256 and timestamp
policy; manifests verify it before reusing cached stages. Timestamp normalisation
is a declared transformation, not a silently repaired original field.

For a larger deployment, store snapshots under immutable object-store keys such
as `steel-energy/plant01/raw/2026-10-01/<sha256>.parquet`; a manifest or DVC pointer
in Git references each object. DVC/object-store automation is a proposed extension,
not a required external service for this bundled classroom dataset.

## Models and policies

Registry name: `steel-energy-interval-forecast`. Each completed calibration logs a
model, schema signature, dependency requirements, environment and frozen manifest
to an MLflow run, then registers an immutable model version. Alias `candidate`
points to the latest reviewed-for-analysis candidate. Serving the benchmark uses
its explicit version/checksum; a moving alias cannot silently change predictions.

No `champion` is created automatically. Promotion requires passing the declared
offline gates plus a prospective local pilot and explicit energy-manager/data-owner
approval. Alert threshold, margin and calibration pass/fail travel in the frozen
manifest; model registration alone cannot turn on operational advice. Rollback in
a future deployment restores the full model/schema/policy bundle, not just weights.

## Environment

The application resolves versions in `requirements.lock`; the Linux-only
SQLAlchemy dependency `greenlet` is additionally pinned in the Dockerfile. The
Python 3.12 base is pinned to a multi-platform OCI index digest. Airflow 2.10.5 uses
the bundled official Python 3.12 constraint file in an isolated environment.
`random_state=42`, single CPU-thread limits and fixed chronological windows improve
repeatability. Exact timing and bitwise results are not promised across CPU/BLAS
implementations. Keep runtime package versions and the Docker image identifier
with each submitted evidence set.

## Study lock and failures

Each stage writes its final JSON manifest atomically only after successful work.
One advisory file lock serialises writers on the shared local volume. A matching
completed stage is reused; changed data/source/configuration is rejected. The
frozen model checksum is checked before final evaluation. A process failure may
retry the same frozen computation without authorising tuning on test outcomes.
Incomplete development attempts can leave clearly distinct MLflow runs; the
completed summary explicitly identifies which runs generated the report.

The file lock and SQLite backend fit one-host teaching execution. Multi-host
workers require distributed locking, transactional stage coordination and a shared
database/object store. No claim of high availability is made for this deployment.
