# Assignment 2 engineering upgrade

The user approved six review recommendations. Keep the original model, selected
parameters, study cutoffs and test predictions unchanged. Do not tune on test data.

- [x] Add an in-memory, thread-safe model store, verified reload on file change,
  stable health/errors and real HTTP load measurements.
- [x] Add a separate integrity ledger: lockfile/runtime fingerprint, checksums of
  completed-stage artifacts and explicit audited adoption of legacy evidence.
  Failed partial stages remain retryable; completed evidence may not be rewritten.
- [x] Test invalid bundles, changed dependencies, artifact damage, partial retry,
  concurrent model access and registry outage. Preserve existing leakage tests.
- [x] Add descriptive missed-peak diagnostics from saved predictions and temporal
  block permutation importance on October calibration data; no new test selection.
- [x] Add a responsive, offline-capable dashboard to the existing API with actual
  artifact data, date filtering and a labelled historical replay request.
- [x] Extend Docker CI to service readiness, tracking/registry smoke test, HTTP,
  DAG imports and failure recovery. Do not claim a hosted GitHub run without one.
- [x] Update report, README, verification evidence and portable ZIP; verify Docker,
  browser desktop/mobile layout and every changed PDF page.

Design: FastAPI serves static HTML/CSS/JS and JSON evidence. No CDN, node runtime,
third-party dashboard service or additional database is necessary. Georgia display
type, compact monospaced labels, ivory background (#f6f4ed), ink (#183c49), amber
(#a96720) and restrained red (#a24431) highlight operational status and uncertainty.
