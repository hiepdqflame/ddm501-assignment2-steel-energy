# Reference results

Measured by the author in Docker on Linux ARM64. These are not loaded as completed pipeline outputs by the teacher quick start. A fresh run trains models and creates new MLflow records. Original run IDs refer to the author's registry; the teacher's run IDs will differ.

Model weights, feature caches and registry databases are excluded to keep the submission portable. The full point predictions, metrics, frozen configuration, figures and verification logs are retained.

Historical verification files describe the version tested at that time.
For example, `report-qa.json` records the initial 17-page report and 17-test suite;
the delivered report has 20 pages and the expanded suite has 30 tests. See
`engineering/fresh-zip-check.txt` and `docs/verification.md` for the later checks.
References to ZIP reproduction describe checks already executed before the
GitHub submission format was selected; they are not instructions to submit a ZIP.
