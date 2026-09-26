# Five-minute assessor demonstration

Run the two benchmark commands in README first; the initial image download and
training are preparation, not part of this five-minute walkthrough.

## 1. Frame the task (30 seconds)

"This project forecasts the next target 15-minute electricity interval from
strictly available past readings. It implements the Assignment 1 design on a
real, licensed UCI dataset. It is a historical replay, not a live plant controller."

## 2. Show measured evidence (90 seconds)

Open http://localhost:18030. Explain MAE **4.703 kWh**, baseline **7.515 kWh** and
**37.4% reduction**. Select **2018-12-19** to inspect a difficult day: the red dots
are missed high-load intervals, not omitted observations. Explain why forecast
skill does not override the failed alert gate.

Scroll to the ten-configuration comparison. Selection uses July-September only;
October calibrates the alert margin, and November-December is the frozen test.
The permutation analysis uses October, and describes the already frozen model.

## 3. Make one real request (45 seconds)

Click **Run historical replay**. The UI sends 672 completed readings to `/predict`.
The response includes a real model prediction, registry version and disabled
production approval. `/docs` shows the strict request contract. The API caches a
verified model revision; it returns 503 if that bundle becomes unavailable.

## 4. Show lineage and orchestration (90 seconds)

Open http://localhost:15030. Inspect `steel-energy-assignment2-end_of_day`, the
configuration/fold runs and the registered candidate. A model version is a saved
artifact with lineage, not permission to deploy it.

Open http://localhost:18081 if the orchestration profile is running. Show the
eight-task research DAG and its `confirm_final` parameter. The default skips the
locked-test task and downstream tasks. The daily evidence monitor verifies frozen
outputs; it does not retrain on an already observed test period.

## 5. Close with limitations (45 seconds)

"The model meets the forecast gate, but high-load recall is 67.9%, below 80%.
The midnight interpretation, meter arrival delay and generalisation to another
plant remain unconfirmed. A prospective pilot and human approval are necessary.
I do not claim measured financial savings from a historical forecast benchmark."

For engineering evidence, show `docs/verification.md`, the 30-test result and
`reference-results/engineering/http-benchmark.json`. The bundled GitHub workflow
has local Docker verification; a hosted Actions run is not claimed.
