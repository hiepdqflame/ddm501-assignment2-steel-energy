# Dataset card

## Source and licence

Steel Industry Energy Consumption, UCI Machine Learning Repository.
Sathishkumar V E, Changsun Shin and Yongyun Cho. DOI: 10.24432/C52G8C.
The source identifies the facility as DAEWOO Steel Co. Ltd, Gwangyang, South Korea.

https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated
on the source page. https://creativecommons.org/licenses/by/4.0/

Downloaded 24 September 2026. The CSV in `data/raw/` is unmodified.
SHA-256: `9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`.
Maintain creator/source attribution and indicate modifications when sharing.

## Observations and transformations

35,040 rows, 11 columns, no empty values and no duplicate literal timestamps.
The source has one zero-energy row. The CSV's final row within each day is 00:00
with that same date. The primary convention treats it as next-day midnight and
uses interval-end energy totals. This is an explicitly unconfirmed interpretation.
The sensitivity study sorts literal timestamps; it is limited to development.

Calendar encodings, shifted lags and rolling means are project-derived features.
No future fill or target-interval electrical covariate is used. Original UCI
metadata lists `Load_Type` as target; this work intentionally reframes the task
as future `Usage_kWh` regression. A trained energy model is not a Load_Type classifier.

## Intended use and limitations

Intended for reproducible teaching experiments and historical forecast evaluation.
The dataset has no meter-arrival logs, equipment constraints, operator actions,
production mix, tariffs or intervention outcomes. It cannot establish savings,
causal peak reduction or the safety of equipment control. It represents a single
historical site/year; different sites and current operations require fresh data.
No personal employee records are included, but industrial energy traces can reveal
operating patterns. A live deployment needs confidentiality and access controls.
