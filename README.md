# Domain-Classifier Drift Detector

`detect_drift` measures whether the distribution of an arbitrary tabular
dataset changed between two date windows, using a domain classifier
(classifier two-sample test). It labels window-1 rows `0` and window-2 rows
`1`, trains a LightGBM classifier to tell them apart, and reports how
separable the windows are.

- **AUC ≈ 0.5** → windows are indistinguishable → no drift.
- **Higher AUC** → more drift. Bands: `<0.6` none, `0.6–0.7` mild,
  `0.7–0.8` moderate, `≥0.8` severe.
- A **permutation test** reports a p-value for whether the AUC is meaningful.
- **`top_drivers`** lists the features the classifier relied on most
  (meaningful only when drift is significant).

## Setup

A virtual environment is already provided in `.venv/`. To recreate it:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Usage

```python
from drift_detector import detect_drift

result = detect_drift(
    df,
    date_column="event_date",
    start1="2024-01-01", end1="2024-01-31",
    start2="2024-03-01", end2="2024-03-31",
    n_permutations=100,   # significance-test iterations
)
print(result)
```

`result` is a one-row DataFrame with columns: `start_date_1`, `end_date_1`,
`start_date_2`, `end_date_2`, `n_window_1`, `n_window_2`, `n_features`,
`auc`, `drift_label`, `p_value`, `top_drivers`. Run several comparisons and
`pd.concat` them into a report table.

### Key options
- `features=[...]` — test specific columns (otherwise all columns except the
  date column are used, with constant and near-unique ID-like non-float
  columns auto-dropped).
- `n_splits=5` — stratified K-fold count for the AUC.
- `n_permutations=100` — permutation-test iterations.
- `thresholds=(0.6, 0.7, 0.8)` — drift-band cut points.
- `top_n=5` — number of drift drivers to report.

## Rolling drift over time

`detect_drift_rolling` bins the timeline into consecutive periods by a pandas
frequency (offset alias) and runs `detect_drift` across them, returning one row
per comparison:

```python
from drift_detector import detect_drift_rolling

# Each month vs the next:
report = detect_drift_rolling(df, "event_date", "MS", mode="consecutive")

# Every month vs the most recent month as the baseline:
report = detect_drift_rolling(
    df, "event_date", "MS", mode="baseline", baseline="last"
)
```

- `freq` is a pandas offset alias: `"MS"` (month start), `"W"`, `"QS"`, `"30D"`, ...
- `mode="consecutive"` compares each period to the next; `mode="baseline"`
  compares every period to one fixed reference.
- `baseline` (baseline mode): `"first"` (default), `"last"`, or an explicit
  `(start, end)` tuple. The reference stays in the window-1 columns.
- Periods with fewer than 2 rows are dropped with a warning.
- Extra keyword arguments (`features`, `n_splits`, `n_permutations`,
  `thresholds`, `top_n`, `random_state`) are forwarded to each `detect_drift`
  call.

The result is a stacked DataFrame using the same columns as `detect_drift`.

## Run the demo and tests

```bash
./.venv/Scripts/python.exe example.py
./.venv/Scripts/python.exe -m pytest -q
```
