# Domain-Classifier Drift Detector — Design Spec

**Date:** 2026-05-20
**Status:** Approved design, pending spec review

## 1. Purpose

Provide a single function, `detect_drift`, that measures whether the
distribution of an arbitrary tabular dataset has changed between two date
windows. It uses the **domain-classifier** approach (a.k.a. classifier
two-sample test, C2ST):

- Pool the rows from both windows.
- Label window-1 rows `0` and window-2 rows `1`.
- Train a classifier to distinguish the two windows.
- If the two periods come from the same distribution, the classifier cannot
  beat chance (ROC-AUC ≈ 0.5). The more separable the windows are, the more
  the data has drifted.
- A permutation test establishes whether the observed AUC is meaningful or
  could have arisen by chance.

The function must work on any input table with mixed column types (numeric,
categorical, missing values) without per-dataset configuration.

## 2. Public API

```python
detect_drift(
    df,                    # pandas.DataFrame
    date_column,           # str: name of the column holding dates/timestamps
    start1, end1,          # window 1 bounds (anything pd.to_datetime accepts)
    start2, end2,          # window 2 bounds
    *,
    features=None,         # optional explicit feature list; default = all columns except date_column
    n_splits=5,            # stratified K-fold count for the AUC estimate
    n_permutations=100,    # permutation-test iterations (configurable)
    thresholds=(0.6, 0.7, 0.8),  # (mild, moderate, severe) AUC cut points
    top_n=5,               # number of drift drivers to report
    random_state=0,
) -> pandas.DataFrame      # one row
```

### Argument notes
- `date_column` is **always excluded** from the features (otherwise the
  classifier separates windows trivially by date).
- `features=None` → use all columns except `date_column`, then auto-drop
  ID-like columns (see §3.2). A caller-supplied `features` list is used
  verbatim (still excluding `date_column` if present) and bypasses the
  auto-drop heuristic.
- Date bounds are **inclusive** on both ends and compared as timestamps
  after `pd.to_datetime`. Note: a date-only bound like `"2024-01-31"`
  parses to midnight, so timestamps later that day fall outside the window.
  This is documented behavior, not a bug.

## 3. Pipeline

### 3.1 Window slicing
1. Convert `df[date_column]` and the four bounds via `pd.to_datetime`.
2. Window 1 = rows where `start1 <= date <= end1`; window 2 likewise.
3. Concatenate into a working frame `X` with label vector `y`
   (0 = window 1, 1 = window 2).
4. If the two windows overlap in time, emit a `UserWarning` but proceed.

### 3.2 Feature selection
1. Start from all columns except `date_column` (or the caller's `features`).
2. When auto-selecting, **drop near-unique ID-like columns**: any column
   where `n_unique / n_non_null > 0.95` (e.g. row IDs, UUIDs, raw
   timestamps) that would let the classifier separate windows trivially.
   Constant columns (single unique value) are also dropped as uninformative.
3. Cast `object`/`string`/`bool` columns to pandas `category` dtype so
   **LightGBM** uses its native categorical handling. Numeric columns are
   passed through as-is; LightGBM handles `NaN` natively. No imputation,
   scaling, or one-hot encoding pipeline is needed.

### 3.3 AUC via stratified K-fold cross-validation
1. Use `StratifiedKFold(n_splits, shuffle=True, random_state)`.
2. For each fold, fit a `LGBMClassifier` on the training portion and
   predict probabilities on the held-out fold.
3. Concatenate the out-of-fold predicted probabilities and compute a single
   `roc_auc_score(y, oof_proba)`. This is the reported `auc` — an honest
   out-of-sample estimate that uses all rows.

### 3.4 Drift interpretation
Given `thresholds = (t1, t2, t3)` defaulting to `(0.6, 0.7, 0.8)`:

| AUC range        | `drift_label` |
|------------------|---------------|
| `auc < t1`       | `none`        |
| `t1 <= auc < t2` | `mild`        |
| `t2 <= auc < t3` | `moderate`    |
| `auc >= t3`      | `severe`      |

(AUC is symmetric around 0.5; values below 0.5 are treated as `none`.)

### 3.5 Permutation test
1. Repeat `n_permutations` times: shuffle `y`, recompute the K-fold AUC the
   same way as §3.3.
2. p-value = `(#{permuted AUC >= observed AUC} + 1) / (n_permutations + 1)`.
   The `+1` correction keeps the p-value strictly positive.
3. A low p-value (e.g. `< 0.05`) means the observed separability is unlikely
   under the null hypothesis of "no drift".

### 3.6 Drift drivers
1. Average each fold model's **gain-based** feature importance across the K
   fold models trained in §3.3 (no extra training needed).
2. Normalize the averaged importances to sum to 1 (shares).
3. Sort descending and take the top `top_n` as a list of dicts:
   `[{"feature": name, "importance": share}, ...]`.
4. **Caveat (documented in code + output):** drivers are only meaningful
   when drift is significant (high AUC / low p-value). At AUC ≈ 0.5 the
   importances are noise. They are always reported; interpretation is left
   to the caller.

## 4. Output

A **one-row** `pandas.DataFrame` (so multiple comparisons can be
`pd.concat`-ed into a report table):

| column          | type     | meaning                                   |
|-----------------|----------|-------------------------------------------|
| `start_date_1`  | Timestamp| parsed window-1 start                     |
| `end_date_1`    | Timestamp| parsed window-1 end                       |
| `start_date_2`  | Timestamp| parsed window-2 start                     |
| `end_date_2`    | Timestamp| parsed window-2 end                       |
| `n_window_1`    | int      | rows in window 1                          |
| `n_window_2`    | int      | rows in window 2                          |
| `n_features`    | int      | features actually used after auto-drop    |
| `auc`           | float    | cross-validated ROC-AUC                    |
| `drift_label`   | str      | none / mild / moderate / severe           |
| `p_value`       | float    | permutation-test p-value                  |
| `top_drivers`   | object   | list of `{feature, importance}` dicts     |

## 5. Edge cases & errors

| Condition                                   | Behavior                              |
|---------------------------------------------|---------------------------------------|
| Either window has 0 rows                    | `ValueError` with a clear message     |
| Only one class present after slicing        | `ValueError`                          |
| `date_column` not in `df`                   | `ValueError`                          |
| Smallest class smaller than `n_splits`      | Auto-reduce `n_splits` (min 2); warn  |
| Fewer than 2 samples per class total        | `ValueError` (cannot cross-validate)  |
| All features dropped as ID-like / constant  | `ValueError` suggesting `features=`   |
| Windows overlap in time                     | `UserWarning`, proceed                |
| Unparseable date bound                      | Propagate `pd.to_datetime` error      |

## 6. Deliverables

- `drift_detector.py` — the `detect_drift` function and small private
  helpers (window slicing, feature prep, CV-AUC, permutation, drivers).
- `test_drift_detector.py` — pytest suite, built test-first:
  - synthetic **no-drift** data → AUC ≈ 0.5, high p-value, label `none`.
  - synthetic **injected drift** → high AUC, low p-value, severe label,
    and the injected column appears in `top_drivers`.
  - edge cases: empty window, single class, ID-column auto-drop, tiny
    windows reducing `n_splits`, overlap warning, output schema/shape.
- `requirements.txt` — pandas, numpy, scikit-learn, lightgbm, pytest.
- `.venv/` — local virtual environment (git-ignored).
- `.gitignore` — ignore `.venv/`, `__pycache__/`, `*.pyc`.
- `example.py` — runnable demo on synthetic data printing the result table.
- `README.md` — short usage section.

## 7. Implementation method

Built **test-driven** (red → green → refactor): write each test, watch it
fail, implement the minimum to pass, refactor. Verify with the full pytest
suite before claiming completion.

## 8. Risks

- **Python 3.14 wheels:** The environment runs Python 3.14.5. During setup
  we verify LightGBM (and numpy/pandas/scikit-learn) provide 3.14 wheels.
  If LightGBM has no compatible wheel, fall back to scikit-learn's
  `HistGradientBoostingClassifier`, which also supports native categorical
  features and missing values; the rest of the design is unchanged. The
  chosen backend will be recorded in the README.

## 9. Out of scope (YAGNI)

- Per-feature univariate AUC (gain importance covers "which columns drive
  drift" cheaply; can be added later if needed).
- More than two windows / rolling drift over time (caller can loop and
  `concat`).
- Plotting / visualization.
- SHAP-based explanations.
