# Rolling-Window Drift Helper — Design Spec

**Date:** 2026-05-20
**Status:** Approved design, pending spec review
**Builds on:** `detect_drift` (see 2026-05-20-domain-classifier-drift-detector-design.md)

## 1. Purpose

Provide `detect_drift_rolling`, a convenience wrapper that bins a dataset's
timeline into consecutive periods by a pandas frequency and runs `detect_drift`
across them, stacking the one-row results into a single report table. Two
comparison modes:

- **consecutive** — each period vs the next (Jan→Feb, Feb→Mar, …); catches
  drift between adjacent periods.
- **baseline** — every period vs one fixed reference window; catches
  cumulative drift away from a reference. The reference can be the first
  period, the most recent period, or an explicit date range.

## 2. Public API

```python
detect_drift_rolling(
    df,                       # pandas.DataFrame
    date_column,              # str
    freq,                     # pandas offset alias: "MS", "W", "QS", "30D", ...
    *,
    mode="consecutive",       # "consecutive" | "baseline"
    baseline="first",         # baseline mode only: "first" | "last" | (start, end)
    start=None,               # restrict timeline start (default: data min)
    end=None,                 # restrict timeline end (default: data max)
    **detect_drift_kwargs,    # forwarded: features, n_splits, n_permutations,
                              #            thresholds, top_n, random_state
) -> pandas.DataFrame         # one row per comparison
```

### Argument notes
- `freq` is a pandas **offset alias** (the `pd.date_range` family: `"MS"`,
  `"W"`, `"QS"`, `"30D"`). Note: pandas *Period* aliases (`"M"`) differ and are
  **not** what this function takes — it bins via `pd.date_range`, which accepts
  the offset forms.
- `baseline` is only consulted when `mode="baseline"`. `None` is accepted as an
  alias for `"first"`.
- `**detect_drift_kwargs` pass straight through to each `detect_drift` call.

## 3. Pipeline

### 3.1 Validate & resolve the timeline
1. If `date_column` not in `df` → `ValueError`.
2. If `mode` not in `{"consecutive", "baseline"}` → `ValueError`.
3. Parse `df[date_column]` with `pd.to_datetime`. Resolve the timeline range:
   `lo = pd.to_datetime(start) if start else dates.min()`,
   `hi = pd.to_datetime(end) if end else dates.max()`.

### 3.2 Build calendar-aligned period windows
1. Snap the low bound back to a period boundary:
   `first_edge = pd.tseries.frequencies.to_offset(freq).rollback(lo)`.
   (For tick offsets like `"30D"`, `rollback` is a no-op, so `first_edge == lo`.)
2. `edges = pd.date_range(start=first_edge, end=hi, freq=freq)`.
3. Form windows from consecutive edges, inclusive and non-overlapping:
   - window `i` = `[edges[i], edges[i+1] - 1ns]` for `i < len(edges)-1`
   - last window = `[edges[-1], hi]`
   Because each window ends 1ns before the next begins, `detect_drift`'s own
   overlap check stays silent.

### 3.3 Drop unusable periods
1. Count rows of `df` falling in each window (`lo_w <= date <= hi_w`).
2. Drop any period with `< 2` rows (cannot cross-validate) and emit a
   `UserWarning` naming the dropped spans.
3. If fewer than 2 usable periods remain → `ValueError`.

### 3.4 Build comparison pairs
Let the surviving periods, chronological, be `P1 … PN`.

- `mode="consecutive"`: pairs `(window_1=P_i, window_2=P_{i+1})` for
  `i = 1 … N-1`.
- `mode="baseline"`: resolve the reference window `B`:
  - `"first"` / `None` → `P1`; candidates = `P2 … PN`
  - `"last"` → `PN`; candidates = `P1 … P(N-1)`
  - `(start, end)` tuple → `B = (pd.to_datetime(start), pd.to_datetime(end))`;
    candidates = `P1 … PN`
  Pairs: `(window_1=B, window_2=C)` for each candidate `C` in chronological
  order. `B` is therefore fixed in the window-1 columns across all rows.

### 3.5 Execute
For each pair, call
`detect_drift(df, date_column, s1, e1, s2, e2, **detect_drift_kwargs)` and
collect the one-row result. `detect_drift`'s per-call warnings (e.g. n_splits
reduction for small periods, or an overlap warning if an explicit `baseline`
range overlaps a candidate) propagate to the caller.

`pd.concat(rows, ignore_index=True)` → the returned report.

## 4. Output

A `pandas.DataFrame` with **one row per comparison**, using `detect_drift`'s
exact 11-column schema (`start_date_1`, `end_date_1`, `start_date_2`,
`end_date_2`, `n_window_1`, `n_window_2`, `n_features`, `auc`, `drift_label`,
`p_value`, `top_drivers`). The four date columns identify each comparison; no
extra columns are added.

## 5. Edge cases & errors

| Condition                                   | Behavior                              |
|---------------------------------------------|---------------------------------------|
| `date_column` not in `df`                   | `ValueError`                          |
| `mode` not recognized                       | `ValueError`                          |
| Invalid `freq`                              | propagate pandas error                |
| Period with `< 2` rows                      | drop it, `UserWarning` naming span    |
| Fewer than 2 usable periods                 | `ValueError`                          |
| Small (2–4 row) periods                     | kept; `detect_drift` reduces n_splits and warns |
| Explicit `baseline` overlaps a candidate    | `detect_drift` emits overlap warning  |

## 6. Deliverables

- `detect_drift_rolling` added to `drift_detector.py` (small private helper
  `_build_period_windows(dates, freq, lo, hi)` for §3.2 to keep it testable).
- Tests appended to `test_drift_detector.py`:
  - window builder produces non-overlapping inclusive calendar windows for
    `"MS"` and `"30D"`.
  - consecutive mode on monthly synthetic data with drift injected in a later
    month → that month's transition shows high AUC; output has N-1 rows.
  - baseline `"first"` vs `"last"` → reference fixed in window-1 columns;
    correct row counts and candidate ordering.
  - `< 2`-row period dropped with warning.
  - errors: missing date column, bad mode, fewer than 2 usable periods.
  - kwargs forwarding (e.g. `features=[...]`, small `n_permutations`).
- README: a short "Rolling drift" usage section.

## 7. Implementation method

Test-driven (red → green → refactor), committed in small steps. Full pytest
suite must pass before completion.

## 8. Out of scope (YAGNI)

- Explicit `windows=[(start,end), ...]` override (freq covers the need).
- A human-readable `comparison` label column (date columns suffice).
- Period (`"M"`) alias support — offset aliases match the chosen UX.
- Plotting.
