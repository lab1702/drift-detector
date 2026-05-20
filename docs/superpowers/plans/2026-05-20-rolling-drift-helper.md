# Rolling-Window Drift Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `detect_drift_rolling`, which bins a dataset's timeline into consecutive periods by a pandas frequency and runs `detect_drift` across them (consecutive or fixed-baseline mode), returning one stacked report DataFrame.

**Architecture:** Two functions added to the existing `drift_detector.py`: a small pure helper `_build_period_windows(freq, lo, hi)` that produces calendar-aligned, non-overlapping, inclusive `[start, end]` windows via `pd.date_range` + offset `rollback`; and `detect_drift_rolling`, which validates inputs, drops sparse periods, builds comparison pairs, calls `detect_drift` per pair, and `pd.concat`s the rows. Output reuses `detect_drift`'s exact 11-column schema.

**Tech Stack:** Python 3.14, pandas 3.0 (`date_range`, `to_offset(...).rollback`), plus the existing `detect_drift` stack (numpy, scikit-learn, LightGBM). Tests with pytest 9.

**Conventions for every command:**
- Run pytest through the venv: `./.venv/Scripts/python.exe -m pytest ...`
- Working directory: repo root `C:/Users/lab17/tmp/datadrift`. Branch: `feat/rolling-drift`.
- Both functions are APPENDED to the existing `drift_detector.py`; tests are APPENDED to the existing `test_drift_detector.py`. Preserve all existing content.

**Verified facts (pandas 3.0 in this venv):**
- `to_offset("MS").rollback(Timestamp("2024-01-05"))` → `2024-01-01`.
- `to_offset("30D").rollback(Timestamp("2024-01-05"))` → `2024-01-05` (tick offsets: identity).
- Windows are non-overlapping (`end_i = next_start - 1ns`), so `detect_drift`'s overlap check stays silent.
- All-data-in-one-month with `freq="MS"` → exactly 1 window.

---

### Task 1: Period-window builder (`_build_period_windows`)

**Files:**
- Modify: `drift_detector.py` (append one function)
- Modify: `test_drift_detector.py` (append tests)

- [ ] **Step 1: Write the failing tests**

APPEND to `test_drift_detector.py`:

```python
from drift_detector import _build_period_windows


def test_build_period_windows_monthly_non_overlapping():
    lo = pd.Timestamp("2024-01-05")
    hi = pd.Timestamp("2024-03-20")
    w = _build_period_windows("MS", lo, hi)
    assert w[0][0] == pd.Timestamp("2024-01-01")  # snapped back to month start
    assert w[1][0] == pd.Timestamp("2024-02-01")
    assert len(w) == 3
    for (s1, e1), (s2, e2) in zip(w, w[1:]):
        assert e1 < s2  # non-overlapping
    assert w[-1][1] == hi  # last window ends at hi


def test_build_period_windows_tick_freq_starts_at_lo():
    lo = pd.Timestamp("2024-01-05")
    hi = pd.Timestamp("2024-02-20")
    w = _build_period_windows("30D", lo, hi)
    assert w[0][0] == lo  # tick offset: rollback is identity
    assert w[-1][1] == hi


def test_build_period_windows_single_period():
    w = _build_period_windows("MS", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-27"))
    assert len(w) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name '_build_period_windows'`.

- [ ] **Step 3: Write minimal implementation**

APPEND to `drift_detector.py`:

```python
def _build_period_windows(freq, lo, hi):
    """Build calendar-aligned, non-overlapping, inclusive [start, end] windows.

    Snaps ``lo`` back to a period boundary with the offset's ``rollback`` (a
    no-op for tick offsets like ``"30D"``), generates edges with
    ``pd.date_range``, and forms windows ``[edges[i], edges[i+1] - 1ns]`` with
    the final window ending at ``hi``. ``freq`` is a pandas offset alias
    (``"MS"``, ``"W"``, ``"QS"``, ``"30D"``, ...).
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    first_edge = offset.rollback(pd.Timestamp(lo))
    edges = pd.date_range(start=first_edge, end=pd.Timestamp(hi), freq=freq)
    windows = []
    for i in range(len(edges)):
        w_start = edges[i]
        if i + 1 < len(edges):
            w_end = edges[i + 1] - pd.Timedelta(nanoseconds=1)
        else:
            w_end = pd.Timestamp(hi)
        windows.append((w_start, w_end))
    return windows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add period-window builder for rolling drift"
```

---

### Task 2: Rolling orchestrator (`detect_drift_rolling`)

**Files:**
- Modify: `drift_detector.py` (append one function)
- Modify: `test_drift_detector.py` (append a synthetic-data helper + tests)

- [ ] **Step 1: Write the failing tests**

APPEND to `test_drift_detector.py`. `_EXPECTED_COLUMNS` is already defined earlier in this file (from the `detect_drift` schema test) — reuse it; do not redefine it.

```python
from drift_detector import detect_drift_rolling


def _make_monthly_df(seed=0, drift_from="2024-04"):
    """5 monthly periods (Jan..May), ~80 rows each. Feature `x` mean-shifts
    from 0 to 5 starting at `drift_from`, so the transition into that month
    is the only strongly-drifting consecutive pair."""
    rng = np.random.RandomState(seed)
    frames = []
    for m in ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]:
        n = 80
        dates = pd.to_datetime(m + "-01") + pd.to_timedelta(
            rng.randint(0, 27, n), unit="D"
        )
        mean = 5.0 if m >= drift_from else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "x": rng.normal(mean, 1, n),
                    "noise": rng.normal(0, 1, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_rolling_consecutive_row_count_and_schema():
    df = _make_monthly_df(seed=1)
    res = detect_drift_rolling(
        df, "date", "MS", mode="consecutive", n_permutations=5, random_state=0
    )
    assert list(res.columns) == _EXPECTED_COLUMNS
    assert len(res) == 4  # 5 months -> 4 consecutive pairs
    assert res.index.tolist() == list(range(4))  # concat(ignore_index=True)


def test_rolling_consecutive_flags_drift_transition():
    df = _make_monthly_df(seed=1, drift_from="2024-04")
    res = detect_drift_rolling(
        df, "date", "MS", mode="consecutive", n_permutations=5, random_state=0
    ).reset_index(drop=True)
    # the March -> April transition (window_2 == April) is the drift step
    apr = res[res["start_date_2"].dt.month == 4]
    assert apr["auc"].iloc[0] >= 0.8
    assert apr["drift_label"].iloc[0] == "severe"
    # a non-transition pair (Jan -> Feb) should not be severe
    feb = res[res["start_date_2"].dt.month == 2]
    assert feb["drift_label"].iloc[0] != "severe"


def test_rolling_baseline_first_fixed_reference():
    df = _make_monthly_df(seed=2)
    res = detect_drift_rolling(
        df, "date", "MS", mode="baseline", baseline="first",
        n_permutations=5, random_state=0,
    ).reset_index(drop=True)
    assert len(res) == 4  # first is baseline, 4 candidates
    assert (res["start_date_1"] == pd.Timestamp("2024-01-01")).all()  # fixed ref
    assert list(res["start_date_2"].dt.month) == [2, 3, 4, 5]


def test_rolling_baseline_last_fixed_reference():
    df = _make_monthly_df(seed=3)
    res = detect_drift_rolling(
        df, "date", "MS", mode="baseline", baseline="last",
        n_permutations=5, random_state=0,
    ).reset_index(drop=True)
    assert len(res) == 4
    assert (res["start_date_1"] == pd.Timestamp("2024-05-01")).all()  # fixed ref
    assert list(res["start_date_2"].dt.month) == [1, 2, 3, 4]


def test_rolling_drops_sparse_period_with_warning():
    rng = np.random.RandomState(0)
    dense = [
        pd.DataFrame(
            {
                "date": pd.to_datetime(m + "-01")
                + pd.to_timedelta(rng.randint(0, 27, 60), unit="D"),
                "x": rng.normal(0, 1, 60),
            }
        )
        for m in ["2024-01", "2024-02", "2024-03"]
    ]
    sparse = pd.DataFrame({"date": [pd.Timestamp("2024-04-10")], "x": [0.0]})
    df = pd.concat(dense + [sparse], ignore_index=True)
    with pytest.warns(UserWarning):
        res = detect_drift_rolling(
            df, "date", "MS", mode="consecutive", n_permutations=5, random_state=0
        )
    assert len(res) == 2  # April (1 row) dropped -> Jan,Feb,Mar -> 2 pairs
    assert set(res["start_date_2"].dt.month) == {2, 3}


def test_rolling_forwards_kwargs_features():
    df = _make_monthly_df(seed=5)
    res = detect_drift_rolling(
        df, "date", "MS", mode="consecutive",
        features=["x"], n_permutations=5, random_state=0,
    )
    assert (res["n_features"] == 1).all()


def test_rolling_missing_date_column_raises():
    df = _make_monthly_df()
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "nope", "MS")


def test_rolling_bad_mode_raises():
    df = _make_monthly_df()
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "date", "MS", mode="sideways")


def test_rolling_too_few_periods_raises():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.randint(0, 27, 50), unit="D"),
            "x": rng.normal(0, 1, 50),
        }
    )  # all in January -> 1 usable period
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "date", "MS")


def test_rolling_bad_baseline_raises():
    df = _make_monthly_df()
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "date", "MS", mode="baseline", baseline="middle")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name 'detect_drift_rolling'`.

- [ ] **Step 3: Write minimal implementation**

APPEND to `drift_detector.py`:

```python
def detect_drift_rolling(
    df,
    date_column,
    freq,
    *,
    mode="consecutive",
    baseline="first",
    start=None,
    end=None,
    **detect_drift_kwargs,
):
    """Run ``detect_drift`` across periods of a timeline and stack the rows.

    Bins the data's date range into consecutive, calendar-aligned periods of
    width ``freq`` (a pandas offset alias such as ``"MS"``, ``"W"``, ``"QS"``,
    ``"30D"``) and compares them:

    - ``mode="consecutive"``: each period vs the next.
    - ``mode="baseline"``: every period vs one fixed reference window, chosen by
      ``baseline``: ``"first"`` (earliest period, the default), ``"last"`` (most
      recent period), or an explicit ``(start, end)`` tuple. The reference is
      held in the window-1 columns across all rows.

    Periods with fewer than 2 rows are dropped (cannot cross-validate) with a
    warning. Extra keyword arguments are forwarded to each ``detect_drift``
    call. Returns a DataFrame with one row per comparison, using
    ``detect_drift``'s schema.
    """
    if date_column not in df.columns:
        raise ValueError(f"date_column {date_column!r} not in dataframe.")
    if mode not in ("consecutive", "baseline"):
        raise ValueError(
            f"mode must be 'consecutive' or 'baseline', got {mode!r}."
        )

    dates = pd.to_datetime(df[date_column])
    lo = pd.to_datetime(start) if start is not None else dates.min()
    hi = pd.to_datetime(end) if end is not None else dates.max()

    windows = _build_period_windows(freq, lo, hi)

    usable, dropped = [], []
    for w_start, w_end in windows:
        n = int(((dates >= w_start) & (dates <= w_end)).sum())
        (usable if n >= 2 else dropped).append((w_start, w_end))
    if dropped:
        spans = ", ".join(f"[{s.date()}..{e.date()}]" for s, e in dropped)
        warnings.warn(
            f"Dropping {len(dropped)} period(s) with <2 rows: {spans}",
            UserWarning,
        )
    if len(usable) < 2:
        raise ValueError(
            f"Need at least 2 usable periods (>=2 rows each); found {len(usable)}."
        )

    if mode == "consecutive":
        pairs = list(zip(usable, usable[1:]))
    else:  # baseline
        if baseline is None or baseline == "first":
            ref, candidates = usable[0], usable[1:]
        elif baseline == "last":
            ref, candidates = usable[-1], usable[:-1]
        elif isinstance(baseline, (tuple, list)) and len(baseline) == 2:
            ref = (pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1]))
            candidates = list(usable)
        else:
            raise ValueError(
                "baseline must be 'first', 'last', or a (start, end) tuple; "
                f"got {baseline!r}."
            )
        pairs = [(ref, cand) for cand in candidates]

    rows = [
        detect_drift(
            df, date_column, w1[0], w1[1], w2[0], w2[1], **detect_drift_kwargs
        )
        for w1, w2 in pairs
    ]
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS (all tests). Note the runtime (rolling tests invoke `detect_drift` several times each).

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add detect_drift_rolling (consecutive + baseline modes)"
```

---

### Task 3: Docs — rolling section in README and example

**Files:**
- Modify: `README.md` (append a section)
- Modify: `example.py` (append a rolling demo)

- [ ] **Step 1: Append a "Rolling drift over time" section to `README.md`**

Append at the end of `README.md`:

```markdown
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
```

NOTE: when writing the section, keep the inner Python code block as a normal
triple-backtick block; the block above is shown inside this plan only.

- [ ] **Step 2: Append a rolling demo to `example.py`**

Append these lines (indented 4 spaces, continuing the existing `if __name__ == "__main__":` block) at the very end of `example.py`, after the current "Top drivers" loop:

```python
    from drift_detector import detect_drift_rolling

    print("\nRolling (consecutive months):")
    rolling = detect_drift_rolling(
        df, date_column="event_date", freq="MS", mode="consecutive",
        n_permutations=50,
    )
    cols = ["start_date_2", "end_date_2", "n_window_2", "auc", "drift_label", "p_value"]
    print(rolling[cols].to_string(index=False))
```

- [ ] **Step 3: Run the example**

Run: `./.venv/Scripts/python.exe example.py`
Expected: the original single-comparison report still prints, followed by a
"Rolling (consecutive months)" table with one row per consecutive month pair.
Paste the output into your report.

- [ ] **Step 4: Commit**

```bash
git add README.md example.py
git commit -m "docs: document and demo rolling drift helper"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all tests PASS (the prior 33 plus the new rolling tests).

- [ ] **Step 2: Run the example end-to-end**

Run: `./.venv/Scripts/python.exe example.py`
Expected: both the single and rolling reports print without error.

- [ ] **Step 3: Confirm clean git status**

Run: `git status --short`
Expected: empty.

---

## Notes for the implementer

- **Append only.** Both `drift_detector.py` and `test_drift_detector.py` already
  contain the `detect_drift` implementation and its tests — never rewrite them.
- **Determinism / speed:** rolling tests use small `n_permutations` (5) because
  they call `detect_drift` several times each; the assertions are about row
  counts, schema, fixed-reference columns, and AUC magnitude on a strong
  injected shift — not p-value thresholds. Do not raise `n_permutations` to
  "stabilize" anything.
- **Why offset aliases (not Period aliases):** binning uses `pd.date_range`,
  which takes `"MS"`/`"QS"`; pandas `Period` rejects those. This is deliberate.
