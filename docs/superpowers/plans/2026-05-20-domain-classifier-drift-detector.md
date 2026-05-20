# Domain-Classifier Drift Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `detect_drift`, a function that detects distribution drift between two date windows of any tabular dataset using a domain-classifier (C2ST) with a cross-validated AUC, a permutation-test p-value, and per-feature drift drivers.

**Architecture:** A single module `drift_detector.py` exposes `detect_drift` and small private helpers (window slicing, feature prep, K-fold AUC, permutation test, drivers). Window-1 rows are labeled 0 and window-2 rows 1; a LightGBM classifier is cross-validated to produce an out-of-fold ROC-AUC, interpreted into none/mild/moderate/severe bands, validated with a label-permutation test, and explained via averaged gain importances. Output is a one-row DataFrame.

**Tech Stack:** Python 3.14, pandas 3.0, numpy 2.4, scikit-learn 1.8 (StratifiedKFold, roc_auc_score), LightGBM 4.6, pytest 9. A `.venv/` already exists with all packages installed.

**Conventions for every command in this plan:**
- Run Python/pytest through the venv: `./.venv/Scripts/python.exe -m pytest ...`
- Working directory is the repo root `C:/Users/lab17/tmp/datadrift`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Write `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
pytest>=7.0
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Verify the venv satisfies requirements**

Run: `./.venv/Scripts/python.exe -c "import pandas, numpy, sklearn, lightgbm, pytest; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add requirements and gitignore"
```

---

### Task 2: Drift-label banding (`_label_drift`)

**Files:**
- Create: `drift_detector.py`
- Create: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`:

```python
import warnings

import numpy as np
import pandas as pd
import pytest

from drift_detector import _label_drift


@pytest.mark.parametrize(
    "auc,expected",
    [
        (0.30, "none"),
        (0.50, "none"),
        (0.599, "none"),
        (0.60, "mild"),
        (0.699, "mild"),
        (0.70, "moderate"),
        (0.799, "moderate"),
        (0.80, "severe"),
        (0.99, "severe"),
    ],
)
def test_label_drift_bands(auc, expected):
    assert _label_drift(auc) == expected


def test_label_drift_custom_thresholds():
    assert _label_drift(0.62, thresholds=(0.55, 0.65, 0.75)) == "mild"
    assert _label_drift(0.80, thresholds=(0.55, 0.65, 0.75)) == "severe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'drift_detector'` or `ImportError: cannot import name '_label_drift'`

- [ ] **Step 3: Write minimal implementation**

Create `drift_detector.py` with the full import block (used by later tasks) and the function:

```python
"""Domain-classifier (C2ST) drift detector for arbitrary tabular data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _label_drift(auc, thresholds=(0.6, 0.7, 0.8)):
    """Map an AUC to a drift severity label.

    AUC is symmetric around 0.5; values below the first threshold (including
    anything <= 0.5) are treated as ``none``.
    """
    t1, t2, t3 = thresholds
    if auc < t1:
        return "none"
    if auc < t2:
        return "mild"
    if auc < t3:
        return "moderate"
    return "severe"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add drift-label banding"
```

---

### Task 3: Feature preparation (`_prepare_features`)

**Files:**
- Modify: `drift_detector.py`
- Modify: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`:

```python
from drift_detector import _prepare_features


def test_prepare_features_auto_drops_id_and_constant_keeps_float():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 10),
            "row_id": range(10),          # int, all-unique -> drop (ID-like)
            "const": [7] * 10,            # constant -> drop
            "measure": np.arange(10.0),   # float, all-unique -> KEEP (drift signal)
            "category": ["a", "b"] * 5,   # low-cardinality string -> keep + cast
        }
    )
    X = _prepare_features(frame, "date")
    assert set(X.columns) == {"measure", "category"}
    assert isinstance(X["category"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_float_dtype(X["measure"])


def test_prepare_features_explicit_list_bypasses_autodrop():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "row_id": range(5),
            "measure": np.arange(5.0),
        }
    )
    X = _prepare_features(frame, "date", features=["row_id", "measure"])
    assert list(X.columns) == ["row_id", "measure"]


def test_prepare_features_drops_date_even_if_in_feature_list():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"] * 5), "measure": np.arange(5.0)}
    )
    X = _prepare_features(frame, "date", features=["date", "measure"])
    assert "date" not in X.columns


def test_prepare_features_all_dropped_raises():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 6),
            "row_id": range(6),
            "const": [1] * 6,
        }
    )
    with pytest.raises(ValueError):
        _prepare_features(frame, "date")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name '_prepare_features'`

- [ ] **Step 3: Write minimal implementation**

Add to `drift_detector.py`:

```python
def _prepare_features(frame, date_column, features=None, uniqueness_threshold=0.95):
    """Select feature columns and make them LightGBM-ready.

    - Always excludes ``date_column``.
    - When ``features`` is None: drop constant columns and near-unique
      non-float (ID-like) columns. Float columns are exempt from the
      uniqueness drop.
    - When ``features`` is given: use it verbatim (minus ``date_column``),
      skipping the auto-drop heuristic.
    - Casts object/string/bool columns to ``category`` and any leftover
      datetime feature columns to int64 so LightGBM can consume them.
    """
    if features is None:
        cols = [c for c in frame.columns if c != date_column]
        auto = True
    else:
        cols = [c for c in features if c != date_column]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"features not found in dataframe: {missing}")
        auto = False

    X = frame[cols].copy()

    if auto:
        keep = []
        for c in X.columns:
            s = X[c]
            nunique = s.nunique(dropna=True)
            if nunique <= 1:
                continue  # constant / all-null -> uninformative
            if not pd.api.types.is_float_dtype(s):
                n_non_null = int(s.notna().sum())
                if n_non_null > 0 and nunique / n_non_null > uniqueness_threshold:
                    continue  # ID-like
            keep.append(c)
        X = X[keep]

    if X.shape[1] == 0:
        raise ValueError(
            "No usable features after dropping ID-like/constant columns; "
            "pass an explicit `features=[...]` list."
        )

    for c in X.columns:
        dt = X[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            continue
        if (
            pd.api.types.is_bool_dtype(dt)
            or dt == object
            or pd.api.types.is_string_dtype(dt)
        ):
            X[c] = X[c].astype("category")
        elif pd.api.types.is_datetime64_any_dtype(dt):
            X[c] = X[c].astype("int64")

    return X
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add feature preparation with ID-column auto-drop"
```

---

### Task 4: Cross-validated AUC (`_make_model`, `_resolve_n_splits`, `_cv_auc`)

**Files:**
- Modify: `drift_detector.py`
- Modify: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`:

```python
from drift_detector import _cv_auc, _resolve_n_splits


def test_cv_auc_high_on_separable_data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {"f": np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])}
    )
    y = np.array([0] * 200 + [1] * 200)
    auc, importances = _cv_auc(X, y, n_splits=5, random_state=0, collect_importance=True)
    assert auc > 0.9
    assert len(importances) == 5


def test_cv_auc_near_half_on_identical_data():
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"f": rng.normal(0, 1, 400)})
    y = np.array([0] * 200 + [1] * 200)
    auc = _cv_auc(X, y, n_splits=5, random_state=0)
    assert 0.4 <= auc <= 0.6


def test_resolve_n_splits_reduces_and_warns():
    y = np.array([0, 0, 0, 1, 1, 1])  # smallest class = 3
    with pytest.warns(UserWarning):
        n = _resolve_n_splits(y, 5)
    assert n == 3


def test_resolve_n_splits_raises_when_class_too_small():
    y = np.array([0, 1, 1, 1])  # smallest class = 1
    with pytest.raises(ValueError):
        _resolve_n_splits(y, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name '_cv_auc'`

- [ ] **Step 3: Write minimal implementation**

Add to `drift_detector.py`:

```python
def _make_model(random_state):
    """LightGBM classifier tuned to be robust on small windows and quiet.

    ``importance_type='gain'`` makes ``feature_importances_`` return gains.
    """
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        importance_type="gain",
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )


def _resolve_n_splits(y, n_splits):
    """Clamp n_splits to the smallest class size; warn if reduced."""
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        raise ValueError(
            "Each window needs at least 2 rows to cross-validate "
            f"(smallest window has {min_class})."
        )
    if min_class < n_splits:
        warnings.warn(
            f"Reducing n_splits from {n_splits} to {min_class} "
            f"(smallest class has {min_class} rows).",
            UserWarning,
        )
        return min_class
    return n_splits


def _cv_auc(X, y, n_splits, random_state, collect_importance=False):
    """Out-of-fold ROC-AUC over a stratified K-fold.

    ``n_splits`` must already be resolved (see ``_resolve_n_splits``) so the
    permutation loop does not re-warn. Returns the AUC, or ``(auc, [imp,...])``
    when ``collect_importance`` is True (one gain-importance array per fold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)
    importances = []
    for train_idx, test_idx in skf.split(X, y):
        model = _make_model(random_state)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        if collect_importance:
            importances.append(np.asarray(model.feature_importances_, dtype=float))
    auc = roc_auc_score(y, oof)
    if collect_importance:
        return auc, importances
    return auc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add cross-validated AUC with n_splits resolution"
```

---

### Task 5: Permutation test (`_permutation_pvalue`)

**Files:**
- Modify: `drift_detector.py`
- Modify: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`:

```python
from drift_detector import _permutation_pvalue


def test_permutation_pvalue_significant_when_separable():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {"f": np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])}
    )
    y = np.array([0] * 150 + [1] * 150)
    observed = _cv_auc(X, y, n_splits=5, random_state=0)
    p = _permutation_pvalue(X, y, observed, n_splits=5, n_permutations=20, random_state=0)
    assert p <= 0.1


def test_permutation_pvalue_bounds():
    # p-value is always in (0, 1] thanks to the +1 correction
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f": rng.normal(0, 1, 200)})
    y = np.array([0] * 100 + [1] * 100)
    observed = _cv_auc(X, y, n_splits=5, random_state=0)
    p = _permutation_pvalue(X, y, observed, n_splits=5, n_permutations=10, random_state=0)
    assert 0.0 < p <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name '_permutation_pvalue'`

- [ ] **Step 3: Write minimal implementation**

Add to `drift_detector.py`:

```python
def _permutation_pvalue(X, y, observed_auc, n_splits, n_permutations, random_state):
    """Label-permutation test for the C2ST AUC.

    Shuffle the labels ``n_permutations`` times, recompute the K-fold AUC, and
    return ``(#{perm_auc >= observed} + 1) / (n_permutations + 1)``. The +1
    keeps the p-value strictly positive.
    """
    rng = np.random.RandomState(random_state)
    count = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_auc = _cv_auc(X, y_perm, n_splits, random_state)
        if perm_auc >= observed_auc:
            count += 1
    return (count + 1) / (n_permutations + 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add permutation-test p-value"
```

---

### Task 6: Drift drivers (`_top_drivers`)

**Files:**
- Modify: `drift_detector.py`
- Modify: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`:

```python
from drift_detector import _top_drivers


def test_top_drivers_ranks_and_normalizes():
    importances = [np.array([10.0, 0.0, 2.0]), np.array([6.0, 0.0, 2.0])]
    names = ["a", "b", "c"]
    drivers = _top_drivers(importances, names, top_n=2)
    assert [d["feature"] for d in drivers] == ["a", "c"]
    assert len(drivers) == 2
    full = _top_drivers(importances, names, top_n=3)
    assert abs(sum(d["importance"] for d in full) - 1.0) < 1e-9


def test_top_drivers_handles_all_zero_importance():
    importances = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    drivers = _top_drivers(importances, ["a", "b"], top_n=2)
    assert len(drivers) == 2
    assert all(d["importance"] == 0.0 for d in drivers)


def test_top_drivers_empty_when_no_importances():
    assert _top_drivers([], ["a"], top_n=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name '_top_drivers'`

- [ ] **Step 3: Write minimal implementation**

Add to `drift_detector.py`:

```python
def _top_drivers(importances, feature_names, top_n):
    """Average per-fold gain importances, normalize to shares, take top_n.

    Returns a list of ``{"feature": name, "importance": share}`` sorted
    descending. Drivers are only meaningful when drift is significant (high
    AUC / low p-value); at AUC ~ 0.5 they are noise.
    """
    if not importances:
        return []
    mean_imp = np.mean(np.vstack(importances), axis=0)
    total = mean_imp.sum()
    shares = mean_imp / total if total > 0 else np.zeros_like(mean_imp)
    order = np.argsort(shares)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": float(shares[i])} for i in order
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add drift-driver ranking from gain importances"
```

---

### Task 7: Orchestrator (`detect_drift`)

**Files:**
- Modify: `drift_detector.py`
- Modify: `test_drift_detector.py`

- [ ] **Step 1: Write the failing test**

Add to `test_drift_detector.py`. The `_make_drift_df` helper builds two date windows; with `drift=True` the float feature `x` is mean-shifted in window 2.

```python
from drift_detector import detect_drift

_W1 = ("2024-01-01", "2024-01-31")
_W2 = ("2024-03-01", "2024-03-31")
_EXPECTED_COLUMNS = [
    "start_date_1", "end_date_1", "start_date_2", "end_date_2",
    "n_window_1", "n_window_2", "n_features", "auc", "drift_label",
    "p_value", "top_drivers",
]


def _make_drift_df(n_per=400, drift=False, seed=0):
    rng = np.random.RandomState(seed)
    dates1 = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.randint(0, 30, n_per), unit="D"
    )
    dates2 = pd.to_datetime("2024-03-01") + pd.to_timedelta(
        rng.randint(0, 30, n_per), unit="D"
    )
    x1 = rng.normal(0, 1, n_per)
    x2 = rng.normal(3, 1, n_per) if drift else rng.normal(0, 1, n_per)
    return pd.DataFrame(
        {
            "date": pd.concat(
                [pd.Series(dates1), pd.Series(dates2)], ignore_index=True
            ),
            "x": np.concatenate([x1, x2]),
            "noise": rng.normal(0, 1, 2 * n_per),
            "cat": rng.choice(["a", "b", "c"], 2 * n_per),
        }
    )


def test_detect_drift_no_drift():
    df = _make_drift_df(n_per=400, drift=False, seed=1)
    res = detect_drift(df, "date", *_W1, *_W2, n_permutations=50, random_state=0)
    assert res["auc"].iloc[0] <= 0.6
    assert res["drift_label"].iloc[0] == "none"
    assert res["p_value"].iloc[0] > 0.05


def test_detect_drift_with_injected_drift_and_driver():
    df = _make_drift_df(n_per=400, drift=True, seed=2)
    res = detect_drift(df, "date", *_W1, *_W2, n_permutations=50, random_state=0)
    assert res["auc"].iloc[0] >= 0.8
    assert res["drift_label"].iloc[0] == "severe"
    assert res["p_value"].iloc[0] <= 0.05
    assert res["top_drivers"].iloc[0][0]["feature"] == "x"


def test_detect_drift_output_schema():
    df = _make_drift_df(n_per=100, drift=True, seed=3)
    res = detect_drift(df, "date", *_W1, *_W2, n_permutations=10, random_state=0)
    assert list(res.columns) == _EXPECTED_COLUMNS
    assert len(res) == 1
    assert res["n_window_1"].iloc[0] > 0
    assert res["n_window_2"].iloc[0] > 0


def test_detect_drift_explicit_features():
    df = _make_drift_df(n_per=100, drift=True, seed=5)
    res = detect_drift(
        df, "date", *_W1, *_W2, features=["x"], n_permutations=10, random_state=0
    )
    assert res["n_features"].iloc[0] == 1


def test_detect_drift_empty_window_raises():
    df = _make_drift_df(n_per=50)
    with pytest.raises(ValueError):
        detect_drift(df, "date", "2020-01-01", "2020-01-31", *_W2, n_permutations=5)


def test_detect_drift_missing_date_column_raises():
    df = _make_drift_df(n_per=10)
    with pytest.raises(ValueError):
        detect_drift(df, "nope", *_W1, *_W2, n_permutations=5)


def test_detect_drift_all_features_dropped_raises():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15"] * 5 + ["2024-03-15"] * 5),
            "row_id": range(10),
            "const": [7] * 10,
        }
    )
    with pytest.raises(ValueError):
        detect_drift(df, "date", *_W1, *_W2, n_permutations=5)


def test_detect_drift_overlapping_windows_warn():
    df = _make_drift_df(n_per=100, drift=False, seed=6)
    with pytest.warns(UserWarning):
        detect_drift(
            df, "date", "2024-01-01", "2024-03-31", "2024-01-15", "2024-04-30",
            n_permutations=5, random_state=0,
        )


def test_detect_drift_small_window_reduces_splits_warn():
    df = _make_drift_df(n_per=3, drift=True, seed=4)
    with pytest.warns(UserWarning):
        res = detect_drift(df, "date", *_W1, *_W2, n_permutations=5, random_state=0)
    assert len(res) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: FAIL with `ImportError: cannot import name 'detect_drift'`

- [ ] **Step 3: Write minimal implementation**

Add to `drift_detector.py`:

```python
def detect_drift(
    df,
    date_column,
    start1,
    end1,
    start2,
    end2,
    *,
    features=None,
    n_splits=5,
    n_permutations=100,
    thresholds=(0.6, 0.7, 0.8),
    top_n=5,
    random_state=0,
):
    """Detect distribution drift between two date windows of ``df``.

    Trains a LightGBM domain classifier to distinguish window-1 rows (label 0)
    from window-2 rows (label 1). Returns a one-row DataFrame with the
    cross-validated ROC-AUC, a none/mild/moderate/severe label, a
    permutation-test p-value, and the top drift-driving features.

    Date bounds are inclusive and compared as timestamps after
    ``pd.to_datetime``; a bare date like ``"2024-01-31"`` is midnight, so later
    times that day fall outside the window.
    """
    if date_column not in df.columns:
        raise ValueError(f"date_column {date_column!r} not in dataframe.")

    dates = pd.to_datetime(df[date_column])
    s1, e1 = pd.to_datetime(start1), pd.to_datetime(end1)
    s2, e2 = pd.to_datetime(start2), pd.to_datetime(end2)

    if max(s1, s2) <= min(e1, e2):
        warnings.warn("Window 1 and window 2 overlap in time.", UserWarning)

    m1 = (dates >= s1) & (dates <= e1)
    m2 = (dates >= s2) & (dates <= e2)
    n1, n2 = int(m1.sum()), int(m2.sum())
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"Empty window: window 1 has {n1} rows, window 2 has {n2} rows."
        )

    frame = pd.concat([df.loc[m1], df.loc[m2]], ignore_index=True)
    y = np.array([0] * n1 + [1] * n2)

    X = _prepare_features(frame, date_column, features)
    n_splits_resolved = _resolve_n_splits(y, n_splits)

    auc, importances = _cv_auc(
        X, y, n_splits_resolved, random_state, collect_importance=True
    )
    label = _label_drift(auc, thresholds)
    p_value = _permutation_pvalue(
        X, y, auc, n_splits_resolved, n_permutations, random_state
    )
    drivers = _top_drivers(importances, list(X.columns), top_n)

    return pd.DataFrame(
        [
            {
                "start_date_1": s1,
                "end_date_1": e1,
                "start_date_2": s2,
                "end_date_2": e2,
                "n_window_1": n1,
                "n_window_2": n2,
                "n_features": X.shape[1],
                "auc": auc,
                "drift_label": label,
                "p_value": p_value,
                "top_drivers": drivers,
            }
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest test_drift_detector.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add drift_detector.py test_drift_detector.py
git commit -m "feat: add detect_drift orchestrator"
```

---

### Task 8: Example script and README

**Files:**
- Create: `example.py`
- Create: `README.md`

- [ ] **Step 1: Write `example.py`**

```python
"""Demo: detect_drift on synthetic data (one drifting case, one stable case)."""

import numpy as np
import pandas as pd

from drift_detector import detect_drift


def make_data(seed=0):
    rng = np.random.RandomState(seed)
    n = 500
    dates1 = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.randint(0, 30, n), unit="D"
    )
    dates2 = pd.to_datetime("2024-03-01") + pd.to_timedelta(
        rng.randint(0, 30, n), unit="D"
    )
    df = pd.DataFrame(
        {
            "event_date": pd.concat(
                [pd.Series(dates1), pd.Series(dates2)], ignore_index=True
            ),
            "amount": np.concatenate(
                [rng.normal(100, 15, n), rng.normal(130, 15, n)]  # drifted up
            ),
            "score": rng.normal(0, 1, 2 * n),  # stable
            "region": rng.choice(["north", "south", "east", "west"], 2 * n),
            "customer_id": range(2 * n),  # ID-like -> auto-dropped
        }
    )
    return df


if __name__ == "__main__":
    df = make_data()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    result = detect_drift(
        df,
        date_column="event_date",
        start1="2024-01-01",
        end1="2024-01-31",
        start2="2024-03-01",
        end2="2024-03-31",
        n_permutations=100,
    )

    print("Drift report:")
    print(result.drop(columns="top_drivers").to_string(index=False))
    print("\nTop drivers:")
    for d in result["top_drivers"].iloc[0]:
        print(f"  {d['feature']:<12} {d['importance']:.3f}")
```

- [ ] **Step 2: Run the example**

Run: `./.venv/Scripts/python.exe example.py`
Expected: prints a one-row drift report with `drift_label` = `severe` (or `moderate`), a low `p_value`, and `amount` as the top driver. `customer_id` should not appear (auto-dropped).

- [ ] **Step 3: Write `README.md`**

```markdown
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

## Run the demo and tests

```bash
./.venv/Scripts/python.exe example.py
./.venv/Scripts/python.exe -m pytest -q
```
```

- [ ] **Step 4: Commit**

```bash
git add example.py README.md
git commit -m "docs: add example script and README"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all tests PASS, no errors.

- [ ] **Step 2: Run the example end-to-end**

Run: `./.venv/Scripts/python.exe example.py`
Expected: drift report prints with `amount` as top driver and a severe/moderate label.

- [ ] **Step 3: Confirm clean git status**

Run: `git status --short`
Expected: empty (everything committed; `.venv/` ignored).

---

## Notes for the implementer

- **Determinism:** all randomness flows through `random_state`. If a
  threshold assertion in Task 7's no-drift test is unlucky on a given
  machine, nudge the data `seed` rather than loosening the C2ST logic — the
  no-drift AUC should sit near 0.5.
- **LightGBM noise:** `verbosity=-1` and `n_jobs=1` are set in `_make_model`
  to keep output clean and runs deterministic. Do not remove them.
- **Performance:** the permutation test runs `n_permutations × n_splits`
  LightGBM fits. Tests use small `n_permutations`; the default is 100.
