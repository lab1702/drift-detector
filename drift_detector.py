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
