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
