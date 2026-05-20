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
