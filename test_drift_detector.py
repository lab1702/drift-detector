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
