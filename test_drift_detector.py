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
