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
