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
    # seed=2 is a mid-null no-drift sample (AUC~0.505); a single fixed seed's
    # permutation p-value is ~uniform under the null, so this avoids a
    # marginal-by-chance p near the 0.05 boundary.
    df = _make_drift_df(n_per=400, drift=False, seed=2)
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


def test_detect_drift_unknown_feature_raises():
    df = _make_drift_df(n_per=20, drift=True, seed=7)
    with pytest.raises(ValueError):
        detect_drift(
            df, "date", *_W1, *_W2, features=["does_not_exist"], n_permutations=5
        )


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
    apr = res[res["start_date_2"].dt.month == 4]
    assert apr["auc"].iloc[0] >= 0.8
    assert apr["drift_label"].iloc[0] == "severe"
    feb = res[res["start_date_2"].dt.month == 2]
    assert feb["drift_label"].iloc[0] != "severe"


def test_rolling_baseline_first_fixed_reference():
    df = _make_monthly_df(seed=2)
    res = detect_drift_rolling(
        df, "date", "MS", mode="baseline", baseline="first",
        n_permutations=5, random_state=0,
    ).reset_index(drop=True)
    assert len(res) == 4
    assert (res["start_date_1"] == pd.Timestamp("2024-01-01")).all()
    assert list(res["start_date_2"].dt.month) == [2, 3, 4, 5]


def test_rolling_baseline_last_fixed_reference():
    df = _make_monthly_df(seed=3)
    res = detect_drift_rolling(
        df, "date", "MS", mode="baseline", baseline="last",
        n_permutations=5, random_state=0,
    ).reset_index(drop=True)
    assert len(res) == 4
    assert (res["start_date_1"] == pd.Timestamp("2024-05-01")).all()
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
    assert len(res) == 2
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
    )
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "date", "MS")


def test_rolling_bad_baseline_raises():
    df = _make_monthly_df()
    with pytest.raises(ValueError):
        detect_drift_rolling(df, "date", "MS", mode="baseline", baseline="middle")
