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

    # Rolling drift across several months (consecutive periods).
    # `amount` shifts up starting in March, so only the Feb -> Mar step drifts.
    from drift_detector import detect_drift_rolling

    rng = np.random.RandomState(0)
    monthly = pd.concat(
        [
            pd.DataFrame(
                {
                    "event_date": pd.to_datetime(m + "-01")
                    + pd.to_timedelta(rng.randint(0, 27, 300), unit="D"),
                    "amount": rng.normal(130 if m >= "2024-03" else 100, 15, 300),
                    "score": rng.normal(0, 1, 300),
                }
            )
            for m in ["2024-01", "2024-02", "2024-03", "2024-04"]
        ],
        ignore_index=True,
    )

    print("\nRolling (consecutive months):")
    rolling = detect_drift_rolling(
        monthly, date_column="event_date", freq="MS", mode="consecutive",
        n_permutations=50,
    )
    cols = ["start_date_2", "end_date_2", "n_window_2", "auc", "drift_label", "p_value"]
    print(rolling[cols].to_string(index=False))
