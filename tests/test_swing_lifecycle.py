import numpy as np
import pandas as pd

from src.signals import swing_lifecycle_frame, swing_technical_snapshot


def test_swing_lifecycle_frame_returns_production_score_series():
    daily_idx = pd.date_range("2021-01-01", periods=260 * 5, freq="B")
    weekly_idx = pd.date_range("2021-01-01", periods=260, freq="W-FRI")

    stock_daily = pd.DataFrame(
        {
            "Close": np.linspace(50, 180, len(daily_idx)),
            "High": np.linspace(51, 181, len(daily_idx)),
            "Low": np.linspace(49, 179, len(daily_idx)),
            "Volume": np.linspace(1000, 5000, len(daily_idx)),
        },
        index=daily_idx,
    )
    stock_weekly = pd.DataFrame({"Close": np.linspace(45, 170, len(weekly_idx))}, index=weekly_idx)
    bench_weekly = pd.DataFrame({"Close": np.linspace(100, 220, len(weekly_idx))}, index=weekly_idx)

    out = swing_lifecycle_frame(stock_daily=stock_daily, stock_weekly=stock_weekly, bench_weekly=bench_weekly)

    assert out.empty is False
    assert "Production Score" in out.columns
    non_na_scores = out["Production Score"].dropna()
    assert non_na_scores.empty is False
    assert non_na_scores.between(-1.0, 1.0).all()


def test_swing_technical_snapshot_contains_core_indicators():
    daily_idx = pd.date_range("2021-01-01", periods=400, freq="B")
    weekly_idx = pd.date_range("2021-01-01", periods=230, freq="W-FRI")

    stock_daily = pd.DataFrame(
        {
            "Close": np.linspace(80, 160, len(daily_idx)),
            "High": np.linspace(81, 161, len(daily_idx)),
            "Low": np.linspace(79, 159, len(daily_idx)),
            "Volume": np.linspace(2000, 9000, len(daily_idx)),
        },
        index=daily_idx,
    )
    stock_weekly = pd.DataFrame({"Close": np.linspace(75, 150, len(weekly_idx))}, index=weekly_idx)
    bench_weekly = pd.DataFrame({"Close": np.linspace(90, 140, len(weekly_idx))}, index=weekly_idx)

    snap = swing_technical_snapshot(stock_daily=stock_daily, stock_weekly=stock_weekly, bench_weekly=bench_weekly)

    assert snap.empty is False
    assert {"Indicator", "Value"}.issubset(snap.columns)
    assert "Production Score" in set(snap["Indicator"])
    assert "Weekly Qualified" in set(snap["Indicator"])
