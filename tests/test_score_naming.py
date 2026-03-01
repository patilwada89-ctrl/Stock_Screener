import numpy as np
import pandas as pd

from src.signals import build_swing_decision_trace, portfolio_lifecycle_frame, swing_lifecycle_frame


def test_portfolio_and_swing_score_columns_are_separated():
    daily_idx = pd.date_range("2021-01-01", periods=260 * 5, freq="B")
    weekly_idx = pd.date_range("2021-01-01", periods=260, freq="W-FRI")
    monthly_idx = pd.date_range("2018-01-31", periods=96, freq="M")

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
    stock_monthly = pd.DataFrame(
        {"Close": np.linspace(70, 240, len(monthly_idx))}, index=monthly_idx
    )

    p = portfolio_lifecycle_frame(
        stock_weekly=stock_weekly, stock_monthly=stock_monthly, bench_weekly=bench_weekly
    )
    s = swing_lifecycle_frame(
        stock_daily=stock_daily, stock_weekly=stock_weekly, bench_weekly=bench_weekly
    )

    assert "Health Score" in p.columns
    assert "Production Score" not in p.columns
    assert "Production Score" in s.columns
    assert "Health Score" not in s.columns


def test_build_swing_decision_trace_contains_rules_and_score_name():
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

    trace = build_swing_decision_trace(
        stock_daily=stock_daily,
        stock_weekly=stock_weekly,
        bench_weekly=bench_weekly,
        buy_threshold=0.3,
        sell_threshold=-0.2,
        signal_ticker="TEST",
        benchmark="SPY",
        name="Test Name",
    )

    assert trace is not None
    assert trace.score_name == "Production Score"
    assert len(trace.rules) == 3
    assert len(trace.components) == 7
