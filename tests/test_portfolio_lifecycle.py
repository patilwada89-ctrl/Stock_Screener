import numpy as np
import pandas as pd

from src.signals import decision_from_health_score, portfolio_lifecycle_frame


def test_decision_thresholds_buy_hold_sell():
    assert decision_from_health_score(0.6, buy_threshold=0.4, sell_threshold=-0.2) == "Buy"
    assert decision_from_health_score(-0.4, buy_threshold=0.4, sell_threshold=-0.2) == "Sell"
    assert decision_from_health_score(0.1, buy_threshold=0.4, sell_threshold=-0.2) == "Hold"


def test_portfolio_lifecycle_frame_outputs_health_score_series():
    weekly_idx = pd.date_range("2021-01-01", periods=230, freq="W-FRI")
    monthly_idx = pd.date_range("2018-01-31", periods=96, freq="M")

    stock_weekly = pd.DataFrame({"Close": np.linspace(100, 220, len(weekly_idx))}, index=weekly_idx)
    bench_weekly = pd.DataFrame({"Close": np.linspace(90, 160, len(weekly_idx))}, index=weekly_idx)
    stock_monthly = pd.DataFrame({"Close": np.linspace(70, 240, len(monthly_idx))}, index=monthly_idx)

    out = portfolio_lifecycle_frame(stock_weekly=stock_weekly, stock_monthly=stock_monthly, bench_weekly=bench_weekly)

    assert out.empty is False
    assert "Health Score" in out.columns
    assert "Risk Flag" in out.columns
    assert out["Health Score"].between(-1.0, 1.0).all()
