import numpy as np
import pandas as pd

from src.signals import evaluate_weekly_hard_filter


def test_weekly_hard_filter_passes_on_trending_stock_with_rising_rs():
    idx = pd.date_range("2021-01-01", periods=230, freq="W-FRI")
    stock = pd.Series(np.linspace(50, 200, len(idx)), index=idx)
    bench = pd.Series(np.linspace(100, 220, len(idx)), index=idx)

    result = evaluate_weekly_hard_filter(stock, bench)

    assert result["qualified"] is True
    assert result["rule_alignment"] is True
    assert result["rule_slope"] is True
    assert result["rule_rs"] is True


def test_weekly_hard_filter_fails_when_recent_structure_breaks():
    idx = pd.date_range("2021-01-01", periods=230, freq="W-FRI")
    stock = pd.Series(np.linspace(50, 200, len(idx)), index=idx)
    stock.iloc[-15:] = np.linspace(180, 130, 15)
    bench = pd.Series(np.linspace(100, 220, len(idx)), index=idx)

    result = evaluate_weekly_hard_filter(stock, bench)

    assert result["qualified"] is False
    assert (
        result["rule_alignment"] is False
        or result["rule_slope"] is False
        or result["rule_rs"] is False
    )
