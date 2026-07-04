"""Direction-aware behavior of the fixed Volume and Volatility components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import daily_components


def _frame(close, volume, ranges):
    idx = pd.date_range("2022-01-03", periods=len(close), freq="B")
    close = np.asarray(close, dtype=float)
    half = np.asarray(ranges, dtype=float) / 2.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + half,
            "Low": close - half,
            "Close": close,
            "Volume": np.asarray(volume, dtype=float),
        },
        index=idx,
    )


def test_volume_confirm_is_negative_on_high_volume_down_day():
    n = 40
    close = [100.0 + i for i in range(n)]
    close[-1] = close[-2] - 1.0  # down close on the final bar
    volume = [1000.0] * n
    volume[-1] = 100_000.0  # volume spike confirms the down move
    out = daily_components(_frame(close, volume, [2.0] * n))
    assert out["status"] == "OK"
    assert out["components"]["Volume_Confirm"] == -1


def test_volume_confirm_is_positive_on_high_volume_up_day():
    n = 40
    close = [100.0 + i for i in range(n)]  # final bar is an up close
    volume = [1000.0] * n
    volume[-1] = 100_000.0
    out = daily_components(_frame(close, volume, [2.0] * n))
    assert out["status"] == "OK"
    assert out["components"]["Volume_Confirm"] == 1


def test_volatility_expansion_can_be_negative_when_range_contracts():
    n = 40
    close = [100.0 + 0.1 * i for i in range(n)]  # near-flat price
    # Wide ranges early, contracting hard for the final stretch.
    ranges = [4.0] * 30 + list(np.linspace(3.5, 0.4, 10))
    volume = [1000.0] * n
    out = daily_components(_frame(close, volume, ranges))
    assert out["status"] == "OK"
    assert out["components"]["Volatility_Expansion"] == -1
