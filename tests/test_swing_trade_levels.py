import pytest
from helpers import load_ohlcv_fixture

from src.signals import swing_trade_levels


def test_levels_status_and_entry():
    daily = load_ohlcv_fixture()
    res = swing_trade_levels(daily, atr_mult=2.0, swing_lookback=10)
    assert res["status"] == "OK"
    assert res["entry"] == pytest.approx(float(daily["Close"].iloc[-1]))
    assert res["atr14"] > 0
    assert len(res["stops"]) == 2


def test_atr_stop_and_targets_are_consistent():
    daily = load_ohlcv_fixture()
    res = swing_trade_levels(daily, atr_mult=2.0, swing_lookback=10, r_multiples=(1.0, 2.0, 3.0))
    entry = res["entry"]
    atr_row = res["stops"][0]
    assert atr_row["type"].startswith("ATR")
    assert atr_row["stop"] == pytest.approx(entry - 2.0 * res["atr14"])

    for row in res["stops"]:
        risk = row["risk_per_share"]
        if risk == risk:  # skip NaN (invalid) rows
            assert row["stop"] == pytest.approx(entry - risk)
            assert row["targets"]["1R"] == pytest.approx(entry + risk)
            assert row["targets"]["2R"] == pytest.approx(entry + 2 * risk)
            assert row["targets"]["3R"] == pytest.approx(entry + 3 * risk)


def test_swing_low_stop_uses_lookback_min():
    daily = load_ohlcv_fixture()
    res = swing_trade_levels(daily, swing_lookback=10)
    swing_row = res["stops"][1]
    assert swing_row["type"].startswith("Swing low")
    assert swing_row["stop"] == pytest.approx(float(daily["Low"].tail(10).min()))


def test_insufficient_history_returns_status():
    daily = load_ohlcv_fixture().head(10)
    res = swing_trade_levels(daily)
    assert res["status"] != "OK"
