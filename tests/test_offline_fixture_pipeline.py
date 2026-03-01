from __future__ import annotations

from helpers import load_ohlcv_fixture

from src.data import to_monthly, to_weekly
from src.ratings import technical_ratings
from src.signals import daily_components


def test_offline_fixture_runs_signal_and_ratings_pipeline():
    daily = load_ohlcv_fixture()

    weekly = to_weekly(daily)
    monthly = to_monthly(daily)
    daily_eval = daily_components(daily)
    ratings = technical_ratings(daily)

    assert weekly.empty is False
    assert monthly.empty is False
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(weekly.columns)
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(monthly.columns)
    assert daily_eval["status"] == "OK"
    assert ratings["status"] == "OK"
