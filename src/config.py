"""Application-wide configuration constants."""

from __future__ import annotations

DOWNLOAD_TTL_SECONDS = 86_400
YF_LOOKBACK_PERIOD = "5y"
YF_INTERVAL = "1d"

US_BENCHMARK = "SPY"
# STOXX Europe 600 proxy on Yahoo Finance. Override via CSV Benchmark column.
EU_BENCHMARK = "EXSA.DE"

REGION_TO_BENCHMARK = {
    "US": US_BENCHMARK,
    "EU": EU_BENCHMARK,
}

REQUIRED_CSV_COLUMNS = ["Name", "Region", "SignalTicker"]
OPTIONAL_CSV_COLUMNS = ["TradeTicker_DE", "Benchmark"]

WEEKLY_RECENTLY_LOST_LOOKBACK = 6

RISK_SORT_ORDER = {
    "Breakdown": 0,
    "Watch": 1,
    "OK": 2,
}
