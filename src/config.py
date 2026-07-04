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

# Production Score weighting: four equal factor families (Momentum / Trend /
# Volume / Volatility), each contributing 25%. The Momentum family bundles four
# collinear oscillator signals that share its 25% (6.25% each) so momentum is not
# quadruple-counted relative to the single-signal families. Weights sum to 1.0
# and every component is in [-1, 1], so the weighted sum is already in [-1, 1].
PRODUCTION_COMPONENT_WEIGHTS = {
    "RSI14_State": 1 / 16,
    "RSI_Accel": 1 / 16,
    "MACD_Hist_Sign": 1 / 16,
    "MACD_Hist_Accel": 1 / 16,
    "Price_vs_EMA20": 1 / 4,
    "Volume_Confirm": 1 / 4,
    "Volatility_Expansion": 1 / 4,
}
