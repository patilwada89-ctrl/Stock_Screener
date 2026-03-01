# Snapshot-Only TA Stock Screener

Streamlit app for technical-analysis screening using only current and recent candles (no forecasting model).

The app has 3 tabs:
- `Portfolio`: long-term portfolio health monitor
- `Swing`: momentum swing screener with decision board
- `Stock Details`: lifecycle + technical breakdown for the selected stock

## What This Project Does

- Loads stock universes from CSV (example file or upload).
- Downloads price history from Yahoo Finance (`yfinance`).
- Builds daily, weekly, and monthly frames.
- Evaluates:
  - Portfolio health score and risk flag.
  - Swing weekly hard-filter qualification + production score.
- Produces `Buy` / `Hold` / `Sell` decisions from adjustable thresholds.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input CSV Format

Required columns:
- `Name`
- `Region`
- `SignalTicker`

Optional columns:
- `TradeTicker_DE`
- `Benchmark`

Notes:
- `Region` supports `EU` and `US` plus aliases such as `ROW`, `WORLD`, `GLOBAL`, `DE`, `GERMANY`, `USA`, etc.
- If `Benchmark` is empty, it is inferred from region.
- CSV delimiter is auto-detected, so both comma and semicolon exports are supported.

Example files:
- `examples/portfolio.csv`
- `examples/watchlist.csv`
- `examples/xfra_swing_trading_universe.csv`

In the current UI:
- Portfolio tab default uses `examples/portfolio.csv`
- Swing tab default uses `examples/xfra_swing_trading_universe.csv`

## Scoring Model Summary

### Portfolio health score
Portfolio score is normalized to `[-1, +1]` from:
- Monthly regime (`Bull`, `Neutral`, `Bear`)
- Weekly alignment (`Strong`, `Weak`, `Broken`)
- Weekly relative-strength direction (`Rising` / `Falling`)
- Weekly RSI momentum state

Risk flag is derived from regime/alignment/RS:
- `Breakdown`
- `Watch`
- `OK`

### Swing production score
Only stocks passing weekly hard filter are scored:
- `EMA20 > EMA50 > EMA200`
- `EMA20[t] > EMA20[t-3]`
- `RS_EMA20[t] > RS_EMA20[t-3]`

Qualified stocks receive an equal-weight production score in `[-1, +1]` from 7 daily components:
- `RSI14_State`
- `RSI_Accel`
- `MACD_Hist_Sign`
- `MACD_Hist_Accel`
- `Price_vs_EMA20`
- `Volume_Confirm`
- `Volatility_Expansion`

The Swing tab also includes:
- Action Board with threshold-based decision labels
- Recently Lost Alignment list (default lookback: 6 weeks)
- Weights Lab (custom ranking preview)

## Decision Thresholds

- Portfolio default thresholds: Sell `<= -0.25`, Buy `>= 0.35`
- Swing default thresholds: Sell `<= -0.20`, Buy `>= 0.30`

Thresholds are adjustable in the UI and reused by Stock Details where possible.

## Data, Resampling, and Validation

- Data source: `yfinance`
- Download window: `5y`, interval `1d`
- Price preference: `Adj Close` fallback to `Close`
- Weekly bars: resampled to `W-FRI`
- Monthly bars: resampled to month-end
- Incomplete current period bars are dropped
- Download cache uses Streamlit cache (`ttl=86400`)

Minimum data requirements per ticker:
- Weekly history: at least 210 bars
- Monthly history: at least 24 bars

## Benchmarks

Defaults in `src/config.py`:
- US: `SPY`
- EU: `EXSA.DE`

You can override benchmark per row in the CSV using the `Benchmark` column.

## Project Structure

```text
app.py                  # Streamlit UI and tab rendering
src/config.py           # constants (benchmarks, TTL, schema, sort order)
src/data.py             # CSV loading, normalization, download, resampling
src/indicators.py       # EMA/RSI/MACD/ATR/RS helpers
src/signals.py          # screening logic, scoring, lifecycle frames
src/ui_helpers.py       # file picker + table formatting helpers
tests/                  # unit tests for core logic
examples/               # sample CSV universes
```

## Testing

```bash
pytest
```

Current test coverage includes:
- CSV loading and region normalization
- Weekly hard filter behavior
- Monthly regime and lifecycle calculations
- Score normalization bounds
- Decision helper behavior

## Common Issues

- `Download failed or empty data`: usually invalid ticker, temporary Yahoo issue, or delisted symbol.
- `Insufficient weekly history (<210 bars)` or `Insufficient monthly history`: ticker has too little history for model rules.
- `Benchmark issue`: benchmark symbol is invalid or unavailable.

Use the `Status / Excluded` section in Swing for filter failures and diagnostics.
