# API Contracts (Living)

## `src/data.py`

- `load_universe_csv(source) -> DataFrame`
  - Required cols: `Name`, `Region`, `SignalTicker`
  - Optional cols: `TradeTicker_DE`, `Benchmark`
  - Returns normalized columns including inferred `Benchmark`.
- `download_history(ticker) -> DataFrame`
  - Returns daily OHLCV with `Close` (Adj Close fallback).
- `to_weekly(df_daily)`, `to_monthly(df_daily)`
  - Input daily OHLCV; output resampled OHLCV.
- `fetch_ticker_data(ticker) -> TickerData`
  - `TickerData(ticker, daily, weekly, monthly, status)`.

## `src/indicators.py`

Indicator primitives all accept pandas Series and return Series unless noted:
- `ema`, `sma`, `rsi`, `macd_hist`, `atr`, `atr_percent`
- `momentum_pct`, `awesome_oscillator`, `cci`, `stochastic_oscillator`
- `relative_strength(stock_close, benchmark_close)`

## `src/signals.py`

- Weekly filter:
  - `weekly_filter_frame`, `evaluate_weekly_hard_filter`
- Daily components and score:
  - `daily_components` (returns component signals + values + `score`)
- Swing:
  - `evaluate_swing_stock`
  - `swing_lifecycle_frame` with `Production Score`
  - `swing_technical_snapshot`
  - `build_swing_decision_trace`
- Portfolio:
  - `evaluate_portfolio_stock`
  - `portfolio_lifecycle_frame` with `Health Score`
- Utility:
  - `decision_from_health_score`, `decision_from_production_score`
  - `rank_qualified`, `apply_custom_weights`, `sort_portfolio_for_risk`

## `src/ratings.py`

- `technical_ratings(daily_df) -> dict`
  - Blocks: `oscillators`, `moving_averages`, `summary`
- `screener_snapshot(daily_df) -> dict`
  - Flattened TradingView-like screener values.

## `src/decision_trace.py`

- Dataclasses:
  - `RuleTrace`
  - `ComponentTrace`
  - `DecisionTrace` with `to_dict()` for debug/UI.
