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
  - `daily_components` (returns component signals + values + `score`); `score` is the
    family-weighted Production Score (Volume is direction-gated, Volatility is symmetric)
  - `production_score_from_components(components) -> float` — applies the four equal
    factor families in `config.PRODUCTION_COMPONENT_WEIGHTS`, result in `[-1, 1]`
- Swing:
  - `evaluate_swing_stock`
  - `swing_lifecycle_frame` with `Production Score`
  - `swing_technical_snapshot`
  - `build_swing_decision_trace`
  - `swing_trade_levels(daily_df, atr_mult=2.0, swing_lookback=10, r_multiples=(1,2,3)) -> dict`
    — ATR and swing-low stops with risk-per-share and R-multiple targets
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

## `src/ui_helpers.py`

Pure display builders (no Streamlit import); `app.py` renders the returned objects/markup.
- Charts: `prepare_lifecycle_frame`, `lifecycle_score_chart`, `decision_change_points`,
  `clean_display_df`
- TradingView-style swing view (return HTML/SVG strings):
  - `rating_gauge_svg(score)` — semicircular gauge, needle maps score in `[-1, 1]`
  - `funnel_strip_html(universe, qualified, buy, watch, avoid)`
  - `swing_pick_card_html(*, ticker, name, region, decision, prod_score, tv_rating, setup, risk_flag, entry, stop, target_2r, featured)`
  - `screener_heat_table_html(rows)` — color-coded display-only screener table

## `src/decision_trace.py`

- Dataclasses:
  - `RuleTrace`
  - `ComponentTrace`
  - `DecisionTrace` with `to_dict()` for debug/UI.
