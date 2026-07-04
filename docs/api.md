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

- `prepare_lifecycle_frame(lifecycle, score_col, decision_col, window=104)` — pure tidy
  for lifecycle charts (framework-agnostic).

## `src/plotly_charts.py`

- `rating_gauge_figure(score)` — TradingView-style gauge (needle maps score in `[-1, 1]`).
- `lifecycle_score_figure(dates, scores, buy, sell, score_name)` — score line + thresholds.

## Universe workflow

- `src/frankfurt_universe.py` — reused pipeline: `build_universe`, `resolve_tickers`,
  `fetch_fundamentals`, `apply_filter`, `export(df, cfg, out_path=None)`, plus `Config`.
- `src/universe_config.py` — `load_universe_config(path=None) -> Config` from `config.yaml`.
- `src/universe_jobs.py` — `start_download_job(cfg)`, `start_screen_job(cfg, input_csv)`,
  `job_status(cfg, job)` (background threads + `data/job_status.json`).
- `src/universe_adapter.py` — `screened_csv_to_universe(path)` → app universe schema.

## `src/decision_trace.py`

- Dataclasses:
  - `RuleTrace`
  - `ComponentTrace`
  - `DecisionTrace` with `to_dict()` for debug/UI.
