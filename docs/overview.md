# Overview

## Data Flow

1. Symbols + region/benchmark from CSV (`load_universe_csv`).
2. Data fetch (`download_history`) and normalization to OHLCV.
3. Timeframe transforms (`to_weekly`, `to_monthly`).
4. Indicator computation (`src/indicators.py`).
5. Signal/rule evaluation (`src/signals.py`) + ratings (`src/ratings.py`).
6. UI rendering and interactions (`app.py`).

## Responsibility Split

- `app.py`
  - Streamlit pages/tabs, widgets, state, and visual formatting.
- `src/data.py`
  - Input schema handling, benchmark inference, timeframe prep.
- `src/indicators.py`
  - EMA/RSI/MACD/ATR/CCI/Stoch/AO/momentum primitives.
- `src/signals.py`
  - Portfolio and Swing rules, scoring, lifecycle frames, decision trace creation.
- `src/ratings.py`
  - Oscillator/MA/Summary ratings and screener snapshots.
