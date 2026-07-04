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
- `src/ui_helpers.py`
  - Pure display builders: Altair lifecycle charts and the TradingView-style
    swing-view HTML/SVG (rating gauge, funnel strip, pick cards, heat table).
    No Streamlit import — `app.py` renders these via `st.markdown`.

## Swing tab layout

The Swing tab is a screening funnel, top to bottom:
1. Funnel strip — Universe → Qualified → Buy/Watch/Avoid counts.
2. Top swing picks — ranked cards with a Production-Score gauge, a TradingView
   summary-rating pill, and ATR-based entry/stop/2R levels; each has an Analyze
   button that opens the stock in Stock Details.
3. Full screener — color-coded heat table for scanning, with a selectable table
   (in an expander) for click-through. Recently-lost, weights lab, and excluded
   rows live in collapsed expanders below.
