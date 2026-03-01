# AGENTS Guide

Stock_Screener is a Streamlit snapshot-only technical-analysis app with three user flows: Portfolio health tracking, Swing momentum screening, and Stock Details diagnostics. The product intent is deterministic snapshot evaluation using previous completed candles and region-native benchmarks; no forecasting, no prediction targets, and no historical journaling as a required feature.

## Architecture Map

- `/Users/ashish/vscode/Stock_Screener/app.py`
  - Streamlit-only UI layer.
  - Page state/session handling, table rendering, controls, chart composition.
  - Calls pure computation functions from `src/`.
- `/Users/ashish/vscode/Stock_Screener/src/`
  - Domain logic and data operations only.
  - `data.py`: CSV loading, normalization, download helpers, resampling.
  - `indicators.py`: indicator primitives.
  - `signals.py`: rule logic, scoring, lifecycle outputs, decision traces.
  - `ratings.py`: TradingView-style rating computations.
  - `config.py`: constants.
  - **No Streamlit imports in `src/`.**
- `/Users/ashish/vscode/Stock_Screener/tests/`
  - Offline deterministic unit tests.
  - Network calls are excluded from default test runs.

## Golden Rules

- Snapshot-only TA: compute from last completed candle per timeframe.
- No forecasting/prediction logic.
- Keep `src/` functions pure and testable.
- Never import `streamlit` in `src/`; UI belongs in `app.py`.
- Preserve existing behavior unless a change is explicitly requested or needed for reproducibility/test stability.

## Runbook

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make dev
```

Useful commands:

```bash
make test
make lint
make format
make lock
```

## Definition Of Done

A PR is done only when:
- `make lint` and `make test` pass.
- behavior changes are covered by tests where feasible.
- docs are updated when behavior/API/contracts change:
  - `README.md` and/or files under `/Users/ashish/vscode/Stock_Screener/docs/`
  - update `AGENTS.md` if contributor/agent workflow assumptions changed.
