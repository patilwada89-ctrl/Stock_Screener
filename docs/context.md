# Context Pack

## Purpose

- Provide a snapshot-only technical-analysis stock screener in Streamlit.
- Separate Portfolio health and Swing momentum workflows.
- Keep scoring deterministic and reproducible from completed candles.
- Support CSV-based universes with region-aware benchmark handling.
- Keep codebase agent-friendly across Claude/Cursor/Codex.

## Architecture Summary

- `app.py`: Streamlit UI rendering, controls, session state, chart/table presentation.
- `src/`: data normalization, indicators, rule engines, scoring, ratings, trace objects.
- `tests/`: offline deterministic unit tests.

## Conventions

- No `streamlit` imports in `src/`.
- `src/` functions should be pure where possible.
- DataFrame contracts follow OHLCV naming (`Open`, `High`, `Low`, `Close`, `Volume`).
- Swing uses `Production Score`; Portfolio uses `Health Score`.

## Key Commands

```bash
make dev
make test
make lint
make format
make lock
```

## Where To Read Next

- Overview: `/Users/ashish/vscode/Stock_Screener/docs/overview.md`
- API contracts: `/Users/ashish/vscode/Stock_Screener/docs/api.md`
- Decisions: `/Users/ashish/vscode/Stock_Screener/docs/decisions/`
