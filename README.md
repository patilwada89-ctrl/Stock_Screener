# Snapshot TA Screener (Streamlit)

A Streamlit app for snapshot-only technical analysis with three tabs:
- `Portfolio`: long-term health tracking (`Health Score`)
- `Swing`: momentum screening (`Production Score`) with `Action Board` and `Screener Table`
- `Stock Details`: deep-dive for the currently selected stock from Portfolio/Swing

## Context Pack (Read First)

- Agent/contributor guide: `AGENTS.md`
- Project context: `docs/context.md`
- Architecture/dataflow: `docs/overview.md`
- Module contracts: `docs/api.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Commands

```bash
make dev      # streamlit run app.py
make test     # pytest -q
make lint     # ruff check .
make format   # ruff format .
```

## Dependency locking (pip-tools)

Runtime and dev lockfiles are generated from:
- `requirements.in` -> `requirements.txt`
- `requirements-dev.in` -> `requirements-dev.txt`

Recompile locks:

```bash
make lock
# or explicitly:
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
```

Note: `requirements*.txt` are generated files and should not be hand-edited.
Lock files are interpreter-specific; compile them from your target runtime (recommended: Python 3.10/3.11).

## CSV schema

Both `portfolio.csv` and `watchlist.csv` use the same schema.

Required columns:
- `Name`
- `Region` (`EU` or `US`; aliases like `ROW`, `WORLD`, `GLOBAL`, `DE`, `GERMANY` are auto-normalized)
- `SignalTicker`

Optional columns:
- `TradeTicker_DE` (display-only identifier)
- `Benchmark` (override benchmark; if empty it is inferred from region)

Delimiter handling:
- Loader auto-detects `,` and `;`.

Examples:
- `examples/portfolio.csv`
- `examples/watchlist.csv`
- `examples/xfra_swing_trading_universe.csv`

## Score definitions

- `Health Score`:
  - Portfolio-only score from monthly regime + weekly alignment + weekly RS + weekly momentum.
- `Production Score`:
  - Swing-only score from 7 daily momentum components.
  - Used in Swing `Action Board`, Swing lifecycle, and Stock Details decision card.

## Key swing terms

- `Qualified (Weekly)`: all weekly hard filters pass (`EMA20>EMA50>EMA200`, EMA20 slope positive, RS rising).
- `SetupType`: `Breakout`, `Pullback`, or `Trend` from daily setup rules.
- `Decision`: `Buy` / `Hold` / `Sell` from configurable thresholds.

## Stock Details workflow

1. Click a row in Portfolio or Swing.
2. Open `Stock Details` tab.
3. Use `Prev` / `Next` to step through the active ranked list.

`Stock Details` includes:
- Swing decision card (Production Score, Decision, qualification, setup, risk + reason)
- TradingView-style ratings blocks (Oscillators / Summary / Moving Averages)
- Swing lifecycle chart with threshold lines and decision-change markers
- Why-this-decision breakdown (weekly rule checks + daily component signals)
- Latest indicators (concise table + advanced expander)

## Benchmarks

- US default: `SPY`
- EU default: `EXSA.DE`

Configure in `src/config.py`.

## Debug mode

Use sidebar `Debug mode` to inspect:
- selected `DecisionTrace` JSON
- date dtype/min/max diagnostics
- intermediate values used by rules

See `docs/debugging.md`.

## Tests

```bash
make test
```

Current tests include rule checks, score naming separation, lifecycle/date prep, and loader compatibility.
