# Snapshot TA Screener (Dash + Plotly)

A Dash + Plotly app for snapshot-only technical analysis with four tabs:
- `Universe`: two-step Frankfurt/Xetra download → screen workflow (background jobs, config-driven)
- `Portfolio`: long-term health tracking (`Health Score`)
- `Swing`: momentum screening (`Production Score`) — a TradingView-style funnel of
  ranked pick cards (rating gauge + trade levels) over a color-coded screener table
- `Stock Details`: deep-dive for the currently selected stock from Portfolio/Swing,
  including a Trade Levels panel (ATR and swing-low stops with R-multiple targets)

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
make dev      # python app.py
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
This repository CI runs tests on Python 3.10 and 3.11, so dependency bounds in `requirements.in`
must remain compatible with both versions.

## CSV schema

Portfolio, Swing, and Stock Details all run against a single **active universe**:
the most recently screened CSV from the Universe tab if one exists, else the
bundled example (`examples/xfra_swing_trading_universe.csv`). Any CSV loaded
this way must follow the same schema.

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

1. Select a row in the Portfolio or Swing table, or click a pick card's
   `Analyze →` button on the Swing tab.
2. Open the `Stock Details` tab — it renders for whichever stock was last selected.

`Stock Details` includes:
- Swing decision card (Production Score, Decision, qualification, setup, risk + reason)
- Trade levels (ATR and swing-low stops with risk-per-share and R-multiple targets)
- TradingView-style ratings blocks (Oscillators / Summary / Moving Averages)
- Why-this-decision breakdown (weekly rule checks + daily component signals)
- Swing lifecycle chart (Production Score history overlaid with weekly price)

## Benchmarks

- US default: `SPY`
- EU default: `EXSA.DE`

Configure in `src/config.py`.

## Debugging

There is no separate debug-mode UI toggle. The "Why this decision" panel on
`Stock Details` surfaces the weekly-rule and daily-component values driving a
decision directly; `DecisionTrace.debug` carries additional intermediate
values (weekly filter tail, raw daily values) for anyone inspecting results
programmatically. See `docs/debugging.md` for common data issues.

## Tests

```bash
make test
```

Current tests include rule checks, score naming separation, lifecycle/date prep, and loader compatibility.
