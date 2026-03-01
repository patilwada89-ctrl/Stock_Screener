# Snapshot TA Screener (Streamlit)

A Streamlit app for snapshot-only technical analysis with three tabs:
- `Portfolio`: long-term health tracking (`Health Score`)
- `Swing`: momentum screening (`Production Score`) with `Action Board` and `Screener Table`
- `Stock Details`: deep-dive for the currently selected stock from Portfolio/Swing

## Context Pack (Read First)

- Agent/contributor guide: `/Users/ashish/vscode/Stock_Screener/AGENTS.md`
- Project context: `/Users/ashish/vscode/Stock_Screener/docs/context.md`
- Architecture/dataflow: `/Users/ashish/vscode/Stock_Screener/docs/overview.md`
- Module contracts: `/Users/ashish/vscode/Stock_Screener/docs/api.md`

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

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
- `/Users/ashish/vscode/Stock_Screener/examples/portfolio.csv`
- `/Users/ashish/vscode/Stock_Screener/examples/watchlist.csv`
- `/Users/ashish/vscode/Stock_Screener/examples/xfra_swing_trading_universe.csv`

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

Configure in `/Users/ashish/vscode/Stock_Screener/src/config.py`.

## Debug mode

Use sidebar `Debug mode` to inspect:
- selected `DecisionTrace` JSON
- date dtype/min/max diagnostics
- intermediate values used by rules

See `/Users/ashish/vscode/Stock_Screener/docs/debugging.md`.

## Tests

```bash
pytest
```

Current tests include rule checks, score naming separation, lifecycle/date prep, and loader compatibility.
