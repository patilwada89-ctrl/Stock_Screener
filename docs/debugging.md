# Debugging Guide

Enable sidebar `Debug mode` to inspect diagnostics for the selected stock.

## What Debug mode shows

- `DecisionTrace` JSON:
  - score, decision, qualification, setup type, risk flag/reason
  - weekly rule pass/fail values
  - daily component contributions
  - notes/warnings
- Date diagnostics:
  - dtype, min/max timestamp, monotonicity for daily/weekly dataframes
- Intermediate values:
  - weekly filter tail values
  - latest daily values used by component signals

## Common data issues

1. Missing ticker data (`Download failed or empty data`)
- Check symbol format on Yahoo Finance.

2. Insufficient history
- Weekly or monthly bars may be too short for EMA windows.

3. Missing volume
- Volume-based signals degrade to neutral by design.

4. Odd chart axis
- Lifecycle charts use explicit datetime Date columns.
- If axis still looks wrong, inspect Date diagnostics in Debug mode.

## Quick checks

```bash
pytest
streamlit run app.py
```
