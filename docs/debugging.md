# Debugging Guide

There is no separate debug-mode UI toggle in the Dash app. To inspect why a
stock got a particular decision:

- The **Stock Details** tab's "Why this decision" panel shows the weekly
  hard-filter rule values and daily component signals directly.
- `src.signals.build_swing_decision_trace` returns the full `DecisionTrace`,
  whose `.debug` dict carries additional intermediate values (weekly filter
  tail rows, raw daily indicator values) for anyone inspecting results from a
  script or the test suite rather than the UI.

## Common data issues

1. Missing ticker data (`Download failed or empty data`)
- Check symbol format on Yahoo Finance.

2. Insufficient history
- Weekly or monthly bars may be too short for EMA windows.

3. Missing volume
- Volume-based signals degrade to neutral by design.

4. Odd chart axis
- Lifecycle charts use explicit datetime `Date` columns (see
  `src.ui_helpers.prepare_lifecycle_frame`).
- If axis still looks wrong, inspect the `DecisionTrace.debug` values above.

## Quick checks

```bash
make test
make dev
```
