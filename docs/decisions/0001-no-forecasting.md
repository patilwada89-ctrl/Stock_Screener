# 0001 - No Forecasting

## Problem
The app could drift into predictive features that are not aligned with snapshot TA intent.

## Decision
Keep analysis snapshot-only and rule-based on completed candles; no forecasting modules.

## Consequences
- Deterministic outputs.
- Easier testing and reproducibility.
- Users should not interpret signals as predictions.
