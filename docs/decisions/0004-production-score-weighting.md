# 0004 - Production Score Weighting And Trade Levels

## Problem
The Swing Production Score was a flat average of seven daily components, which had
three methodology flaws:
- **Momentum was quadruple-counted.** Four of the seven components (RSI state, RSI
  accel, MACD sign, MACD accel) are collinear momentum signals, so an equal-weight
  average let momentum dominate price/volume/volatility ~4:1.
- **Volatility was treated as bullish-only and directionless.** Expansion scored `0/+1`
  regardless of price direction, so a stock breaking down on expanding range still
  scored a positive volatility vote.
- **Two components were asymmetric.** Volume and Volatility could only emit `0/+1`,
  biasing the average upward versus the tri-state momentum/trend components.

There was also no exit/risk readout: the app produced Buy/Hold/Sell labels with no
stop, risk, or target context.

## Decision
- Score by **four equal factor families** (Momentum / Trend / Volume / Volatility), 25%
  each; the four momentum signals share the Momentum family (6.25% each). Weights live
  in `config.PRODUCTION_COMPONENT_WEIGHTS`, applied by `production_score_from_components`.
- **Volume confirmation is direction-gated**: above-average volume votes with the day's
  direction (`+1` up, `-1` down); quiet days vote `0`.
- **Volatility expansion is symmetric** vs its SMA20 (`+1` expanding, `-1` contracting).
- Add `swing_trade_levels`: ATR(14)-multiple and swing-low stops with risk-per-share and
  R-multiple targets, surfaced in a Stock Details "Trade Levels" panel. It is a
  deterministic calculator — no position sizing, no advice.

## Consequences
- Production Score values shift versus prior releases; existing threshold defaults and
  presets are retained, but their distribution differs (documented in `docs/logic.md`).
- Score stays bounded to `[-1, 1]` (weights sum to 1.0, components in `[-1, 1]`).
- `DecisionTrace` component `weight`s now reflect family weighting instead of a flat 1.0.
- New unit tests cover the weighting contract, the direction-aware components, and the
  trade-level arithmetic; offline/deterministic per decision 0002.
