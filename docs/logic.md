# Logic Reference

## Portfolio logic

### Health Score inputs
- Monthly regime (`1M`):
  - `Bull` if `Close_month > EMA20_month` and `EMA20_month[t] > EMA20_month[t-3]`
  - `Bear` if `Close_month < EMA20_month` and `EMA20_month[t] < EMA20_month[t-3]`
  - else `Neutral`
- Weekly alignment (`1W`): `Strong` / `Weak` / `Broken`
- Weekly RS: rising if `RS_EMA20_week[t] > RS_EMA20_week[t-3]`
- Weekly momentum state from weekly RSI behavior

Health Score is normalized to `[-1, +1]`.

## Fundamental Universe Screening

When downloading the Frankfurt/Xetra universe, fundamental filters are applied to produce the swing-ready universe. This ensures all screened stocks have strong turnover, positive current ratios, and low debt-to-equity ratios.

Configured thresholds (in `config.yaml`):
- `min_price`: 5.0
- `min_turnover`: 1,000,000 (calculated as `price` * `averageVolume`)
- `min_current_ratio`: 1.0
- `max_debt_to_equity`: 150.0
- `min_market_cap`: 300,000,000

The original complete dataset with fundamentals is saved to `universe_fundamentals.csv`, and the post-screened data goes to `universe_screened.csv`.

## Swing logic

### Weekly hard filter (gate)
Must pass all:
- `EMA20_week > EMA50_week > EMA200_week`
- `EMA20_week[t] > EMA20_week[t-3]`
- `RS_EMA20_week[t] > RS_EMA20_week[t-3]`

### Production Score components (daily)
Seven signals, each in `[-1, 0, +1]`:
- RSI14 state
- RSI acceleration
- MACD histogram sign
- MACD histogram acceleration
- Price vs EMA20
- Volume confirmation — direction-gated: above-average volume scores `+1` on an up
  day, `-1` on a down day, and `0` when volume is below average (quiet days do not vote).
- Volatility expansion — `ATR14%` vs its `SMA20`, symmetric: `+1` expanding, `-1` contracting.

These are combined by **four equal factor families** (25% each) rather than a flat
average, so the collinear momentum oscillators cannot dominate:
- Momentum (25%): RSI state, RSI accel, MACD sign, MACD accel — averaged within the
  family (`6.25%` each)
- Trend (25%): Price vs EMA20
- Volume (25%): Volume confirmation
- Volatility (25%): Volatility expansion

Weights live in `config.PRODUCTION_COMPONENT_WEIGHTS` and are applied by
`production_score_from_components`. Weights sum to `1.0` and every component is in
`[-1, 1]`, so Production Score stays in `[-1, +1]`.

### Trade levels (Stock Details)
`swing_trade_levels` reports two stop candidates from the latest completed daily
candle — an `ATR(14)`-multiple stop and a swing-low (lowest low over `N` bars) stop —
each with risk-per-share and R-multiple targets (`1R`/`2R`/`3R`). This is deterministic
price/ATR arithmetic; it does not size positions or constitute advice.

## Ratings mapping (TradingView-style blocks)

### Oscillators block uses
- RSI(14)
- Momentum(10)
- AO
- CCI(20)
- Stoch (14,3,3)
- MACD histogram sign

### Moving Averages block uses
- Close vs EMA20/EMA50/EMA200
- Close vs SMA20/SMA50/SMA200

### Summary block
- Combined oscillator + moving average signals

### Rating label mapping
- `score >= 0.6` -> `Strong Buy`
- `0.2 <= score < 0.6` -> `Buy`
- `-0.2 < score < 0.2` -> `Neutral`
- `-0.6 < score <= -0.2` -> `Sell`
- `score <= -0.6` -> `Strong Sell`
