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

## Swing logic

### Weekly hard filter (gate)
Must pass all:
- `EMA20_week > EMA50_week > EMA200_week`
- `EMA20_week[t] > EMA20_week[t-3]`
- `RS_EMA20_week[t] > RS_EMA20_week[t-3]`

### Production Score components (daily)
Equal-weight components in `[-1,0,+1]` (with some confirmation-only `0/+1`):
- RSI14 state
- RSI acceleration
- MACD histogram sign
- MACD histogram acceleration
- Price vs EMA20
- Volume confirmation
- Volatility expansion (ATR14% vs SMA20)

Production Score is normalized to `[-1, +1]`.

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
