# 0005. Fundamental Swing Screener

Date: 2026-07-04

## Context

The `Stock_Screener` app initially evaluated technicals to filter the universe, but lacked a robust fundamental gate to filter out low-liquidity or fundamentally unhealthy companies prior to running technical analysis. We needed a set of fundamental screening criteria that could be calculated purely from `yfinance` data to act as a "safety net" for swing trading.

## Decision

We integrated a professional Analyst's fundamental screener into the universe builder. The following metrics are now required for a stock to pass the initial screen:
- **Minimum Turnover**: >= $1,000,000 (calculated dynamically as `price * averageVolume`). This guarantees sufficient liquidity to enter and exit swing trades smoothly without slippage.
- **Current Ratio**: >= 1.0. This ensures the company can cover its short-term liabilities with short-term assets, mitigating bankruptcy risk during the hold period.
- **Debt-to-Equity Ratio**: <= 150%. This filters out heavily over-leveraged companies.
- **Minimum Price**: >= $5.0. Penny stocks are excluded to reduce volatility and spread costs.
- **Minimum Market Cap**: >= $300,000,000. Excludes micro-cap companies for stability.

We also output the full pre-screened dataset (`universe_fundamentals.csv`) so that the raw fundamental data is preserved before being filtered into the final `universe_screened.csv` output.

## Consequences

- The universe builder process now successfully filters out low-quality/illiquid names, resulting in a cleaner and more actionable swing trading universe.
- Adding fields to `yfinance` fetch queries slightly increases payload size but does not incur additional network round-trips.
- The `config.yaml` file now centrally manages these thresholds, allowing future tuning without touching Python code.
