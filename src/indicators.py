"""Indicator utilities for technical signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as pta
except Exception:  # pragma: no cover - optional fallback in runtime env
    pta = None


def ema(series: pd.Series, length: int) -> pd.Series:
    if pta is not None:
        out = pta.ema(series, length=length)
        if out is not None:
            return out
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    if pta is not None:
        out = pta.sma(series, length=length)
        if out is not None:
            return out
    return series.rolling(length, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    if pta is not None:
        out = pta.rsi(series, length=length)
        if out is not None:
            return out.fillna(50.0)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.fillna(50.0)
    return out


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    if pta is not None:
        out = pta.macd(series, fast=fast, slow=slow, signal=signal)
        if out is not None and not out.empty:
            return out[f"MACDh_{fast}_{slow}_{signal}"]

    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if pta is not None:
        out = pta.atr(high=high, low=low, close=close, length=length)
        if out is not None:
            return out

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    atr_val = atr(high, low, close, length=length)
    return atr_val / close.replace(0, np.nan)


def relative_strength(stock_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    aligned = pd.concat([stock_close, benchmark_close], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["stock", "bench"]
    return aligned["stock"] / aligned["bench"].replace(0, np.nan)


def slope_positive(series: pd.Series, lookback: int = 3) -> bool:
    if len(series.dropna()) <= lookback:
        return False
    return bool(series.iloc[-1] > series.iloc[-(lookback + 1)])
