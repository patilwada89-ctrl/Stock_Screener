"""Technical ratings for Swing views (TradingView-style blocks)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    awesome_oscillator,
    cci,
    ema,
    macd_hist,
    momentum_pct,
    rsi,
    sma,
    stochastic_oscillator,
)


def rating_label(score: float) -> str:
    if pd.isna(score):
        return "Neutral"
    if score >= 0.6:
        return "Strong Buy"
    if score >= 0.2:
        return "Buy"
    if score <= -0.6:
        return "Strong Sell"
    if score <= -0.2:
        return "Sell"
    return "Neutral"


def _signal_from_value(value: float, buy_if_above: float = 0.0, sell_if_below: float = 0.0) -> int:
    if pd.isna(value):
        return 0
    if value > buy_if_above:
        return 1
    if value < sell_if_below:
        return -1
    return 0


def _rating_block(signals: list[int]) -> dict[str, Any]:
    if not signals:
        return {"score": 0.0, "label": "Neutral", "buy": 0, "neutral": 0, "sell": 0}
    arr = np.array(signals, dtype=float)
    score = float(np.clip(arr.mean(), -1.0, 1.0))
    return {
        "score": score,
        "label": rating_label(score),
        "buy": int((arr > 0).sum()),
        "neutral": int((arr == 0).sum()),
        "sell": int((arr < 0).sum()),
    }


def _latest(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.iloc[-1])


def oscillator_snapshot(daily_df: pd.DataFrame) -> dict[str, Any]:
    close = daily_df["Close"]
    high = daily_df.get("High", close)
    low = daily_df.get("Low", close)

    rsi14 = rsi(close, 14)
    mom10 = momentum_pct(close, 10)
    ao = awesome_oscillator(high, low)
    cci20 = cci(high, low, close, 20)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, 14, 3, 3)
    hist = macd_hist(close)

    rsi_v = _latest(rsi14)
    mom_v = _latest(mom10)
    ao_v = _latest(ao)
    cci_v = _latest(cci20)
    stoch_k_v = _latest(stoch_k)
    stoch_d_v = _latest(stoch_d)
    hist_v = _latest(hist)

    sig_rsi = 1 if pd.notna(rsi_v) and rsi_v < 30 else (-1 if pd.notna(rsi_v) and rsi_v > 70 else 0)
    sig_mom = _signal_from_value(mom_v, buy_if_above=0.0, sell_if_below=0.0)
    sig_ao = _signal_from_value(ao_v, buy_if_above=0.0, sell_if_below=0.0)
    sig_cci = 1 if pd.notna(cci_v) and cci_v > 100 else (-1 if pd.notna(cci_v) and cci_v < -100 else 0)
    sig_stoch = 0
    if pd.notna(stoch_k_v) and pd.notna(stoch_d_v):
        if stoch_k_v < 20 and stoch_k_v > stoch_d_v:
            sig_stoch = 1
        elif stoch_k_v > 80 and stoch_k_v < stoch_d_v:
            sig_stoch = -1
    sig_macd = _signal_from_value(hist_v, buy_if_above=0.0, sell_if_below=0.0)

    signals = [sig_rsi, sig_mom, sig_ao, sig_cci, sig_stoch, sig_macd]
    block = _rating_block(signals)
    block["signals"] = {
        "RSI14": sig_rsi,
        "Momentum10": sig_mom,
        "AO": sig_ao,
        "CCI20": sig_cci,
        "Stoch": sig_stoch,
        "MACD_Hist": sig_macd,
    }
    block["values"] = {
        "RSI14": rsi_v,
        "Momentum10": mom_v,
        "AO": ao_v,
        "CCI20": cci_v,
        "StochK": stoch_k_v,
        "StochD": stoch_d_v,
        "MACD_Hist": hist_v,
    }
    return block


def moving_average_snapshot(daily_df: pd.DataFrame) -> dict[str, Any]:
    close = daily_df["Close"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    close_v = _latest(close)
    ema20_v = _latest(ema20)
    ema50_v = _latest(ema50)
    ema200_v = _latest(ema200)
    sma20_v = _latest(sma20)
    sma50_v = _latest(sma50)
    sma200_v = _latest(sma200)

    def above_ma(ma_val: float) -> int:
        if pd.isna(close_v) or pd.isna(ma_val):
            return 0
        if close_v > ma_val:
            return 1
        if close_v < ma_val:
            return -1
        return 0

    signals = [
        above_ma(ema20_v),
        above_ma(ema50_v),
        above_ma(ema200_v),
        above_ma(sma20_v),
        above_ma(sma50_v),
        above_ma(sma200_v),
    ]

    block = _rating_block(signals)
    block["signals"] = {
        "Close_vs_EMA20": signals[0],
        "Close_vs_EMA50": signals[1],
        "Close_vs_EMA200": signals[2],
        "Close_vs_SMA20": signals[3],
        "Close_vs_SMA50": signals[4],
        "Close_vs_SMA200": signals[5],
    }
    block["values"] = {
        "Close": close_v,
        "EMA20": ema20_v,
        "EMA50": ema50_v,
        "EMA200": ema200_v,
        "SMA20": sma20_v,
        "SMA50": sma50_v,
        "SMA200": sma200_v,
    }
    return block


def technical_ratings(daily_df: pd.DataFrame) -> dict[str, Any]:
    if daily_df.empty or len(daily_df) < 40:
        return {
            "status": "Insufficient daily data for ratings",
            "oscillators": _rating_block([]),
            "moving_averages": _rating_block([]),
            "summary": _rating_block([]),
            "values": {},
        }

    osc = oscillator_snapshot(daily_df)
    ma = moving_average_snapshot(daily_df)
    summary_signals = list(osc["signals"].values()) + list(ma["signals"].values())
    summary = _rating_block(summary_signals)

    values: dict[str, Any] = {}
    values.update(osc.get("values", {}))
    values.update(ma.get("values", {}))

    return {
        "status": "OK",
        "oscillators": osc,
        "moving_averages": ma,
        "summary": summary,
        "values": values,
    }


def screener_snapshot(daily_df: pd.DataFrame) -> dict[str, Any]:
    ratings = technical_ratings(daily_df)
    if ratings.get("status") != "OK":
        return {
            "status": str(ratings.get("status", "Ratings unavailable")),
            "Summary Rating": "Neutral",
            "MA Rating": "Neutral",
            "Osc Rating": "Neutral",
            "Summary Score": 0.0,
            "MA Score": 0.0,
            "Osc Score": 0.0,
            "RSI(14)": np.nan,
            "Momentum(10)": np.nan,
            "AO": np.nan,
            "CCI(20)": np.nan,
            "Stoch %K": np.nan,
            "Stoch %D": np.nan,
        }

    values = ratings.get("values", {})
    return {
        "status": "OK",
        "Summary Rating": str(ratings["summary"]["label"]),
        "MA Rating": str(ratings["moving_averages"]["label"]),
        "Osc Rating": str(ratings["oscillators"]["label"]),
        "Summary Score": float(ratings["summary"]["score"]),
        "MA Score": float(ratings["moving_averages"]["score"]),
        "Osc Score": float(ratings["oscillators"]["score"]),
        "RSI(14)": values.get("RSI14", np.nan),
        "Momentum(10)": values.get("Momentum10", np.nan),
        "AO": values.get("AO", np.nan),
        "CCI(20)": values.get("CCI20", np.nan),
        "Stoch %K": values.get("StochK", np.nan),
        "Stoch %D": values.get("StochD", np.nan),
    }
