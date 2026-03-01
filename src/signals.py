"""Signal definitions and scoring logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.indicators import atr_percent, ema, macd_hist, relative_strength, rsi, sma


def classify_monthly_regime(close_t: float, ema20_t: float, ema20_t_3: float) -> str:
    if pd.isna(close_t) or pd.isna(ema20_t) or pd.isna(ema20_t_3):
        return "Neutral"
    if close_t > ema20_t and ema20_t > ema20_t_3:
        return "Bull"
    if close_t < ema20_t and ema20_t < ema20_t_3:
        return "Bear"
    return "Neutral"


def classify_weekly_alignment(ema20_t: float, ema50_t: float, ema200_t: float) -> str:
    if pd.isna(ema20_t) or pd.isna(ema50_t) or pd.isna(ema200_t):
        return "Broken"
    if ema20_t > ema50_t and ema50_t > ema200_t:
        return "Strong"
    if ema20_t > ema50_t and not (ema50_t > ema200_t):
        return "Weak"
    if ema20_t < ema50_t:
        return "Broken"
    return "Weak"


def classify_weekly_momentum(rsi_t: float, rsi_t_3: float) -> str:
    if pd.isna(rsi_t) or pd.isna(rsi_t_3):
        return "Neutral"
    if rsi_t > 50 and rsi_t > rsi_t_3:
        return "Strengthening"
    if rsi_t < 50 and rsi_t < rsi_t_3:
        return "Weakening"
    return "Neutral"


def classify_risk_flag(monthly_regime: str, weekly_alignment: str, rs_rising: bool) -> str:
    rs_falling = not rs_rising
    if monthly_regime == "Bear" or (weekly_alignment == "Broken" and rs_falling):
        return "Breakdown"
    if monthly_regime == "Bull" and weekly_alignment != "Broken" and rs_rising:
        return "OK"
    if monthly_regime == "Neutral" or rs_falling or weekly_alignment == "Weak":
        return "Watch"
    return "Watch"


def normalize_score(values: list[float]) -> float:
    if not values:
        return 0.0
    score = float(np.mean(values))
    return float(np.clip(score, -1.0, 1.0))


def weekly_filter_frame(stock_weekly_close: pd.Series, bench_weekly_close: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [stock_weekly_close.rename("stock"), bench_weekly_close.rename("bench")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    e20 = ema(aligned["stock"], 20)
    e50 = ema(aligned["stock"], 50)
    e200 = ema(aligned["stock"], 200)
    rs = relative_strength(aligned["stock"], aligned["bench"])
    rs_ema20 = ema(rs, 20)

    out = pd.DataFrame(index=aligned.index)
    out["EMA20"] = e20
    out["EMA50"] = e50
    out["EMA200"] = e200
    out["RS_EMA20"] = rs_ema20
    out["rule_alignment"] = (e20 > e50) & (e50 > e200)
    out["rule_slope"] = e20 > e20.shift(3)
    out["rule_rs"] = rs_ema20 > rs_ema20.shift(3)
    out["qualified"] = out["rule_alignment"] & out["rule_slope"] & out["rule_rs"]
    return out


def evaluate_weekly_hard_filter(stock_weekly_close: pd.Series, bench_weekly_close: pd.Series) -> dict[str, Any]:
    wf = weekly_filter_frame(stock_weekly_close, bench_weekly_close)
    if wf.empty:
        return {
            "qualified": False,
            "rule_alignment": False,
            "rule_slope": False,
            "rule_rs": False,
            "filter_frame": wf,
        }
    latest = wf.iloc[-1]
    return {
        "qualified": bool(latest["qualified"]),
        "rule_alignment": bool(latest["rule_alignment"]),
        "rule_slope": bool(latest["rule_slope"]),
        "rule_rs": bool(latest["rule_rs"]),
        "filter_frame": wf,
    }


def _component_buy_neutral_sell(value: float, buy_threshold: float, sell_threshold: float) -> int:
    if pd.isna(value):
        return 0
    if value > buy_threshold:
        return 1
    if value < sell_threshold:
        return -1
    return 0


def _component_acceleration(current: float, past: float) -> int:
    if pd.isna(current) or pd.isna(past):
        return 0
    if current > past:
        return 1
    if current < past:
        return -1
    return 0


def daily_components(daily_df: pd.DataFrame) -> dict[str, Any]:
    if len(daily_df) < 30:
        return {"status": "Insufficient daily history (<30 bars)"}

    close = daily_df["Close"]
    volume = daily_df["Volume"] if "Volume" in daily_df.columns else pd.Series(index=daily_df.index, dtype=float)
    high = daily_df["High"] if "High" in daily_df.columns else close
    low = daily_df["Low"] if "Low" in daily_df.columns else close

    rsi14 = rsi(close, 14)
    hist = macd_hist(close)
    ema20_d = ema(close, 20)
    vol_sma20 = sma(volume, 20)
    atr14_pct = atr_percent(high, low, close, 14)
    atr14_pct_sma20 = sma(atr14_pct, 20)
    high20_prev = close.rolling(20, min_periods=20).max().shift(1)

    t = -1
    t3 = -4

    rsi_state = _component_buy_neutral_sell(float(rsi14.iloc[t]), buy_threshold=55, sell_threshold=45)
    rsi_accel = _component_acceleration(float(rsi14.iloc[t]), float(rsi14.iloc[t3]))

    hist_t = float(hist.iloc[t])
    hist_t3 = float(hist.iloc[t3])
    if pd.isna(hist_t):
        macd_sign = 0
    elif hist_t > 0:
        macd_sign = 1
    elif hist_t < 0:
        macd_sign = -1
    else:
        macd_sign = 0
    macd_accel = _component_acceleration(hist_t, hist_t3)

    close_t = float(close.iloc[t])
    ema20_t = float(ema20_d.iloc[t])
    if pd.isna(close_t) or pd.isna(ema20_t):
        price_vs_ema20 = 0
    elif close_t > ema20_t:
        price_vs_ema20 = 1
    elif close_t < ema20_t:
        price_vs_ema20 = -1
    else:
        price_vs_ema20 = 0

    vol_t = volume.iloc[t] if len(volume) else np.nan
    vol_sma_t = vol_sma20.iloc[t] if len(vol_sma20) else np.nan
    volume_confirm = 1 if pd.notna(vol_t) and pd.notna(vol_sma_t) and vol_t > vol_sma_t else 0

    atr_pct_t = atr14_pct.iloc[t]
    atr_pct_sma_t = atr14_pct_sma20.iloc[t]
    volatility_expansion = 1 if pd.notna(atr_pct_t) and pd.notna(atr_pct_sma_t) and atr_pct_t > atr_pct_sma_t else 0

    values = [
        rsi_state,
        rsi_accel,
        macd_sign,
        macd_accel,
        price_vs_ema20,
        volume_confirm,
        volatility_expansion,
    ]

    score = normalize_score(values)

    close_20h = float(high20_prev.iloc[t]) if pd.notna(high20_prev.iloc[t]) else np.nan
    breakout = pd.notna(close_20h) and close_t > close_20h
    pullback = (
        pd.notna(ema20_t)
        and ema20_t != 0
        and close_t > ema20_t
        and abs(close_t - ema20_t) / ema20_t <= 0.01
        and float(rsi14.iloc[t]) > float(rsi14.iloc[t3])
    )
    if breakout:
        setup_type = "Breakout"
    elif pullback:
        setup_type = "Pullback"
    else:
        setup_type = "Trend"

    return {
        "status": "OK",
        "score": score,
        "setup_type": setup_type,
        "components": {
            "RSI14_State": rsi_state,
            "RSI_Accel": rsi_accel,
            "MACD_Hist_Sign": macd_sign,
            "MACD_Hist_Accel": macd_accel,
            "Price_vs_EMA20": price_vs_ema20,
            "Volume_Confirm": volume_confirm,
            "Volatility_Expansion": volatility_expansion,
        },
        "values": {
            "Close": close_t,
            "RSI14": float(rsi14.iloc[t]),
            "MACD_Hist": hist_t,
            "EMA20_D": ema20_t,
            "Volume": float(vol_t) if pd.notna(vol_t) else np.nan,
            "Volume_SMA20": float(vol_sma_t) if pd.notna(vol_sma_t) else np.nan,
            "ATR14_Pct": float(atr_pct_t) if pd.notna(atr_pct_t) else np.nan,
            "ATR14_Pct_SMA20": float(atr_pct_sma_t) if pd.notna(atr_pct_sma_t) else np.nan,
            "Close_20D_High_Prev": float(close_20h) if pd.notna(close_20h) else np.nan,
        },
    }


def classify_recently_lost(filter_frame: pd.DataFrame, lookback_weeks: int = config.WEEKLY_RECENTLY_LOST_LOOKBACK) -> tuple[bool, str]:
    if filter_frame.empty or len(filter_frame) < 2:
        return False, ""
    currently_qualified = bool(filter_frame["qualified"].iloc[-1])
    if currently_qualified:
        return False, ""

    recent_window = filter_frame.iloc[max(0, len(filter_frame) - (lookback_weeks + 1)) : -1]
    previously_qualified = recent_window[recent_window["qualified"]]
    if previously_qualified.empty:
        return False, ""
    return True, previously_qualified.index[-1].strftime("%Y-%m-%d")


def failed_weekly_rules_text(filter_frame: pd.DataFrame) -> str:
    if filter_frame.empty:
        return "No weekly signal data"
    latest = filter_frame.iloc[-1]
    failed = []
    if not bool(latest["rule_alignment"]):
        failed.append("EMA20>EMA50>EMA200")
    if not bool(latest["rule_slope"]):
        failed.append("EMA20 slope")
    if not bool(latest["rule_rs"]):
        failed.append("RS rising")
    if not failed:
        return "-"
    return ", ".join(failed)


def evaluate_swing_stock(
    meta: pd.Series,
    stock_daily: pd.DataFrame,
    stock_weekly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    stock_status: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Name": meta["Name"],
        "Region": meta["Region"],
        "SignalTicker": meta["SignalTicker"],
        "TradeTicker_DE": meta.get("TradeTicker_DE", ""),
        "Benchmark": meta["Benchmark"],
        "Status": stock_status,
        "Qualified": False,
    }

    if stock_status != "OK":
        return result

    weekly_eval = evaluate_weekly_hard_filter(stock_weekly["Close"], bench_weekly["Close"])
    filter_frame = weekly_eval["filter_frame"]
    recently_lost, last_qualified_week = classify_recently_lost(filter_frame)

    result.update(
        {
            "Rule_Alignment": weekly_eval["rule_alignment"],
            "Rule_Slope": weekly_eval["rule_slope"],
            "Rule_RS": weekly_eval["rule_rs"],
            "Qualified": weekly_eval["qualified"],
            "RecentlyLost": recently_lost,
            "LastQualifiedWeek": last_qualified_week,
            "FailedRules": failed_weekly_rules_text(filter_frame),
            "_filter_frame": filter_frame,
        }
    )

    if not weekly_eval["qualified"]:
        result["Status"] = "Failed weekly hard filter"
        return result

    daily_eval = daily_components(stock_daily)
    if daily_eval["status"] != "OK":
        result["Status"] = daily_eval["status"]
        return result

    result.update(
        {
            "Status": "OK",
            "Price": round(daily_eval["values"]["Close"], 4),
            "SetupType": daily_eval["setup_type"],
            "ProductionScore": daily_eval["score"],
        }
    )
    result.update(daily_eval["components"])

    return result


def _health_score(
    monthly_regime: str,
    weekly_alignment: str,
    rs_rising: bool,
    weekly_momentum_state: str,
) -> float:
    regime_map = {"Bull": 1, "Neutral": 0, "Bear": -1}
    align_map = {"Strong": 1, "Weak": 0, "Broken": -1}
    momentum_map = {"Strengthening": 1, "Neutral": 0, "Weakening": -1}
    score = normalize_score(
        [
            regime_map.get(monthly_regime, 0),
            align_map.get(weekly_alignment, 0),
            1 if rs_rising else -1,
            momentum_map.get(weekly_momentum_state, 0),
        ]
    )
    return round(score, 4)


def _decision_from_score(score: float, buy_threshold: float, sell_threshold: float) -> str:
    if pd.isna(score):
        return "Hold"
    if score >= buy_threshold:
        return "Buy"
    if score <= sell_threshold:
        return "Sell"
    return "Hold"


def decision_from_health_score(health_score: float, buy_threshold: float, sell_threshold: float) -> str:
    return _decision_from_score(health_score, buy_threshold, sell_threshold)


def decision_from_production_score(production_score: float, buy_threshold: float, sell_threshold: float) -> str:
    return _decision_from_score(production_score, buy_threshold, sell_threshold)


def swing_lifecycle_frame(
    stock_daily: pd.DataFrame,
    stock_weekly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "Qualified",
            "Rule_Alignment",
            "Rule_Slope",
            "Rule_RS",
            "Production Score",
            "SetupType",
            "RSI14_State",
            "RSI_Accel",
            "MACD_Hist_Sign",
            "MACD_Hist_Accel",
            "Price_vs_EMA20",
            "Volume_Confirm",
            "Volatility_Expansion",
        ]
    )
    if stock_daily.empty or stock_weekly.empty or bench_weekly.empty:
        return empty
    if "Close" not in stock_weekly.columns or "Close" not in bench_weekly.columns:
        return empty

    wf = weekly_filter_frame(stock_weekly["Close"], bench_weekly["Close"])
    if wf.empty:
        return empty

    out = pd.DataFrame(index=wf.index)
    out["Qualified"] = wf["qualified"].fillna(False).astype(bool)
    out["Rule_Alignment"] = wf["rule_alignment"].fillna(False).astype(bool)
    out["Rule_Slope"] = wf["rule_slope"].fillna(False).astype(bool)
    out["Rule_RS"] = wf["rule_rs"].fillna(False).astype(bool)
    out["Production Score"] = np.nan
    out["SetupType"] = ""
    out["RSI14_State"] = 0
    out["RSI_Accel"] = 0
    out["MACD_Hist_Sign"] = 0
    out["MACD_Hist_Accel"] = 0
    out["Price_vs_EMA20"] = 0
    out["Volume_Confirm"] = 0
    out["Volatility_Expansion"] = 0

    for dt in out.index:
        daily_slice = stock_daily.loc[stock_daily.index <= dt]
        eval_daily = daily_components(daily_slice)
        if eval_daily.get("status") != "OK":
            continue

        out.at[dt, "SetupType"] = str(eval_daily["setup_type"])
        for comp, comp_value in eval_daily["components"].items():
            out.at[dt, comp] = int(comp_value)

        if bool(out.at[dt, "Qualified"]):
            out.at[dt, "Production Score"] = float(eval_daily["score"])

    return out


def swing_technical_snapshot(
    stock_daily: pd.DataFrame,
    stock_weekly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    empty = pd.DataFrame(columns=["Indicator", "Value"])

    if stock_daily.empty or stock_weekly.empty or bench_weekly.empty:
        return empty
    if "Close" not in stock_weekly.columns or "Close" not in bench_weekly.columns:
        return empty

    daily_eval = daily_components(stock_daily)
    wf = weekly_filter_frame(stock_weekly["Close"], bench_weekly["Close"])
    if wf.empty:
        return empty

    latest_w = wf.iloc[-1]
    rs_t3 = wf["RS_EMA20"].iloc[-4] if len(wf) >= 4 else np.nan
    ema20_t3 = wf["EMA20"].iloc[-4] if len(wf) >= 4 else np.nan

    rows.extend(
        [
            {"Indicator": "1W EMA20", "Value": float(latest_w["EMA20"])},
            {"Indicator": "1W EMA50", "Value": float(latest_w["EMA50"])},
            {"Indicator": "1W EMA200", "Value": float(latest_w["EMA200"])},
            {"Indicator": "1W EMA20[t-3]", "Value": float(ema20_t3) if pd.notna(ema20_t3) else np.nan},
            {"Indicator": "1W RS_EMA20", "Value": float(latest_w["RS_EMA20"]) if pd.notna(latest_w["RS_EMA20"]) else np.nan},
            {"Indicator": "1W RS_EMA20[t-3]", "Value": float(rs_t3) if pd.notna(rs_t3) else np.nan},
            {"Indicator": "Rule Alignment", "Value": bool(latest_w["rule_alignment"])},
            {"Indicator": "Rule EMA20 Slope", "Value": bool(latest_w["rule_slope"])},
            {"Indicator": "Rule RS Rising", "Value": bool(latest_w["rule_rs"])},
            {"Indicator": "Weekly Qualified", "Value": bool(latest_w["qualified"])},
        ]
    )

    if daily_eval.get("status") != "OK":
        rows.append({"Indicator": "Daily Status", "Value": str(daily_eval.get("status", "N/A"))})
        return pd.DataFrame(rows)

    vals = daily_eval["values"]
    comps = daily_eval["components"]
    rows.extend(
        [
            {"Indicator": "1D Close", "Value": float(vals["Close"])},
            {"Indicator": "1D EMA20", "Value": float(vals["EMA20_D"])},
            {"Indicator": "1D RSI14", "Value": float(vals["RSI14"])},
            {"Indicator": "1D MACD Histogram", "Value": float(vals["MACD_Hist"])},
            {"Indicator": "1D Volume", "Value": float(vals["Volume"]) if pd.notna(vals["Volume"]) else np.nan},
            {"Indicator": "1D Volume SMA20", "Value": float(vals["Volume_SMA20"]) if pd.notna(vals["Volume_SMA20"]) else np.nan},
            {"Indicator": "1D ATR14%", "Value": float(vals["ATR14_Pct"]) if pd.notna(vals["ATR14_Pct"]) else np.nan},
            {"Indicator": "1D ATR14% SMA20", "Value": float(vals["ATR14_Pct_SMA20"]) if pd.notna(vals["ATR14_Pct_SMA20"]) else np.nan},
            {"Indicator": "1D Prev 20D High", "Value": float(vals["Close_20D_High_Prev"]) if pd.notna(vals["Close_20D_High_Prev"]) else np.nan},
            {"Indicator": "SetupType", "Value": str(daily_eval["setup_type"])},
            {"Indicator": "Production Score", "Value": float(daily_eval["score"])},
            {"Indicator": "Comp RSI14 State", "Value": int(comps["RSI14_State"])},
            {"Indicator": "Comp RSI Accel", "Value": int(comps["RSI_Accel"])},
            {"Indicator": "Comp MACD Hist Sign", "Value": int(comps["MACD_Hist_Sign"])},
            {"Indicator": "Comp MACD Hist Accel", "Value": int(comps["MACD_Hist_Accel"])},
            {"Indicator": "Comp Price vs EMA20", "Value": int(comps["Price_vs_EMA20"])},
            {"Indicator": "Comp Volume Confirm", "Value": int(comps["Volume_Confirm"])},
            {"Indicator": "Comp Volatility Expansion", "Value": int(comps["Volatility_Expansion"])},
        ]
    )

    return pd.DataFrame(rows)


def portfolio_lifecycle_frame(
    stock_weekly: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "1M Regime",
            "1W Alignment",
            "1W RS",
            "1W Momentum State",
            "Risk Flag",
            "Health Score",
        ]
    )
    if stock_weekly.empty or stock_monthly.empty or bench_weekly.empty:
        return empty
    if "Close" not in stock_weekly.columns or "Close" not in stock_monthly.columns or "Close" not in bench_weekly.columns:
        return empty

    weekly = pd.concat(
        [
            stock_weekly["Close"].rename("stock"),
            bench_weekly["Close"].rename("bench"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if weekly.empty:
        return empty

    month_close = stock_monthly["Close"].dropna()
    if month_close.empty:
        return empty
    ema20_m = ema(month_close, 20)

    month_regime_vals = []
    for idx in range(len(month_close)):
        ema20_t_3 = float(ema20_m.iloc[idx - 3]) if idx >= 3 else np.nan
        month_regime_vals.append(
            classify_monthly_regime(
                float(month_close.iloc[idx]),
                float(ema20_m.iloc[idx]),
                ema20_t_3,
            )
        )
    month_regime = pd.Series(month_regime_vals, index=month_close.index, dtype="object")

    close_week = weekly["stock"]
    ema20_w = ema(close_week, 20)
    ema50_w = ema(close_week, 50)
    ema200_w = ema(close_week, 200)

    rs_week = relative_strength(close_week, weekly["bench"])
    rs_ema20_w = ema(rs_week, 20)
    rs_rising = rs_ema20_w > rs_ema20_w.shift(3)

    rsi_week = rsi(close_week, 14)
    rsi_prev = rsi_week.shift(3)

    out = pd.DataFrame(index=weekly.index)
    out["1M Regime"] = month_regime.reindex(out.index, method="ffill").fillna("Neutral")
    out["1W Alignment"] = [
        classify_weekly_alignment(float(e20), float(e50), float(e200))
        for e20, e50, e200 in zip(ema20_w, ema50_w, ema200_w)
    ]
    out["1W RS"] = np.where(rs_rising.fillna(False), "Rising", "Falling")
    out["1W Momentum State"] = [
        classify_weekly_momentum(float(cur), float(prev) if pd.notna(prev) else np.nan)
        for cur, prev in zip(rsi_week, rsi_prev)
    ]

    health_scores = []
    risk_flags = []
    for monthly_regime, weekly_alignment, rs_val, weekly_momentum in zip(
        out["1M Regime"],
        out["1W Alignment"],
        rs_rising.fillna(False),
        out["1W Momentum State"],
    ):
        rs_bool = bool(rs_val)
        health_scores.append(_health_score(monthly_regime, weekly_alignment, rs_bool, weekly_momentum))
        risk_flags.append(classify_risk_flag(monthly_regime, weekly_alignment, rs_bool))

    out["Risk Flag"] = risk_flags
    out["Health Score"] = health_scores
    out = out.dropna(subset=["Health Score"])
    return out


def evaluate_portfolio_stock(
    meta: pd.Series,
    stock_daily: pd.DataFrame,
    stock_weekly: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
    stock_status: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Name": meta["Name"],
        "Region": meta["Region"],
        "SignalTicker": meta["SignalTicker"],
        "TradeTicker_DE": meta.get("TradeTicker_DE", ""),
        "Benchmark": meta["Benchmark"],
        "Price": np.nan,
        "1M Regime": "Neutral",
        "1W Alignment": "Broken",
        "1W RS": "Falling",
        "1W Momentum State": "Neutral",
        "Risk Flag": "Watch",
        "Health Score": np.nan,
        "Status": stock_status,
    }

    if stock_status != "OK":
        return result

    close_daily = stock_daily["Close"]
    close_week = stock_weekly["Close"]
    close_month = stock_monthly["Close"]

    ema20_m = ema(close_month, 20)
    monthly_regime = classify_monthly_regime(
        float(close_month.iloc[-1]),
        float(ema20_m.iloc[-1]),
        float(ema20_m.iloc[-4]) if len(ema20_m) >= 4 else np.nan,
    )

    ema20_w = ema(close_week, 20)
    ema50_w = ema(close_week, 50)
    ema200_w = ema(close_week, 200)
    weekly_alignment = classify_weekly_alignment(float(ema20_w.iloc[-1]), float(ema50_w.iloc[-1]), float(ema200_w.iloc[-1]))

    rs_week = relative_strength(close_week, bench_weekly["Close"])
    rs_ema20_w = ema(rs_week, 20)
    rs_rising = False
    if len(rs_ema20_w.dropna()) >= 4:
        rs_rising = bool(rs_ema20_w.iloc[-1] > rs_ema20_w.iloc[-4])

    rsi_week = rsi(close_week, 14)
    weekly_momentum_state = classify_weekly_momentum(
        float(rsi_week.iloc[-1]),
        float(rsi_week.iloc[-4]) if len(rsi_week) >= 4 else np.nan,
    )

    risk_flag = classify_risk_flag(monthly_regime, weekly_alignment, rs_rising)
    health_score = _health_score(monthly_regime, weekly_alignment, rs_rising, weekly_momentum_state)

    result.update(
        {
            "Price": round(float(close_daily.iloc[-1]), 4),
            "1M Regime": monthly_regime,
            "1W Alignment": weekly_alignment,
            "1W RS": "Rising" if rs_rising else "Falling",
            "1W Momentum State": weekly_momentum_state,
            "Risk Flag": risk_flag,
            "Health Score": health_score,
            "Status": "OK",
        }
    )

    return result


def rank_qualified(qualified_df: pd.DataFrame) -> pd.DataFrame:
    if qualified_df.empty:
        return qualified_df
    out = qualified_df.copy()
    out = out.sort_values("ProductionScore", ascending=False).reset_index(drop=True)
    out["ProductionRank"] = np.arange(1, len(out) + 1)
    return out


def apply_custom_weights(qualified_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if qualified_df.empty:
        return qualified_df

    comp_cols = [
        "RSI14_State",
        "RSI_Accel",
        "MACD_Hist_Sign",
        "MACD_Hist_Accel",
        "Price_vs_EMA20",
        "Volume_Confirm",
        "Volatility_Expansion",
    ]
    wsum = sum(float(weights.get(c, 0.0)) for c in comp_cols)
    out = qualified_df.copy()

    if wsum <= 0:
        out["CustomScore"] = 0.0
    else:
        weighted = sum(out[c] * float(weights.get(c, 0.0)) for c in comp_cols)
        out["CustomScore"] = (weighted / wsum).clip(-1.0, 1.0)

    out = out.sort_values("CustomScore", ascending=False).reset_index(drop=True)
    out["CustomRank"] = np.arange(1, len(out) + 1)

    prod_ranks = out.set_index("SignalTicker")["ProductionRank"].to_dict()
    out["RankDelta"] = out.apply(lambda r: int(prod_ranks.get(r["SignalTicker"], r["CustomRank"]) - r["CustomRank"]), axis=1)
    return out


def sort_portfolio_for_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_risk_order"] = out["Risk Flag"].map(config.RISK_SORT_ORDER).fillna(99)
    out = out.sort_values(["_risk_order", "Health Score"], ascending=[True, True]).drop(columns=["_risk_order"])
    out = out.reset_index(drop=True)
    return out
