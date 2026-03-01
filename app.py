from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data import fetch_ticker_data, load_universe_csv
from src.decision_trace import DecisionTrace
from src.ratings import screener_snapshot, technical_ratings
from src.signals import (
    apply_custom_weights,
    build_swing_decision_trace,
    daily_components,
    decision_from_health_score,
    decision_from_production_score,
    evaluate_portfolio_stock,
    evaluate_swing_stock,
    portfolio_lifecycle_frame,
    rank_qualified,
    sort_portfolio_for_risk,
    swing_lifecycle_frame,
    swing_technical_snapshot,
)
from src.ui_helpers import (
    clean_display_df,
    lifecycle_score_chart,
    prepare_lifecycle_frame,
)

st.set_page_config(page_title="Snapshot TA Screener", layout="wide")
st.title("Snapshot-Only Technical-Analysis Stock Screener")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("stock_screener.app")
DEBUG_MODE = st.sidebar.checkbox("Debug mode", value=False, key="debug_mode")

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"

PORTFOLIO_EXAMPLE = EXAMPLES_DIR / "portfolio.csv"
WATCHLIST_EXAMPLE = EXAMPLES_DIR / "xfra_swing_trading_universe.csv"

DATAFRAME_HAS_SELECTION = "on_select" in inspect.signature(st.dataframe).parameters


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    st.experimental_rerun()


THRESHOLD_PRESETS: dict[str, tuple[float, float]] = {
    "Conservative": (-0.30, 0.45),
    "Balanced": (-0.20, 0.30),
    "Aggressive": (-0.10, 0.15),
}

POSITIVE_WORDS = {
    "buy",
    "strong buy",
    "strong",
    "rising",
    "strengthening",
    "ok",
    "yes",
}
NEGATIVE_WORDS = {
    "sell",
    "strong sell",
    "broken",
    "falling",
    "weakening",
    "breakdown",
    "no",
}
NEUTRAL_WORDS = {"hold", "watch", "weak", "neutral", "no signal", "n/a"}


def _build_data_cache(universe_df: pd.DataFrame) -> dict[str, object]:
    all_tickers = set(universe_df["SignalTicker"].dropna().astype(str)) | set(
        universe_df["Benchmark"].dropna().astype(str)
    )
    cache: dict[str, object] = {}
    progress = st.progress(0.0, text="Downloading market data...")
    total = max(1, len(all_tickers))
    for i, ticker in enumerate(sorted(all_tickers), start=1):
        cache[ticker] = _cached_fetch_ticker_data(ticker)
        progress.progress(i / total, text=f"Downloaded {i}/{total}: {ticker}")
    progress.empty()
    return cache


@st.cache_data(ttl=config.DOWNLOAD_TTL_SECONDS, show_spinner=False)
def _cached_fetch_ticker_data(ticker: str):
    return fetch_ticker_data(ticker)


def pick_universe_df(
    title: str, example_path: Path, key_prefix: str
) -> tuple[pd.DataFrame | None, str | None]:
    st.subheader(title)
    source = st.radio(
        "Source",
        ["Use example CSV", "Upload CSV"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    try:
        if source == "Use example CSV":
            df = load_universe_csv(example_path)
            st.caption(f"Using example file: `{example_path}`")
            return df, None

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key=f"{key_prefix}_upload",
            help="CSV schema: Name, Region, SignalTicker (+ optional TradeTicker_DE, Benchmark)",
        )
        if uploaded is None:
            return None, "Upload a CSV to continue."

        df = load_universe_csv(uploaded)
        return df, None
    except Exception as exc:  # pragma: no cover - UI display branch
        return None, str(exc)


def _tone_for_text(value: object) -> str:
    if isinstance(value, bool):
        return "positive" if value else "negative"

    text = str(value).strip().lower()
    if text in POSITIVE_WORDS:
        return "positive"
    if text in NEGATIVE_WORDS:
        return "negative"
    if text in NEUTRAL_WORDS:
        return "neutral"
    return "neutral"


def _badge_text(value: object) -> str:
    if pd.isna(value):
        return "⚪ n/a"
    tone = _tone_for_text(value)
    marker = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(tone, "⚪")
    return f"{marker} {value}"


def _apply_badges(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(_badge_text)
    return out


def _render_threshold_presets(state_key: str, default_pair: tuple[float, float]) -> None:
    cols = st.columns(3)
    for col, label in zip(cols, THRESHOLD_PRESETS.keys()):
        with col:
            if st.button(label, key=f"{state_key}_preset_{label.lower()}"):
                st.session_state[state_key] = THRESHOLD_PRESETS[label]
                _trigger_rerun()
    if state_key not in st.session_state:
        st.session_state[state_key] = default_pair


def _date_debug_info(df: pd.DataFrame, label: str) -> dict[str, object]:
    if df.empty:
        return {"label": label, "empty": True}
    idx = pd.to_datetime(df.index, errors="coerce")
    return {
        "label": label,
        "dtype": str(idx.dtype),
        "min": str(idx.min()),
        "max": str(idx.max()),
        "rows": int(len(df)),
        "is_monotonic": bool(idx.is_monotonic_increasing),
    }


def _render_debug_panel(
    trace: DecisionTrace,
    stock_daily: pd.DataFrame,
    stock_weekly: pd.DataFrame,
    bench_weekly: pd.DataFrame,
) -> None:
    if not DEBUG_MODE:
        return
    st.sidebar.markdown("### Debug: Selected Trace")
    st.sidebar.json(trace.to_dict())
    st.sidebar.markdown("### Debug: Date Diagnostics")
    st.sidebar.json(
        {
            "stock_daily": _date_debug_info(stock_daily, "stock_daily"),
            "stock_weekly": _date_debug_info(stock_weekly, "stock_weekly"),
            "benchmark_weekly": _date_debug_info(bench_weekly, "benchmark_weekly"),
        }
    )


def _coerce_threshold_pair(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple | list) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
    return default


def _set_selected_stock(row: pd.Series, source: str) -> None:
    region = str(row.get("Region", "")).upper().strip()
    benchmark = str(row.get("Benchmark", "") or "").strip()
    if not benchmark:
        benchmark = config.REGION_TO_BENCHMARK.get(region, config.US_BENCHMARK)

    st.session_state["selected_stock"] = {
        "Source": source,
        "Name": str(row.get("Name", "")).strip(),
        "Region": region,
        "SignalTicker": str(row.get("SignalTicker", "")).strip(),
        "TradeTicker_DE": str(row.get("TradeTicker_DE", "")).strip(),
        "Benchmark": benchmark,
        "Status": str(row.get("Status", "")).strip(),
    }
    logger.info(
        "Selected stock updated: source=%s ticker=%s",
        source,
        st.session_state["selected_stock"]["SignalTicker"],
    )


def _store_ranked_context(source_df: pd.DataFrame, source_label: str) -> None:
    if source_df.empty or "SignalTicker" not in source_df.columns:
        return

    ranked_tickers = [str(t) for t in source_df["SignalTicker"].astype(str).tolist()]
    ranked_meta: dict[str, dict[str, str]] = {}
    for _, r in source_df.iterrows():
        ticker = str(r.get("SignalTicker", "")).strip()
        if not ticker:
            continue
        ranked_meta[ticker] = {
            "Name": str(r.get("Name", "")).strip(),
            "Region": str(r.get("Region", "")).upper().strip(),
            "TradeTicker_DE": str(r.get("TradeTicker_DE", "")).strip(),
            "Benchmark": str(r.get("Benchmark", "")).strip(),
            "Status": str(r.get("Status", "")).strip(),
        }

    st.session_state["selected_ranked_tickers"] = ranked_tickers
    st.session_state["selected_ranked_meta"] = ranked_meta
    st.session_state["selected_ranked_source"] = source_label


def _render_selectable_stock_table(
    source_df: pd.DataFrame,
    display_df: pd.DataFrame,
    table_key: str,
    source_label: str,
    fallback_label: str,
) -> None:
    if source_df.empty:
        st.info("No rows to display.")
        return

    source_df = source_df.reset_index(drop=True)
    display_df = display_df.reset_index(drop=True)

    if DATAFRAME_HAS_SELECTION:
        event = st.dataframe(
            clean_display_df(display_df),
            use_container_width=True,
            hide_index=True,
            key=table_key,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = []
        if hasattr(event, "selection") and hasattr(event.selection, "rows"):
            selected_rows = event.selection.rows

        selected_row = selected_rows[0] if selected_rows else None
        last_row_key = f"{table_key}_last_selected_row"
        previous_row = st.session_state.get(last_row_key)

        # Only update global selected stock when this table selection actually changed.
        if selected_row != previous_row:
            st.session_state[last_row_key] = selected_row
            if isinstance(selected_row, int) and 0 <= selected_row < len(source_df):
                _store_ranked_context(source_df, source_label)
                _set_selected_stock(source_df.iloc[selected_row], source_label)

        st.caption("Click a row to open it in the Stock Details tab.")
        return

    st.dataframe(clean_display_df(display_df), use_container_width=True, hide_index=True)
    options = source_df["SignalTicker"].astype(str).tolist()
    names = source_df.set_index("SignalTicker")["Name"].astype(str).to_dict()
    selected = st.selectbox(
        fallback_label,
        options=options,
        format_func=lambda t: f"{t} - {names.get(t, '')}".strip(" -"),
        key=f"{table_key}_fallback",
    )
    fallback_last_key = f"{table_key}_fallback_last_selected"
    previous_selected = st.session_state.get(fallback_last_key)
    if selected and selected != previous_selected:
        st.session_state[fallback_last_key] = selected
        picked = source_df[source_df["SignalTicker"] == selected].iloc[0]
        _store_ranked_context(source_df, source_label)
        _set_selected_stock(picked, source_label)


def render_portfolio_tab() -> None:
    df, err = pick_universe_df("Long-Term Portfolio Tracker", PORTFOLIO_EXAMPLE, "portfolio")
    if err:
        st.info(err)
        return
    if df is None or df.empty:
        st.warning("No rows found in portfolio CSV.")
        return

    data_cache = _build_data_cache(df)
    rows = []
    for _, meta in df.iterrows():
        stock = data_cache.get(meta["SignalTicker"])
        bench = data_cache.get(meta["Benchmark"])
        stock_status = getattr(stock, "status", "Ticker data missing")
        if bench is None:
            stock_status = "Benchmark data missing"
        elif getattr(bench, "status", "") != "OK":
            stock_status = f"Benchmark issue: {bench.status}"

        row = evaluate_portfolio_stock(
            meta=meta,
            stock_daily=getattr(stock, "daily", pd.DataFrame()),
            stock_weekly=getattr(stock, "weekly", pd.DataFrame()),
            stock_monthly=getattr(stock, "monthly", pd.DataFrame()),
            bench_weekly=getattr(bench, "weekly", pd.DataFrame()),
            stock_status=stock_status,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out = sort_portfolio_for_risk(out)

    st.subheader("Decision Thresholds")
    _render_threshold_presets("portfolio_thresholds", (-0.25, 0.35))
    sell_threshold, buy_threshold = st.slider(
        "Health Score thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=_coerce_threshold_pair(st.session_state.get("portfolio_thresholds"), (-0.25, 0.35)),
        step=0.05,
        key="portfolio_thresholds",
        help="Decision rule: score <= sell threshold => Sell, score >= buy threshold => Buy, otherwise Hold.",
    )
    out["Decision"] = out.apply(
        lambda r: (
            decision_from_health_score(float(r["Health Score"]), buy_threshold, sell_threshold)
            if r["Status"] == "OK"
            else "No Data"
        ),
        axis=1,
    )

    display_cols = [
        "Name",
        "Region",
        "SignalTicker",
        "TradeTicker_DE",
        "Price",
        "1M Regime",
        "1W Alignment",
        "1W RS",
        "1W Momentum State",
        "Risk Flag",
        "Health Score",
        "Decision",
        "Status",
    ]
    portfolio_display = _apply_badges(
        out[display_cols],
        columns=["Decision", "1W Alignment", "1W RS", "1W Momentum State", "Risk Flag"],
    )
    _render_selectable_stock_table(
        source_df=out,
        display_df=portfolio_display,
        table_key="portfolio_main_table",
        source_label="Portfolio",
        fallback_label="Select portfolio stock for Stock Details",
    )

    st.subheader("Portfolio Lifecycle")
    available = out[out["Status"] == "OK"].copy()
    if available.empty:
        st.info("Lifecycle chart appears when at least one portfolio stock has valid data.")
        return

    selected_from_state = st.session_state.get("selected_stock", {})
    default_ticker = ""
    if (
        isinstance(selected_from_state, dict)
        and selected_from_state.get("Source") == "Portfolio"
        and selected_from_state.get("SignalTicker") in set(available["SignalTicker"])
    ):
        default_ticker = str(selected_from_state["SignalTicker"])
    if not default_ticker:
        default_ticker = str(available.iloc[0]["SignalTicker"])

    options = available["SignalTicker"].astype(str).tolist()
    names = available.set_index("SignalTicker")["Name"].astype(str).to_dict()
    selected_ticker = st.selectbox(
        "Select portfolio stock for lifecycle",
        options=options,
        index=max(0, options.index(default_ticker)) if default_ticker in options else 0,
        format_func=lambda t: f"{t} - {names.get(t, '')}".strip(" -"),
        key="portfolio_lifecycle_ticker",
    )
    row = available[available["SignalTicker"] == selected_ticker].iloc[0]
    _set_selected_stock(row, "Portfolio")

    stock = data_cache.get(selected_ticker)
    benchmark_ticker = str(row.get("Benchmark", ""))
    bench = data_cache.get(benchmark_ticker)
    if getattr(stock, "status", "") != "OK" or getattr(bench, "status", "") != "OK":
        st.info("Lifecycle data is unavailable for this selection.")
        return

    lifecycle = portfolio_lifecycle_frame(
        stock_weekly=getattr(stock, "weekly", pd.DataFrame()),
        stock_monthly=getattr(stock, "monthly", pd.DataFrame()),
        bench_weekly=getattr(bench, "weekly", pd.DataFrame()),
    )
    if lifecycle.empty:
        st.info("Lifecycle data is unavailable for this selection.")
        return

    lifecycle = lifecycle.copy()
    lifecycle["Decision"] = lifecycle["Health Score"].apply(
        lambda v: decision_from_health_score(float(v), buy_threshold, sell_threshold)
    )
    p_chart_df = prepare_lifecycle_frame(
        lifecycle=lifecycle,
        score_col="Health Score",
        decision_col="Decision",
        window=104,
    )
    p_chart = lifecycle_score_chart(
        lifecycle_df=p_chart_df.set_index("Date"),
        score_col="Health Score",
        decision_col="Decision",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    if p_chart is None:
        fallback = p_chart_df[["Date", "Health Score"]].set_index("Date")
        fallback["Buy Threshold"] = buy_threshold
        fallback["Sell Threshold"] = sell_threshold
        st.line_chart(fallback, use_container_width=True)
    else:
        st.altair_chart(p_chart, use_container_width=True)

    latest = lifecycle.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Health Score", f"{latest['Health Score']:.4f}")
    c2.metric("Decision", str(latest["Decision"]))
    c3.metric("Risk Flag", str(latest["Risk Flag"]))
    c4.metric("1W Alignment", str(latest["1W Alignment"]))

    p_recent = lifecycle.tail(26).copy()
    p_recent.insert(0, "Week", p_recent.index)
    p_recent = p_recent.reset_index(drop=True)
    p_recent_display = _apply_badges(
        p_recent[
            [
                "Week",
                "Health Score",
                "Decision",
                "Risk Flag",
                "1M Regime",
                "1W Alignment",
                "1W RS",
                "1W Momentum State",
            ]
        ],
        columns=["Decision", "Risk Flag", "1W Alignment", "1W RS", "1W Momentum State"],
    )
    st.dataframe(
        clean_display_df(p_recent_display),
        use_container_width=True,
        hide_index=True,
    )


def render_swing_tab() -> None:
    df, err = pick_universe_df("Momentum Swing Screener", WATCHLIST_EXAMPLE, "swing")
    if err:
        st.info(err)
        return
    if df is None or df.empty:
        st.warning("No rows found in watchlist CSV.")
        return

    data_cache = _build_data_cache(df)
    rows = []
    for _, meta in df.iterrows():
        stock = data_cache.get(meta["SignalTicker"])
        bench = data_cache.get(meta["Benchmark"])
        stock_status = getattr(stock, "status", "Ticker data missing")
        if bench is None:
            stock_status = "Benchmark data missing"
        elif getattr(bench, "status", "") != "OK":
            stock_status = f"Benchmark issue: {bench.status}"

        row = evaluate_swing_stock(
            meta=meta,
            stock_daily=getattr(stock, "daily", pd.DataFrame()),
            stock_weekly=getattr(stock, "weekly", pd.DataFrame()),
            bench_weekly=getattr(bench, "weekly", pd.DataFrame()),
            stock_status=stock_status,
        )
        rows.append(row)

    swing_df = pd.DataFrame(rows)
    qualified = swing_df[swing_df["Qualified"] & (swing_df["Status"] == "OK")].copy()
    qualified = rank_qualified(qualified)
    view_mode = st.radio(
        "View",
        ["Action Board", "Screener Table"],
        horizontal=True,
        key="swing_view_mode",
    )

    st.subheader("1) Swing Screener")
    _render_threshold_presets("swing_thresholds", (-0.20, 0.30))
    sell_threshold, buy_threshold = st.slider(
        "Production Score thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=_coerce_threshold_pair(st.session_state.get("swing_thresholds"), (-0.20, 0.30)),
        step=0.05,
        key="swing_thresholds",
        help=(
            "Production Score is the equal-weight average of 7 daily components after weekly qualification. "
            "Higher means stronger bullish confirmation.\n\n"
            "Decision rule:\n"
            "- score <= Sell threshold -> Sell\n"
            "- score >= Buy threshold -> Buy\n"
            "- otherwise -> Hold\n\n"
            "Practical presets:\n"
            "- Conservative: Sell -0.30, Buy 0.45\n"
            "- Balanced: Sell -0.20, Buy 0.30\n"
            "- Aggressive: Sell -0.10, Buy 0.15"
        ),
    )

    if view_mode == "Action Board":
        if qualified.empty:
            st.warning("No stocks currently pass the weekly hard filter.")
        else:
            qualified["Decision"] = qualified["ProductionScore"].apply(
                lambda s: decision_from_production_score(float(s), buy_threshold, sell_threshold)
            )
            if "selected_stock" not in st.session_state:
                _store_ranked_context(qualified, "Swing")
                _set_selected_stock(qualified.iloc[0], "Swing")
            summary = qualified["Decision"].value_counts().to_dict()
            c1, c2, c3 = st.columns(3)
            c1.metric("Buy", int(summary.get("Buy", 0)))
            c2.metric("Hold", int(summary.get("Hold", 0)))
            c3.metric("Sell", int(summary.get("Sell", 0)))

            action_cols = [
                "ProductionRank",
                "Decision",
                "Name",
                "SignalTicker",
                "TradeTicker_DE",
                "Price",
                "SetupType",
                "ProductionScore",
                "Status",
            ]
            action_display = _apply_badges(qualified[action_cols], columns=["Decision"])
            _render_selectable_stock_table(
                source_df=qualified,
                display_df=action_display,
                table_key="swing_action_table",
                source_label="Swing",
                fallback_label="Select swing stock for Stock Details",
            )

            with st.expander("Show component details", expanded=False):
                q_cols = [
                    "ProductionRank",
                    "Name",
                    "Region",
                    "SignalTicker",
                    "TradeTicker_DE",
                    "Price",
                    "SetupType",
                    "ProductionScore",
                    "Decision",
                    "RSI14_State",
                    "RSI_Accel",
                    "MACD_Hist_Sign",
                    "MACD_Hist_Accel",
                    "Price_vs_EMA20",
                    "Volume_Confirm",
                    "Volatility_Expansion",
                    "Status",
                ]
                st.dataframe(
                    clean_display_df(qualified[q_cols]), use_container_width=True, hide_index=True
                )
    else:
        screener_rows = []
        swing_lookup = {str(r["SignalTicker"]): r for _, r in swing_df.iterrows()}
        for _, meta in df.iterrows():
            ticker = str(meta["SignalTicker"])
            stock = data_cache.get(ticker)
            stock_status = getattr(stock, "status", "Ticker data missing")
            base = swing_lookup.get(ticker, {})

            row = {
                "Symbol": meta.get("TradeTicker_DE", "") or ticker,
                "Name": meta["Name"],
                "Region": meta["Region"],
                "SignalTicker": ticker,
                "TradeTicker_DE": meta.get("TradeTicker_DE", ""),
                "Benchmark": meta["Benchmark"],
                "Qualified (Weekly)": bool(base.get("Qualified", False)),
                "SetupType": base.get("SetupType", ""),
                "Production Score": base.get("ProductionScore", np.nan),
                "Price": base.get("Price", np.nan),
                "Status": base.get("Status", stock_status),
            }

            if stock_status == "OK":
                snap = screener_snapshot(getattr(stock, "daily", pd.DataFrame()))
                row.update(snap)

                # Show Production Score in screener even when weekly filter is not currently qualified.
                if pd.isna(row["Production Score"]):
                    daily_eval = daily_components(getattr(stock, "daily", pd.DataFrame()))
                    if daily_eval.get("status") == "OK":
                        row["Production Score"] = float(daily_eval["score"])
                        row["SetupType"] = str(daily_eval["setup_type"])
            else:
                row.update(
                    {
                        "Summary Rating": "Neutral",
                        "MA Rating": "Neutral",
                        "Osc Rating": "Neutral",
                        "Summary Score": -99.0,
                        "MA Score": -99.0,
                        "Osc Score": -99.0,
                        "RSI(14)": np.nan,
                        "Momentum(10)": np.nan,
                        "AO": np.nan,
                        "CCI(20)": np.nan,
                        "Stoch %K": np.nan,
                        "Stoch %D": np.nan,
                    }
                )

            screener_rows.append(row)

        screener_df = pd.DataFrame(screener_rows)
        if screener_df.empty:
            st.info("No screener rows available.")
        else:
            screener_df = screener_df.sort_values(
                ["Summary Score", "Production Score"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)
            if "Production Score" in screener_df.columns:
                screener_df["Decision"] = screener_df["Production Score"].apply(
                    lambda s: (
                        decision_from_production_score(float(s), buy_threshold, sell_threshold)
                        if pd.notna(s)
                        else "Hold"
                    )
                )
            if "selected_stock" not in st.session_state:
                _store_ranked_context(screener_df, "Swing")
                _set_selected_stock(screener_df.iloc[0], "Swing")
            screener_cols = [
                "Symbol",
                "Name",
                "Summary Rating",
                "MA Rating",
                "Osc Rating",
                "RSI(14)",
                "Momentum(10)",
                "AO",
                "CCI(20)",
                "Stoch %K",
                "Stoch %D",
                "SetupType",
                "Qualified (Weekly)",
                "Decision",
                "Production Score",
                "Price",
                "SignalTicker",
                "Status",
            ]
            screener_display = screener_df[screener_cols].rename(
                columns={
                    "Summary Rating": "Tech Rating",
                    "SetupType": "Pattern",
                }
            )
            screener_display["Qualified (Weekly)"] = screener_display["Qualified (Weekly)"].apply(
                lambda v: "Yes" if bool(v) else "No"
            )
            screener_display = _apply_badges(
                screener_display,
                columns=[
                    "Tech Rating",
                    "MA Rating",
                    "Osc Rating",
                    "Decision",
                    "Qualified (Weekly)",
                ],
            )
            _render_selectable_stock_table(
                source_df=screener_df,
                display_df=screener_display,
                table_key="swing_screener_table",
                source_label="Swing",
                fallback_label="Select screener stock for Stock Details",
            )

    st.subheader("2) Recently Lost Alignment (Last 6 Weeks)")
    if "RecentlyLost" in swing_df.columns:
        lost = swing_df[swing_df["RecentlyLost"] == True].copy()  # noqa: E712
    else:
        lost = swing_df.iloc[0:0].copy()
    if lost.empty:
        st.info("No stocks recently lost weekly alignment in the last 6 weeks.")
    else:
        lost_cols = [
            "Name",
            "Region",
            "SignalTicker",
            "TradeTicker_DE",
            "LastQualifiedWeek",
            "FailedRules",
            "Status",
        ]
        st.dataframe(clean_display_df(lost[lost_cols]), use_container_width=True, hide_index=True)

    st.subheader("3) Weights Lab (Experimental)")
    st.caption("Custom weights do not affect production ranking.")
    if qualified.empty:
        st.info("Weights Lab is available when at least one stock is qualified.")
    else:
        with st.expander("Adjust component weights", expanded=False):
            cols = st.columns(4)
            weights = {}
            component_names = [
                "RSI14_State",
                "RSI_Accel",
                "MACD_Hist_Sign",
                "MACD_Hist_Accel",
                "Price_vs_EMA20",
                "Volume_Confirm",
                "Volatility_Expansion",
            ]
            for idx, comp in enumerate(component_names):
                with cols[idx % 4]:
                    weights[comp] = st.slider(comp, 0.0, 5.0, 1.0, 0.1, key=f"w_{comp}")

            custom = apply_custom_weights(qualified, weights)
            custom_cols = [
                "CustomRank",
                "ProductionRank",
                "RankDelta",
                "Name",
                "SignalTicker",
                "ProductionScore",
                "CustomScore",
                "SetupType",
            ]
            st.dataframe(
                clean_display_df(custom[custom_cols]), use_container_width=True, hide_index=True
            )

    with st.expander("Status / Excluded", expanded=False):
        status_cols = [
            "Name",
            "Region",
            "SignalTicker",
            "TradeTicker_DE",
            "Qualified",
            "Rule_Alignment",
            "Rule_Slope",
            "Rule_RS",
            "FailedRules",
            "Status",
        ]
        for col in status_cols:
            if col not in swing_df.columns:
                swing_df[col] = pd.NA
        st.dataframe(
            clean_display_df(swing_df[status_cols]), use_container_width=True, hide_index=True
        )


def render_stock_details_tab() -> None:
    st.subheader("Stock Details")
    selected = st.session_state.get("selected_stock")
    if not selected:
        st.info("Click a row in Portfolio or Swing Action Board to load stock details here.")
        return

    ranked_tickers = st.session_state.get("selected_ranked_tickers", [])
    ranked_meta = st.session_state.get("selected_ranked_meta", {})
    current_ticker = str(selected.get("SignalTicker", "")).strip()
    if (
        isinstance(ranked_tickers, list)
        and current_ticker in ranked_tickers
        and len(ranked_tickers) > 1
    ):
        idx = ranked_tickers.index(current_ticker)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            prev_clicked = st.button("Prev", disabled=idx <= 0, key="stock_details_prev")
        with nav_col2:
            st.caption(f"Ranked item {idx + 1} of {len(ranked_tickers)}")
        with nav_col3:
            next_clicked = st.button(
                "Next", disabled=idx >= len(ranked_tickers) - 1, key="stock_details_next"
            )

        if prev_clicked or next_clicked:
            new_idx = idx - 1 if prev_clicked else idx + 1
            new_ticker = str(ranked_tickers[new_idx])
            meta = ranked_meta.get(new_ticker, {})
            row = pd.Series(
                {
                    "Name": meta.get("Name", new_ticker),
                    "Region": meta.get("Region", selected.get("Region", "")),
                    "SignalTicker": new_ticker,
                    "TradeTicker_DE": meta.get("TradeTicker_DE", ""),
                    "Benchmark": meta.get("Benchmark", selected.get("Benchmark", "")),
                    "Status": meta.get("Status", ""),
                }
            )
            _set_selected_stock(
                row, str(st.session_state.get("selected_ranked_source", selected.get("Source", "")))
            )
            _trigger_rerun()

    signal_ticker = str(selected.get("SignalTicker", "")).strip()
    benchmark_ticker = str(selected.get("Benchmark", "")).strip()
    if not signal_ticker:
        st.info("Selected row is missing SignalTicker.")
        return

    if not benchmark_ticker:
        region = str(selected.get("Region", "")).upper().strip()
        benchmark_ticker = config.REGION_TO_BENCHMARK.get(region, config.US_BENCHMARK)

    stock = fetch_ticker_data(signal_ticker)
    bench = fetch_ticker_data(benchmark_ticker)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Source", str(selected.get("Source", "")))
    c2.metric("Name", str(selected.get("Name", signal_ticker)))
    c3.metric("Ticker", signal_ticker)
    c4.metric("Benchmark", benchmark_ticker)
    c5.metric("TradeTicker_DE", str(selected.get("TradeTicker_DE", "") or "-"))

    if stock.status != "OK":
        st.warning(f"Stock data unavailable: {stock.status}")
        return
    if bench.status != "OK":
        st.warning(f"Benchmark data unavailable: {bench.status}")
        return

    st.markdown("### Decision Card")
    s_default = _coerce_threshold_pair(st.session_state.get("swing_thresholds"), (-0.20, 0.30))
    _render_threshold_presets("stock_details_swing_thresholds", s_default)
    s_sell, s_buy = st.slider(
        "Swing production thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=_coerce_threshold_pair(
            st.session_state.get("stock_details_swing_thresholds"), s_default
        ),
        step=0.05,
        key="stock_details_swing_thresholds",
    )

    trace = build_swing_decision_trace(
        stock_daily=stock.daily,
        stock_weekly=stock.weekly,
        bench_weekly=bench.weekly,
        buy_threshold=s_buy,
        sell_threshold=s_sell,
        signal_ticker=signal_ticker,
        benchmark=benchmark_ticker,
        name=str(selected.get("Name", signal_ticker)),
    )
    if trace is None:
        st.info("Decision snapshot unavailable for this stock.")
        return
    _render_debug_panel(
        trace=trace, stock_daily=stock.daily, stock_weekly=stock.weekly, bench_weekly=bench.weekly
    )

    dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
    dc1.metric("Production Score", f"{float(trace.score):.4f}")
    dc2.metric("Decision", str(trace.decision))
    dc3.metric("Qualified (Weekly)", "Yes" if bool(trace.qualified) else "No")
    dc4.metric("SetupType", str(trace.setup_type))
    dc5.metric("Risk Flag", str(trace.risk_flag))
    dc6.metric("Thresholds", f"Sell {s_sell:.2f} / Buy {s_buy:.2f}")
    st.caption(f"Risk reason: {trace.risk_reason}")
    st.markdown(
        " ".join(
            [
                _badge_text(trace.decision),
                _badge_text("Yes" if trace.qualified else "No"),
                _badge_text(trace.setup_type),
                _badge_text(trace.risk_flag),
            ]
        )
    )

    st.markdown("### Technical Ratings")
    ratings = technical_ratings(stock.daily)
    if ratings.get("status") != "OK":
        st.info(f"Ratings unavailable: {ratings.get('status', 'n/a')}")
    else:
        r1, r2, r3 = st.columns(3)
        blocks = [
            ("Oscillators", ratings["oscillators"]),
            ("Summary", ratings["summary"]),
            ("Moving Averages", ratings["moving_averages"]),
        ]
        for col, (title, block) in zip((r1, r2, r3), blocks):
            with col:
                st.markdown(f"**{title}**")
                st.metric("Rating", str(block["label"]))
                c_buy, c_neutral, c_sell = st.columns(3)
                c_buy.metric("Buy", int(block["buy"]))
                c_neutral.metric("Neutral", int(block["neutral"]))
                c_sell.metric("Sell", int(block["sell"]))

    st.markdown("### Why This Decision")
    why_left, why_right = st.columns(2)
    with why_left:
        st.caption("Weekly hard filter checks")
        weekly_rules_df = pd.DataFrame(
            [{"Rule": r.name, "Pass": r.passed, "Value": r.value} for r in trace.rules]
        )
        weekly_rules_df["Pass"] = weekly_rules_df["Pass"].apply(
            lambda v: "Yes" if bool(v) else "No"
        )
        weekly_rules_df = _apply_badges(weekly_rules_df, columns=["Pass"])
        st.dataframe(clean_display_df(weekly_rules_df), use_container_width=True, hide_index=True)
    with why_right:
        st.caption("Daily component signals")
        components_df = pd.DataFrame(
            [{"Component": c.name, "Signal": c.signal, "Value": c.value} for c in trace.components]
        )
        components_df["Signal"] = components_df["Signal"].map({1: "Buy", 0: "Neutral", -1: "Sell"})
        components_df = _apply_badges(components_df, columns=["Signal"])
        st.dataframe(clean_display_df(components_df), use_container_width=True, hide_index=True)

    st.markdown("### Swing Lifecycle & Technicals")

    swing_lifecycle = swing_lifecycle_frame(
        stock_daily=stock.daily,
        stock_weekly=stock.weekly,
        bench_weekly=bench.weekly,
    )
    if swing_lifecycle.empty:
        st.info("Swing lifecycle unavailable for this stock.")
    else:
        swing_lifecycle = swing_lifecycle.copy()
        swing_lifecycle["Decision"] = swing_lifecycle["Production Score"].apply(
            lambda v: (
                decision_from_production_score(float(v), s_buy, s_sell)
                if pd.notna(v)
                else "No Signal"
            )
        )

        s_chart_df = prepare_lifecycle_frame(
            lifecycle=swing_lifecycle,
            score_col="Production Score",
            decision_col="Decision",
            window=104,
        )
        s_chart = lifecycle_score_chart(
            lifecycle_df=s_chart_df.set_index("Date"),
            score_col="Production Score",
            decision_col="Decision",
            buy_threshold=s_buy,
            sell_threshold=s_sell,
        )
        if s_chart is None:
            s_fallback = s_chart_df[["Date", "Production Score"]].set_index("Date")
            s_fallback["Buy Threshold"] = s_buy
            s_fallback["Sell Threshold"] = s_sell
            st.line_chart(s_fallback, use_container_width=True)
        else:
            st.altair_chart(s_chart, use_container_width=True)

        s_latest = swing_lifecycle.iloc[-1]
        s_latest_score = s_latest["Production Score"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(
            "Latest Production Score",
            f"{s_latest_score:.4f}" if pd.notna(s_latest_score) else "n/a",
        )
        sc2.metric("Decision", str(s_latest["Decision"]))
        sc3.metric("Qualified (Weekly)", "Yes" if bool(s_latest["Qualified"]) else "No")
        sc4.metric(
            "SetupType", str(s_latest["SetupType"]) if str(s_latest["SetupType"]).strip() else "n/a"
        )

        s_recent = swing_lifecycle.tail(26).copy()
        s_recent.insert(0, "Week", s_recent.index)
        s_recent = s_recent.reset_index(drop=True)
        s_recent_display = _apply_badges(
            s_recent[
                [
                    "Week",
                    "Production Score",
                    "Decision",
                    "Qualified",
                    "Rule_Alignment",
                    "Rule_Slope",
                    "Rule_RS",
                    "SetupType",
                    "RSI14_State",
                    "RSI_Accel",
                    "MACD_Hist_Sign",
                    "MACD_Hist_Accel",
                    "Price_vs_EMA20",
                    "Volume_Confirm",
                    "Volatility_Expansion",
                ]
            ].assign(
                Qualified=lambda d: d["Qualified"].apply(lambda x: "Yes" if bool(x) else "No"),
            ),
            columns=["Decision", "Qualified"],
        )
        st.dataframe(
            clean_display_df(s_recent_display),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("Latest technical indicators")
    technicals = swing_technical_snapshot(
        stock_daily=stock.daily,
        stock_weekly=stock.weekly,
        bench_weekly=bench.weekly,
    )
    if technicals.empty:
        st.info("Technical snapshot unavailable for this stock.")
    else:
        concise = technicals[
            technicals["Indicator"].isin(
                [
                    "Production Score",
                    "SetupType",
                    "Weekly Qualified",
                    "Rule Alignment",
                    "Rule EMA20 Slope",
                    "Rule RS Rising",
                    "1D Close",
                    "1D RSI14",
                    "1D MACD Histogram",
                    "1D EMA20",
                    "1D ATR14%",
                ]
            )
        ]
        st.dataframe(clean_display_df(concise), use_container_width=True, hide_index=True)
        with st.expander("Show advanced indicators", expanded=False):
            st.dataframe(clean_display_df(technicals), use_container_width=True, hide_index=True)


tab_portfolio, tab_swing, tab_stock_details = st.tabs(["Portfolio", "Swing", "Stock Details"])

with tab_portfolio:
    render_portfolio_tab()

with tab_swing:
    render_swing_tab()

with tab_stock_details:
    render_stock_details_tab()
