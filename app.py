from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.data import fetch_ticker_data
from src.signals import (
    apply_custom_weights,
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
from src.ui_helpers import clean_display_df, pick_universe_df

st.set_page_config(page_title="Snapshot TA Screener", layout="wide")
st.title("Snapshot-Only Technical-Analysis Stock Screener")

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"

PORTFOLIO_EXAMPLE = EXAMPLES_DIR / "portfolio.csv"
WATCHLIST_EXAMPLE = EXAMPLES_DIR / "xfra_swing_trading_universe.csv"

DATAFRAME_HAS_SELECTION = "on_select" in inspect.signature(st.dataframe).parameters


def _build_data_cache(universe_df: pd.DataFrame) -> dict[str, object]:
    all_tickers = set(universe_df["SignalTicker"].dropna().astype(str)) | set(universe_df["Benchmark"].dropna().astype(str))
    cache: dict[str, object] = {}
    progress = st.progress(0.0, text="Downloading market data...")
    total = max(1, len(all_tickers))
    for i, ticker in enumerate(sorted(all_tickers), start=1):
        cache[ticker] = fetch_ticker_data(ticker)
        progress.progress(i / total, text=f"Downloaded {i}/{total}: {ticker}")
    progress.empty()
    return cache


def _coerce_threshold_pair(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
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
    sell_threshold, buy_threshold = st.slider(
        "Health Score thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=(-0.25, 0.35),
        step=0.05,
        key="portfolio_thresholds",
        help="Decision rule: score <= sell threshold => Sell, score >= buy threshold => Buy, otherwise Hold.",
    )
    out["Decision"] = out.apply(
        lambda r: decision_from_health_score(float(r["Health Score"]), buy_threshold, sell_threshold)
        if r["Status"] == "OK"
        else "No Data",
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
    _render_selectable_stock_table(
        source_df=out,
        display_df=out[display_cols],
        table_key="portfolio_main_table",
        source_label="Portfolio",
        fallback_label="Select portfolio stock for Stock Details",
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
    qualified = swing_df[(swing_df["Qualified"] == True) & (swing_df["Status"] == "OK")].copy()  # noqa: E712
    qualified = rank_qualified(qualified)

    st.subheader("1) Action Board (Qualified Stocks Only)")
    sell_threshold, buy_threshold = st.slider(
        "Production Score thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=(-0.20, 0.30),
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

    if qualified.empty:
        st.warning("No stocks currently pass the weekly hard filter.")
    else:
        qualified["Decision"] = qualified["ProductionScore"].apply(
            lambda s: decision_from_production_score(float(s), buy_threshold, sell_threshold)
        )
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
        _render_selectable_stock_table(
            source_df=qualified,
            display_df=qualified[action_cols],
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
            st.dataframe(clean_display_df(qualified[q_cols]), use_container_width=True, hide_index=True)

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
            st.dataframe(clean_display_df(custom[custom_cols]), use_container_width=True, hide_index=True)

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
        st.dataframe(clean_display_df(swing_df[status_cols]), use_container_width=True, hide_index=True)


def render_stock_details_tab() -> None:
    st.subheader("Stock Details")
    selected = st.session_state.get("selected_stock")
    if not selected:
        st.info("Click a row in Portfolio or Swing Action Board to load stock details here.")
        return

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

    st.markdown("### Portfolio Lifecycle")
    p_default = _coerce_threshold_pair(st.session_state.get("portfolio_thresholds"), (-0.25, 0.35))
    p_sell, p_buy = st.slider(
        "Portfolio health thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=p_default,
        step=0.05,
        key="stock_details_portfolio_thresholds",
    )

    portfolio_lifecycle = portfolio_lifecycle_frame(
        stock_weekly=stock.weekly,
        stock_monthly=stock.monthly,
        bench_weekly=bench.weekly,
    )
    if portfolio_lifecycle.empty:
        st.info("Portfolio lifecycle unavailable for this stock.")
    else:
        portfolio_lifecycle = portfolio_lifecycle.copy()
        portfolio_lifecycle["Decision"] = portfolio_lifecycle["Health Score"].apply(
            lambda v: decision_from_health_score(float(v), p_buy, p_sell)
        )
        p_chart = portfolio_lifecycle[["Health Score"]].copy()
        p_chart["Buy Threshold"] = p_buy
        p_chart["Sell Threshold"] = p_sell
        st.line_chart(p_chart, use_container_width=True)

        p_latest = portfolio_lifecycle.iloc[-1]
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Latest Health Score", f"{p_latest['Health Score']:.4f}")
        pc2.metric("Decision", str(p_latest["Decision"]))
        pc3.metric("Risk Flag", str(p_latest["Risk Flag"]))
        pc4.metric("1W Alignment", str(p_latest["1W Alignment"]))

        p_recent = portfolio_lifecycle.tail(26).copy()
        p_recent.insert(0, "Week", p_recent.index)
        p_recent = p_recent.reset_index(drop=True)
        st.dataframe(
            clean_display_df(
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
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Swing Lifecycle & Technicals")
    s_default = _coerce_threshold_pair(st.session_state.get("swing_thresholds"), (-0.20, 0.30))
    s_sell, s_buy = st.slider(
        "Swing production thresholds (Sell / Buy)",
        min_value=-1.0,
        max_value=1.0,
        value=s_default,
        step=0.05,
        key="stock_details_swing_thresholds",
    )

    swing_lifecycle = swing_lifecycle_frame(
        stock_daily=stock.daily,
        stock_weekly=stock.weekly,
        bench_weekly=bench.weekly,
    )
    if swing_lifecycle.empty:
        st.info("Swing lifecycle unavailable for this stock.")
    else:
        swing_lifecycle = swing_lifecycle.copy()
        swing_lifecycle["Decision"] = swing_lifecycle["Health Score"].apply(
            lambda v: decision_from_production_score(float(v), s_buy, s_sell) if pd.notna(v) else "No Signal"
        )

        s_chart = swing_lifecycle[["Health Score"]].copy()
        s_chart["Buy Threshold"] = s_buy
        s_chart["Sell Threshold"] = s_sell
        st.line_chart(s_chart, use_container_width=True)

        s_latest = swing_lifecycle.iloc[-1]
        s_latest_score = s_latest["Health Score"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Latest Health Score", f"{s_latest_score:.4f}" if pd.notna(s_latest_score) else "n/a")
        sc2.metric("Decision", str(s_latest["Decision"]))
        sc3.metric("Qualified (Weekly)", "Yes" if bool(s_latest["Qualified"]) else "No")
        sc4.metric("SetupType", str(s_latest["SetupType"]) if str(s_latest["SetupType"]).strip() else "n/a")

        s_recent = swing_lifecycle.tail(26).copy()
        s_recent.insert(0, "Week", s_recent.index)
        s_recent = s_recent.reset_index(drop=True)
        st.dataframe(
            clean_display_df(
                s_recent[
                    [
                        "Week",
                        "Health Score",
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
                ]
            ),
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
        st.dataframe(clean_display_df(technicals), use_container_width=True, hide_index=True)


tab_portfolio, tab_swing, tab_stock_details = st.tabs(["Portfolio", "Swing", "Stock Details"])

with tab_portfolio:
    render_portfolio_tab()

with tab_swing:
    render_swing_tab()

with tab_stock_details:
    render_stock_details_tab()
