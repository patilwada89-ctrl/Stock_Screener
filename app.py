"""Dash + Plotly UI for the snapshot-only technical-analysis screener.

Four tabs, all fully functional: **Universe** (two-step Frankfurt/Xetra
download -> screen workflow via background jobs), **Swing** (weekly hard
filter + daily Production Score), **Portfolio** (monthly/weekly Health Score
tracking), and **Stock Details** (per-stock decision trace, trade levels,
technical ratings, and lifecycle chart). All four read the active universe —
the most recently screened CSV if present, else the bundled example.

All domain logic lives in ``src/`` (unchanged); this file is UI only.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from src import config, universe_adapter, universe_jobs
from src.data import fetch_ticker_data, load_universe_csv
from src.frankfurt_universe import JobCancelled
from src.plotly_charts import (
    lifecycle_score_figure,
    lifecycle_with_price_figure,
    rating_gauge_figure,
)
from src.ratings import screener_snapshot, technical_ratings
from src.signals import (
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
    swing_trade_levels,
)
from src.ui_helpers import prepare_lifecycle_frame
from src.universe_config import load_universe_config

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"
DEFAULT_SWING_CSV = EXAMPLES_DIR / "xfra_swing_trading_universe.csv"

CFG = load_universe_config()
universe_jobs.mark_interrupted(CFG)  # clear any stale "running" left by a prior process

_STATE_COLORS = {
    "idle": "secondary",
    "running": "info",
    "done": "success",
    "error": "danger",
    "stopped": "warning",
    "interrupted": "warning",
}

_LOG_STYLE = {
    "backgroundColor": "#0d1117",
    "color": "#c9d1d9",
    "fontFamily": "monospace",
    "fontSize": "12px",
    "padding": "8px 10px",
    "borderRadius": "6px",
    "maxHeight": "200px",
    "overflowY": "auto",
    "whiteSpace": "pre-wrap",
    "marginBottom": "0",
}


def _pct_label(status: dict) -> str:
    """Badge text for a job: "N%" while running, else the state name (idle/done/error/...)."""
    state = status.get("state", "idle")
    return f"{status.get('percent', 0)}%" if state == "running" else state


def _state_color(status: dict) -> str:
    """Bootstrap color name for a job's badge/alert, from ``_STATE_COLORS``."""
    return _STATE_COLORS.get(status.get("state", "idle"), "secondary")


def _csv_timestamp_label(path: Path) -> str:
    """"Created: YYYY-MM-DD HH:MM:SS" from a CSV's mtime, or "" if it doesn't exist yet."""
    if not path.exists():
        return ""
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return f"Created: {ts.strftime('%Y-%m-%d %H:%M:%S')}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def active_universe_df(active_path: str | None) -> pd.DataFrame:
    """Universe the screener runs on: the screened CSV if present, else the example."""
    if active_path and Path(active_path).exists():
        return universe_adapter.screened_csv_to_universe(active_path)
    return load_universe_csv(str(DEFAULT_SWING_CSV))


def _status_alert(status: dict, idle_text: str) -> dbc.Alert:
    """Render a job's status as a colored ``dbc.Alert`` with a spinner while running."""
    state = status.get("state", "idle")
    message = status.get("message") or idle_text
    spinner = dbc.Spinner(size="sm") if state == "running" else None
    return dbc.Alert(
        [spinner, html.Span(message, className="ms-2")],
        color=_STATE_COLORS.get(state, "secondary"),
        className="mb-0 py-2",
    )


# ---------------------------------------------------------------------------
# Universe tab
# ---------------------------------------------------------------------------


def universe_tab() -> dbc.Container:
    """Layout for the Universe tab: download/screen job cards, logs, and the active-universe table."""
    default_csv = str(CFG.universe_out_path) if CFG.universe_out_path.exists() else ""
    return dbc.Container(
        [
            html.H4("Universe builder", className="mt-3"),
            html.P(
                "Two-step workflow. Both steps hit the network and run in the "
                "background, so the UI stays responsive. Scope and thresholds come "
                "from config.yaml — there are no controls here by design.",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Step 1 — Download universe"),
                        html.P(
                            "Fetches the latest Xetra + Börse Frankfurt instruments, "
                            "resolves ISIN → Yahoo tickers, and writes a screening-ready CSV.",
                            className="text-muted small",
                        ),
                        html.Div(
                            [
                                dbc.Button("Download universe", id="btn-download", color="primary"),
                                dbc.Button(
                                    "Stop",
                                    id="btn-stop-download",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                ),
                                dbc.Button(
                                    "Save CSV file",
                                    id="btn-savefile",
                                    color="secondary",
                                    outline=True,
                                ),
                                dbc.Badge("idle", id="badge-download", color="secondary"),
                            ],
                            className="d-flex align-items-center flex-wrap gap-2",
                        ),
                        html.Div(id="status-download", className="mt-3"),
                        html.Small("", id="ts-download", className="text-muted d-block mt-1"),
                        html.Pre("", id="log-download", className="mt-2", style=_LOG_STYLE),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Step 2 — Choose a CSV, then screen"),
                        dbc.Label("CSV path to screen", html_for="input-csv-path"),
                        dbc.Input(
                            id="input-csv-path",
                            type="text",
                            value=default_csv,
                            placeholder="path to a universe CSV",
                            className="mb-2",
                        ),
                        dcc.Upload(
                            id="upload-csv",
                            children=html.Div(["Drag & drop or ", html.A("select a CSV")]),
                            className="border rounded p-3 text-center text-muted mb-3",
                            multiple=False,
                        ),
                        html.Div(
                            [
                                dbc.Button("Screen stocks", id="btn-screen", color="primary"),
                                dbc.Button(
                                    "Stop",
                                    id="btn-stop-screen",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                ),
                                dbc.Badge("idle", id="badge-screen", color="secondary"),
                            ],
                            className="d-flex align-items-center flex-wrap gap-2",
                        ),
                        html.Div(id="status-screen", className="mt-3"),
                        html.Small("", id="ts-fundamentals", className="text-muted d-block mt-1"),
                        html.Small("", id="ts-screened", className="text-muted d-block"),
                        html.Pre("", id="log-screen", className="mt-2", style=_LOG_STYLE),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Active screening universe"),
                        html.Div(id="active-universe-info", className="text-muted small mb-2"),
                        dash_table.DataTable(
                            id="table-universe",
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 13, "padding": "6px 10px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ]
                )
            ),
            dcc.Download(id="download-csv"),
        ],
        fluid=True,
    )


# ---------------------------------------------------------------------------
# Swing tab: data, compute, and TradingView-style rendering
# ---------------------------------------------------------------------------

# Server-side cache of downloaded market data (per process), mirroring the
# Streamlit TTL cache. Keyed by ticker.
_DATA_CACHE: dict[str, object] = {}


def get_ticker_data(ticker: str):
    """Fetch and cache one ticker's ``TickerData`` for the lifetime of the server process."""
    if ticker not in _DATA_CACHE:
        _DATA_CACHE[ticker] = fetch_ticker_data(ticker)
    return _DATA_CACHE[ticker]


def compute_swing(
    active_path,
    buy_threshold,
    sell_threshold,
    *,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the swing screener over the active universe. Fetches market data
    (cached), then reuses the src.signals pipeline unchanged.

    When called from the background job, *should_stop* is checked each iteration
    and *progress* reports advancement to the UI.
    """
    universe = active_universe_df(active_path)
    tickers = sorted(
        set(universe["SignalTicker"].dropna().astype(str))
        | set(universe["Benchmark"].dropna().astype(str))
    )
    cache: dict[str, object] = {}
    for i, ticker in enumerate(tickers, 1):
        if should_stop is not None and should_stop():
            raise JobCancelled(f"swing stopped while fetching data ({i}/{len(tickers)} tickers)")
        cache[ticker] = get_ticker_data(ticker)
    n_total = len(universe)

    swing_rows, screener_rows = [], []
    for idx, (_, meta) in enumerate(universe.iterrows(), 1):
        if should_stop is not None and should_stop():
            raise JobCancelled(f"swing stopped at {idx}/{n_total}")
        if progress is not None and idx % 5 == 0:
            progress(idx, n_total)
        ticker = str(meta["SignalTicker"])
        stock = cache.get(ticker)
        bench = cache.get(str(meta["Benchmark"]))
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
        swing_rows.append(row)

        srow = {
            "Symbol": meta.get("TradeTicker_DE", "") or ticker,
            "SignalTicker": ticker,
            "Name": meta["Name"],
            "SetupType": row.get("SetupType", ""),
            "Production Score": row.get("ProductionScore", np.nan),
            "Price": row.get("Price", np.nan),
            "Summary Rating": "Neutral",
        }
        if stock_status == "OK":
            snap = screener_snapshot(getattr(stock, "daily", pd.DataFrame()))
            srow["Summary Rating"] = snap.get("Summary Rating", "Neutral")
            if pd.isna(srow["Production Score"]):
                deval = daily_components(getattr(stock, "daily", pd.DataFrame()))
                if deval.get("status") == "OK":
                    srow["Production Score"] = float(deval["score"])
                    srow["SetupType"] = str(deval["setup_type"])
        screener_rows.append(srow)

    swing_df = pd.DataFrame(swing_rows)
    qualified = swing_df[swing_df["Qualified"] & (swing_df["Status"] == "OK")].copy()
    qualified = rank_qualified(qualified)
    if not qualified.empty:
        qualified["Decision"] = qualified["ProductionScore"].apply(
            lambda s: decision_from_production_score(float(s), buy_threshold, sell_threshold)
        )

    screener_df = pd.DataFrame(screener_rows)
    if not screener_df.empty:
        screener_df = screener_df.sort_values(
            "Production Score", ascending=False, na_position="last"
        ).reset_index(drop=True)
        screener_df["Decision"] = screener_df["Production Score"].apply(
            lambda s: (
                decision_from_production_score(float(s), buy_threshold, sell_threshold)
                if pd.notna(s)
                else "Hold"
            )
        )

    counts = {
        "universe": len(universe),
        "qualified": len(qualified),
        "buy": 0,
        "watch": 0,
        "avoid": 0,
    }
    if not qualified.empty:
        dec = qualified["Decision"].value_counts().to_dict()
        counts.update(
            buy=int(dec.get("Buy", 0)),
            watch=int(dec.get("Hold", 0)),
            avoid=int(dec.get("Sell", 0)),
        )
    return {"qualified": qualified, "screener": screener_df, "counts": counts}


_RATING_BADGE = {
    "Strong Buy": "success",
    "Buy": "success",
    "Neutral": "secondary",
    "Sell": "danger",
    "Strong Sell": "danger",
}
_DECISION_BADGE = {"Buy": "success", "Hold": "warning", "Sell": "danger"}


def _funnel(counts: dict) -> html.Div:
    """Row of colored count badges (Universe/Qualified/Buy/Watch/Avoid) for the Swing results header."""

    def chip(text, color):
        """One badge with the given text and Bootstrap color."""
        return dbc.Badge(text, color=color, className="me-2 p-2")

    return html.Div(
        [
            chip(f"Universe {counts['universe']}", "light"),
            chip(f"Qualified {counts['qualified']}", "info"),
            chip(f"Buy {counts['buy']}", "success"),
            chip(f"Watch {counts['watch']}", "warning"),
            chip(f"Avoid {counts['avoid']}", "danger"),
        ],
        className="mb-3",
    )


def _pick_card(pick, tv_rating, levels, featured=False):
    """Card for one top swing pick: score gauge, TV rating badge, ATR trade levels, and an Analyze button."""
    ticker = str(pick["SignalTicker"])
    decision = str(pick.get("Decision", ""))
    levels_row = []
    if levels.get("status") == "OK":
        atr_stop = levels["stops"][0]
        levels_row = [
            html.Small(f"Entry {levels['entry']:.2f}", className="text-muted"),
            html.Small(f"Stop {atr_stop['stop']:.2f}", className="text-danger ms-2"),
            html.Small(f"2R {atr_stop['targets'].get('2R'):.2f}", className="text-success ms-2"),
        ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Strong(ticker),
                        dbc.Badge(
                            decision,
                            color=_DECISION_BADGE.get(decision, "secondary"),
                            className="float-end",
                        ),
                    ]
                ),
                html.Small(str(pick.get("Name", "")), className="text-muted d-block mb-1"),
                dcc.Graph(
                    figure=rating_gauge_figure(float(pick["ProductionScore"])),
                    config={"displayModeBar": False},
                ),
                html.Div(
                    dbc.Badge(tv_rating, color=_RATING_BADGE.get(tv_rating, "secondary")),
                    className="text-center mb-2",
                ),
                html.Div(levels_row, className="mb-2"),
                dbc.Button(
                    "Analyze →",
                    id={"type": "swing-analyze", "ticker": ticker},
                    size="sm",
                    color="primary",
                    outline=True,
                ),
            ]
        ),
        className="h-100",
        style={"border": "2px solid #3778dd"} if featured else None,
    )


def _swing_heat_table(screener_df: pd.DataFrame) -> dash_table.DataTable:
    """Full Swing screener table with heat-map coloring by rating/decision/score sign."""
    cols = [
        "Symbol",
        "SignalTicker",
        "Name",
        "Summary Rating",
        "Production Score",
        "SetupType",
        "Decision",
        "Price",
    ]
    show = screener_df[[c for c in cols if c in screener_df.columns]].rename(
        columns={"Summary Rating": "Rating", "Production Score": "Score", "SetupType": "Setup"}
    )
    style_cond = []
    for label, color in (
        ("Strong Buy", "#97C459"),
        ("Buy", "#C0DD97"),
        ("Neutral", "#E7E5DD"),
        ("Sell", "#F0997B"),
        ("Strong Sell", "#E24B4A"),
    ):
        style_cond.append(
            {
                "if": {"filter_query": f'{{Rating}} = "{label}"', "column_id": "Rating"},
                "backgroundColor": color,
                "color": "#173404",
            }
        )
    for label, color in (("Buy", "#C0DD97"), ("Hold", "#FAC775"), ("Sell", "#F7C1C1")):
        style_cond.append(
            {
                "if": {"filter_query": f'{{Decision}} = "{label}"', "column_id": "Decision"},
                "backgroundColor": color,
                "color": "#173404",
            }
        )
    style_cond.append(
        {"if": {"filter_query": "{Score} > 0", "column_id": "Score"}, "color": "#3B6D11"}
    )
    style_cond.append(
        {"if": {"filter_query": "{Score} < 0", "column_id": "Score"}, "color": "#A32D2D"}
    )
    return dash_table.DataTable(
        id="table-swing",
        data=show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in show.columns],
        row_selectable="single",
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 13, "padding": "6px 10px"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=style_cond,
    )


def _render_swing_from_result(result: dict):
    """Build the Swing tab children from a pre-computed result dict."""
    qualified, screener = result["qualified"], result["screener"]
    tv_lookup = {}
    if not screener.empty:
        tv_lookup = dict(zip(screener["SignalTicker"], screener["Summary Rating"]))

    children = [_funnel(result["counts"])]
    if qualified.empty:
        children.append(dbc.Alert("No stocks pass the weekly hard filter.", color="warning"))
    else:
        children.append(html.H5("Top swing picks"))
        cards = []
        for i, (_, pick) in enumerate(qualified.head(4).iterrows()):
            ticker = str(pick["SignalTicker"])
            levels = swing_trade_levels(getattr(get_ticker_data(ticker), "daily", pd.DataFrame()))
            cards.append(
                dbc.Col(
                    _pick_card(pick, tv_lookup.get(ticker, "Neutral"), levels, featured=(i == 0)),
                    md=3,
                )
            )
        children.append(dbc.Row(cards, className="g-2 mb-4"))
    children.append(html.H5("Full screener"))
    children.append(_swing_heat_table(screener))
    return children


def swing_tab() -> dbc.Container:
    """Layout for the Swing tab: threshold sliders, run/stop controls, and the results container."""
    return dbc.Container(
        [
            html.H4("Swing screener", className="mt-3"),
            html.P(
                "Weekly hard filter gates qualification; the daily Production Score "
                "ranks qualified names. Runs on the active screening universe.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Sell / Buy thresholds"),
                            dcc.RangeSlider(
                                id="swing-thresholds",
                                min=-1.0,
                                max=1.0,
                                step=0.05,
                                value=[-0.20, 0.30],
                                marks={-1: "-1", 0: "0", 1: "1"},
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    "Run screener",
                                    id="btn-run-swing",
                                    color="primary",
                                    className="mt-4",
                                ),
                                dbc.Button(
                                    "Stop",
                                    id="btn-stop-swing",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                    className="mt-4",
                                ),
                                html.Small(
                                    "",
                                    id="swing-last-run",
                                    className="text-muted mt-4 d-inline-block",
                                ),
                            ],
                            className="d-flex align-items-center gap-2",
                        ),
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="swing-status", className="mb-2"),
            html.Div(id="swing-results"),
        ],
        fluid=True,
    )


# ---------------------------------------------------------------------------
# Portfolio tab
# ---------------------------------------------------------------------------


def _cat_styles(column: str, mapping: dict[str, str]) -> list[dict]:
    """DataTable ``style_data_conditional`` entries coloring one column's cells by category value."""
    return [
        {
            "if": {"filter_query": f'{{{column}}} = "{label}"', "column_id": column},
            "backgroundColor": color,
            "color": "#173404",
        }
        for label, color in mapping.items()
    ]


def _metric_col(label: str, value: object, width: int = 2):
    """A small labeled metric (muted label above a bold value) as a Bootstrap column."""
    return dbc.Col(
        [html.Small(label, className="text-muted d-block"), html.H5(str(value))], md=width
    )


def compute_portfolio(active_path, buy_threshold, sell_threshold):
    """Run the Portfolio tracker over the active universe: fetch/cache market data, evaluate each
    stock's Health Score, sort by risk, and attach a Buy/Hold/Sell ``Decision``.

    Returns ``(result_df, ticker_data_cache)`` — the cache is reused by the
    caller to render the lifecycle chart without re-fetching.
    """
    universe = active_universe_df(active_path)
    tickers = set(universe["SignalTicker"].dropna().astype(str)) | set(
        universe["Benchmark"].dropna().astype(str)
    )
    cache = {t: get_ticker_data(t) for t in sorted(tickers)}
    rows = []
    for _, meta in universe.iterrows():
        stock = cache.get(str(meta["SignalTicker"]))
        bench = cache.get(str(meta["Benchmark"]))
        stock_status = getattr(stock, "status", "Ticker data missing")
        if bench is None:
            stock_status = "Benchmark data missing"
        elif getattr(bench, "status", "") != "OK":
            stock_status = f"Benchmark issue: {bench.status}"
        rows.append(
            evaluate_portfolio_stock(
                meta=meta,
                stock_daily=getattr(stock, "daily", pd.DataFrame()),
                stock_weekly=getattr(stock, "weekly", pd.DataFrame()),
                stock_monthly=getattr(stock, "monthly", pd.DataFrame()),
                bench_weekly=getattr(bench, "weekly", pd.DataFrame()),
                stock_status=stock_status,
            )
        )
    out = sort_portfolio_for_risk(pd.DataFrame(rows))
    out["Decision"] = out.apply(
        lambda r: (
            decision_from_health_score(float(r["Health Score"]), buy_threshold, sell_threshold)
            if r["Status"] == "OK"
            else "No Data"
        ),
        axis=1,
    )
    return out, cache


def _portfolio_table(out: pd.DataFrame) -> dash_table.DataTable:
    """Portfolio results table with heat-map coloring by decision/risk/alignment/RS/momentum."""
    cols = [
        "Name",
        "Region",
        "SignalTicker",
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
    show = out[[c for c in cols if c in out.columns]]
    styles = (
        _cat_styles(
            "Decision",
            {"Buy": "#C0DD97", "Hold": "#FAC775", "Sell": "#F7C1C1", "No Data": "#E7E5DD"},
        )
        + _cat_styles("Risk Flag", {"OK": "#C0DD97", "Watch": "#FAC775", "Breakdown": "#F7C1C1"})
        + _cat_styles("1W Alignment", {"Strong": "#C0DD97", "Weak": "#FAC775", "Broken": "#F7C1C1"})
        + _cat_styles("1W RS", {"Rising": "#C0DD97", "Falling": "#F7C1C1"})
        + _cat_styles(
            "1W Momentum State",
            {"Strengthening": "#C0DD97", "Neutral": "#E7E5DD", "Weakening": "#F7C1C1"},
        )
    )
    return dash_table.DataTable(
        id="table-portfolio",
        data=show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in show.columns],
        row_selectable="single",
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 13, "padding": "6px 10px"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=styles,
    )


def _render_portfolio_results(active_path, buy, sell):
    """Run the Portfolio tracker and render its table plus the top-ranked stock's lifecycle chart."""
    try:
        out, cache = compute_portfolio(active_path, buy, sell)
    except Exception as exc:  # noqa: BLE001 - never crash the tab
        return dbc.Alert(f"Could not run portfolio: {exc}", color="danger")
    children = [_portfolio_table(out)]
    available = out[out["Status"] == "OK"]
    if available.empty:
        children.append(dbc.Alert("No portfolio stock has valid data.", color="warning"))
        return children
    row = available.iloc[0]
    stock = cache.get(str(row["SignalTicker"]))
    bench = cache.get(str(row.get("Benchmark", "")))
    lc = portfolio_lifecycle_frame(
        getattr(stock, "weekly", pd.DataFrame()),
        getattr(stock, "monthly", pd.DataFrame()),
        getattr(bench, "weekly", pd.DataFrame()),
    )
    if not lc.empty:
        lc = lc.copy()
        lc["Decision"] = lc["Health Score"].apply(
            lambda v: decision_from_health_score(float(v), buy, sell)
        )
        base = prepare_lifecycle_frame(lc, "Health Score", "Decision", 104)
        fig = lifecycle_score_figure(base["Date"], base["Health Score"], buy, sell, "Health Score")
        children.append(
            html.H5(f"Lifecycle — {row['SignalTicker']} · {row.get('Name', '')}", className="mt-3")
        )
        children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
    return children


def portfolio_tab() -> dbc.Container:
    """Layout for the Portfolio tab: Health Score threshold slider, run button, and results container."""
    return dbc.Container(
        [
            html.H4("Portfolio tracker", className="mt-3"),
            html.P(
                "Long-term health tracking (monthly regime, weekly alignment/RS/momentum) "
                "on the active screening universe.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Sell / Buy thresholds (Health Score)"),
                            dcc.RangeSlider(
                                id="portfolio-thresholds",
                                min=-1.0,
                                max=1.0,
                                step=0.05,
                                value=[-0.25, 0.35],
                                marks={-1: "-1", 0: "0", 1: "1"},
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Run tracker", id="btn-run-portfolio", color="primary", className="mt-4"
                        ),
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Loading(html.Div(id="portfolio-results")),
        ],
        fluid=True,
    )


# ---------------------------------------------------------------------------
# Stock Details tab
# ---------------------------------------------------------------------------


def _trade_levels_block(levels: dict):
    """Render ``swing_trade_levels`` output as an entry/ATR summary plus a stop/target table."""
    if levels.get("status") != "OK":
        return dbc.Alert(f"Trade levels unavailable: {levels.get('status', 'n/a')}", color="info")
    rows = []
    for stop_row in levels["stops"]:
        rec = {
            "Stop Type": stop_row["type"],
            "Stop": round(stop_row["stop"], 2) if pd.notna(stop_row["stop"]) else None,
            "Risk/Share": round(stop_row["risk_per_share"], 2)
            if pd.notna(stop_row["risk_per_share"])
            else None,
        }
        for label, val in stop_row["targets"].items():
            rec[label] = round(val, 2) if pd.notna(val) else None
        rows.append(rec)
    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0]],
        style_cell={"fontSize": 13, "padding": "6px 10px"},
        style_header={"fontWeight": "bold"},
    )
    return html.Div(
        [
            dbc.Row(
                [
                    _metric_col("Entry (last close)", f"{levels['entry']:.2f}"),
                    _metric_col("ATR(14)", f"{levels['atr14']:.2f}"),
                ]
            ),
            table,
            html.Small(
                "Deterministic arithmetic from price and ATR; no position sizing, not advice.",
                className="text-muted",
            ),
        ]
    )


def _ratings_block(ratings: dict):
    """Render ``technical_ratings`` output as three TradingView-style rating cards."""
    if ratings.get("status") != "OK":
        return dbc.Alert(f"Ratings unavailable: {ratings.get('status', 'n/a')}", color="info")
    blocks = [
        ("Oscillators", ratings["oscillators"]),
        ("Summary", ratings["summary"]),
        ("Moving Averages", ratings["moving_averages"]),
    ]
    cols = []
    for title, block in blocks:
        cols.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Strong(title),
                            dbc.Badge(
                                str(block["label"]),
                                color=_RATING_BADGE.get(str(block["label"]), "secondary"),
                                className="d-block my-2",
                            ),
                            html.Small(
                                f"Buy {block['buy']} · Neutral {block['neutral']} · Sell {block['sell']}",
                                className="text-muted",
                            ),
                        ]
                    )
                ),
                md=4,
            )
        )
    return dbc.Row(cols, className="g-2")


def _why_block(trace):
    """Render a ``DecisionTrace`` as side-by-side weekly-rules and daily-component-signal tables."""
    rules = pd.DataFrame(
        [
            {"Rule": r.name, "Pass": "Yes" if r.passed else "No", "Value": r.value}
            for r in trace.rules
        ]
    )
    comps = pd.DataFrame(
        [
            {
                "Component": c.name,
                "Signal": {1: "Buy", 0: "Neutral", -1: "Sell"}.get(c.signal),
                "Value": c.value,
            }
            for c in trace.components
        ]
    )
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Small("Weekly hard filter", className="text-muted"),
                    dash_table.DataTable(
                        data=rules.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in rules.columns],
                        style_cell={"fontSize": 12, "padding": "5px 8px", "textAlign": "left"},
                        style_data_conditional=_cat_styles(
                            "Pass", {"Yes": "#C0DD97", "No": "#F7C1C1"}
                        ),
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    html.Small("Daily component signals", className="text-muted"),
                    dash_table.DataTable(
                        data=comps.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in comps.columns],
                        style_cell={"fontSize": 12, "padding": "5px 8px", "textAlign": "left"},
                        style_data_conditional=_cat_styles(
                            "Signal", {"Buy": "#C0DD97", "Neutral": "#E7E5DD", "Sell": "#F7C1C1"}
                        ),
                    ),
                ],
                md=6,
            ),
        ]
    )


def _swing_lifecycle_block(stock, bench, buy, sell):
    """Chart the trailing-52-week Production Score history overlaid with the stock's weekly close."""
    lc = swing_lifecycle_frame(
        stock_daily=stock.daily, stock_weekly=stock.weekly, bench_weekly=bench.weekly
    )
    if lc.empty:
        return dbc.Alert("Swing lifecycle unavailable for this stock.", color="info")
    lc = lc.copy()
    lc["Decision"] = lc["Production Score"].apply(
        lambda v: (
            decision_from_production_score(float(v), buy, sell) if pd.notna(v) else "No Signal"
        )
    )
    # ~1 year of weekly bars (trailing 52 weeks).
    base = prepare_lifecycle_frame(lc, "Production Score", "Decision", 52)
    if base.empty:
        return dbc.Alert("Not enough lifecycle history yet.", color="info")

    # Overlay the selected stock's weekly close (right axis), aligned to the lifecycle weeks.
    dates = pd.to_datetime(base["Date"])
    prices = pd.Series([np.nan] * len(dates), index=dates)
    if "Close" in stock.weekly.columns:
        weekly_close = stock.weekly["Close"].copy()
        weekly_close.index = pd.to_datetime(weekly_close.index)
        prices = weekly_close.reindex(dates)

    fig = lifecycle_with_price_figure(dates, base["Production Score"], prices, buy, sell)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _render_stock_details_content(selected, active_path, thresholds, atr_mult, lookback):
    """Full Stock Details body for the selected ticker: header, decision card, trade levels,
    ratings, "why" breakdown, and the swing lifecycle chart. Returns an alert if no stock is
    selected or its data/benchmark is unavailable."""
    if not selected or not selected.get("ticker"):
        return dbc.Alert(
            "Select a stock from the Swing or Portfolio tab (row select or Analyze →) to load details.",
            color="light",
        )
    ticker = str(selected["ticker"])
    universe = active_universe_df(active_path)
    match = universe[universe["SignalTicker"].astype(str) == ticker]
    if not match.empty:
        m = match.iloc[0]
        name, benchmark, region = (
            str(m.get("Name", ticker)),
            str(m.get("Benchmark", "")),
            str(m.get("Region", "")),
        )
    else:
        name, benchmark, region = ticker, "", ""
    if not benchmark:
        benchmark = config.REGION_TO_BENCHMARK.get(region.upper(), config.US_BENCHMARK)

    stock, bench = get_ticker_data(ticker), get_ticker_data(benchmark)
    header = dbc.Row(
        [
            _metric_col("Source", selected.get("source", "")),
            _metric_col("Name", name, width=3),
            _metric_col("Ticker", ticker),
            _metric_col("Benchmark", benchmark),
        ]
    )
    if getattr(stock, "status", "") != "OK":
        return [header, dbc.Alert(f"Stock data unavailable: {stock.status}", color="warning")]
    if getattr(bench, "status", "") != "OK":
        return [header, dbc.Alert(f"Benchmark data unavailable: {bench.status}", color="warning")]

    sell, buy = thresholds
    trace = build_swing_decision_trace(
        stock_daily=stock.daily,
        stock_weekly=stock.weekly,
        bench_weekly=bench.weekly,
        buy_threshold=buy,
        sell_threshold=sell,
        signal_ticker=ticker,
        benchmark=benchmark,
        name=name,
    )
    if trace is None:
        return [header, dbc.Alert("Decision snapshot unavailable for this stock.", color="info")]

    decision_card = dbc.Row(
        [
            _metric_col("Production Score", f"{trace.score:.4f}"),
            _metric_col("Decision", trace.decision),
            _metric_col("Qualified", "Yes" if trace.qualified else "No"),
            _metric_col("Setup", trace.setup_type),
            _metric_col("Risk Flag", trace.risk_flag),
        ]
    )
    levels = swing_trade_levels(
        stock.daily, atr_mult=float(atr_mult or 2.0), swing_lookback=int(lookback or 10)
    )
    return [
        header,
        html.Hr(),
        html.H5("Decision card"),
        html.Small(f"Risk reason: {trace.risk_reason}", className="text-muted"),
        decision_card,
        html.Hr(),
        html.H5("Trade levels (risk & targets)"),
        _trade_levels_block(levels),
        html.Hr(),
        html.H5("Technical ratings"),
        _ratings_block(technical_ratings(stock.daily)),
        html.Hr(),
        html.H5("Why this decision"),
        _why_block(trace),
        html.Hr(),
        html.H5("Swing lifecycle — 1Y (score vs price)"),
        _swing_lifecycle_block(stock, bench, buy, sell),
    ]


def stock_details_tab() -> dbc.Container:
    """Layout for the Stock Details tab: thresholds, ATR-stop/lookback inputs, and the details container."""
    return dbc.Container(
        [
            html.H4("Stock details", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Sell / Buy thresholds"),
                            dcc.RangeSlider(
                                id="sd-thresholds",
                                min=-1.0,
                                max=1.0,
                                step=0.05,
                                value=[-0.20, 0.30],
                                marks={-1: "-1", 0: "0", 1: "1"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("ATR stop ×"),
                            dbc.Input(
                                id="sd-atr-mult",
                                type="number",
                                value=2.0,
                                min=0.5,
                                max=10,
                                step=0.5,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Swing-low lookback"),
                            dbc.Input(
                                id="sd-lookback", type="number", value=10, min=2, max=60, step=1
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Loading(html.Div(id="details-content")),
        ],
        fluid=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Snapshot TA Screener",
)
server = app.server

app.layout = dbc.Container(
    [
        html.H2("Snapshot-only technical-analysis stock screener", className="mt-3"),
        dcc.Tabs(
            id="tabs",
            value="universe",
            children=[
                dcc.Tab(label="Universe", value="universe", children=universe_tab()),
                dcc.Tab(label="Swing", value="swing", children=swing_tab()),
                dcc.Tab(label="Portfolio", value="portfolio", children=portfolio_tab()),
                dcc.Tab(label="Stock Details", value="details", children=stock_details_tab()),
            ],
        ),
        dcc.Store(id="store-active-universe"),
        dcc.Store(id="store-selected-stock"),
        dcc.Store(id="store-dl-signal"),
        dcc.Store(id="store-scr-signal"),
        dcc.Store(id="store-stop-download"),
        dcc.Store(id="store-stop-screen"),
        dcc.Store(id="store-swing-signal"),
        dcc.Store(id="store-stop-swing"),
        dcc.Store(id="store-swing-done"),
        dcc.Interval(id="job-poll", interval=2000, n_intervals=0),
        dcc.Interval(id="swing-job-poll", interval=2000, n_intervals=0),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("store-dl-signal", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True,
)
def _start_download(_n):
    """Kick off the background download job when "Download universe" is clicked."""
    universe_jobs.start_download_job(CFG)
    return {"clicked": _n}


@app.callback(
    Output("store-scr-signal", "data"),
    Input("btn-screen", "n_clicks"),
    State("input-csv-path", "value"),
    prevent_initial_call=True,
)
def _start_screen(_n, csv_path):
    """Kick off the background screen job over ``csv_path`` when "Screen stocks" is clicked."""
    if not csv_path or not Path(csv_path).exists():
        universe_jobs._set_status(  # surface a clear message without starting a job
            CFG, universe_jobs.SCREEN, "error", f"CSV not found: {csv_path or '(empty)'}"
        )
        return no_update
    universe_jobs.start_screen_job(CFG, csv_path)
    return {"clicked": _n}


@app.callback(
    Output("download-csv", "data"),
    Input("btn-savefile", "n_clicks"),
    prevent_initial_call=True,
)
def _save_file(_n):
    """Send the downloaded universe CSV to the browser when "Save CSV file" is clicked."""
    if CFG.universe_out_path.exists():
        return dcc.send_file(str(CFG.universe_out_path))
    return no_update


@app.callback(
    Output("input-csv-path", "value"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def _receive_upload(contents, filename):
    """Decode an uploaded CSV, save it under ``data/``, and populate the CSV-path input with it."""
    if not contents:
        return no_update
    import base64

    _header, b64 = contents.split(",", 1)
    dest = Path(CFG.data_dir) / (filename or "uploaded_universe.csv")
    dest.write_bytes(base64.b64decode(b64))
    return str(dest)


@app.callback(
    Output("status-download", "children"),
    Output("status-screen", "children"),
    Output("btn-download", "disabled"),
    Output("btn-screen", "disabled"),
    Output("btn-stop-download", "disabled"),
    Output("btn-stop-screen", "disabled"),
    Output("badge-download", "children"),
    Output("badge-download", "color"),
    Output("badge-screen", "children"),
    Output("badge-screen", "color"),
    Output("log-download", "children"),
    Output("log-screen", "children"),
    Output("ts-download", "children"),
    Output("ts-fundamentals", "children"),
    Output("ts-screened", "children"),
    Output("store-active-universe", "data"),
    Input("job-poll", "n_intervals"),
)
def _poll_jobs(_n):
    """Interval tick: refresh download/screen status alerts, badges, logs, CSV timestamps, and the active-universe path."""
    dl = universe_jobs.job_status(CFG, universe_jobs.DOWNLOAD)
    scr = universe_jobs.job_status(CFG, universe_jobs.SCREEN)
    dl_running = universe_jobs.is_running(universe_jobs.DOWNLOAD)
    scr_running = universe_jobs.is_running(universe_jobs.SCREEN)
    active = None
    if scr.get("state") == "done" and scr.get("output_path") and Path(scr["output_path"]).exists():
        active = scr["output_path"]
    return (
        _status_alert(dl, "Idle. Click “Download universe” to fetch the latest instruments."),
        _status_alert(scr, "Idle. Choose a CSV above, then click “Screen stocks”."),
        dl_running,
        scr_running,
        not dl_running,
        not scr_running,
        _pct_label(dl),
        _state_color(dl),
        _pct_label(scr),
        _state_color(scr),
        universe_jobs.read_log(CFG, universe_jobs.DOWNLOAD),
        universe_jobs.read_log(CFG, universe_jobs.SCREEN),
        _csv_timestamp_label(CFG.universe_out_path),
        _csv_timestamp_label(CFG.universe_fundamentals_out_path),
        _csv_timestamp_label(CFG.screened_out_path),
        active,
    )


@app.callback(
    Output("store-stop-download", "data"),
    Input("btn-stop-download", "n_clicks"),
    prevent_initial_call=True,
)
def _stop_download(_n):
    """Request cancellation of the running download job."""
    universe_jobs.stop_job(CFG, universe_jobs.DOWNLOAD)
    return {"stopped": _n}


@app.callback(
    Output("store-stop-screen", "data"),
    Input("btn-stop-screen", "n_clicks"),
    prevent_initial_call=True,
)
def _stop_screen(_n):
    """Request cancellation of the running screen job."""
    universe_jobs.stop_job(CFG, universe_jobs.SCREEN)
    return {"stopped": _n}


@app.callback(
    Output("table-universe", "data"),
    Output("table-universe", "columns"),
    Output("active-universe-info", "children"),
    Input("store-active-universe", "data"),
)
def _render_universe(active_path):
    """Populate the Universe tab's table from the active (screened or example) universe CSV."""
    try:
        df = active_universe_df(active_path)
    except Exception as exc:  # noqa: BLE001 - never crash the tab on a bad CSV
        return [], [], f"Could not load universe: {exc}"
    source = (
        f"Screened set → {Path(active_path).name}"
        if active_path
        else f"Example universe → {DEFAULT_SWING_CSV.name} (screen a CSV to replace it)"
    )
    info = f"{source} · {len(df)} tickers"
    columns = [{"name": c, "id": c} for c in df.columns]
    return df.to_dict("records"), columns, info


@app.callback(
    Output("store-swing-signal", "data"),
    Input("btn-run-swing", "n_clicks"),
    State("swing-thresholds", "value"),
    State("store-active-universe", "data"),
    prevent_initial_call=True,
)
def _start_swing(_n, thresholds, active_path):
    """Kick off the background swing-screener job when "Run screener" is clicked."""
    sell, buy = thresholds
    universe_jobs.start_swing_job(CFG, compute_swing, active_path, buy, sell)
    return {"clicked": _n}


@app.callback(
    Output("store-stop-swing", "data"),
    Input("btn-stop-swing", "n_clicks"),
    prevent_initial_call=True,
)
def _stop_swing(_n):
    """Request cancellation of the running swing-screener job."""
    universe_jobs.stop_job(CFG, universe_jobs.SWING)
    return {"stopped": _n}


@app.callback(
    Output("swing-results", "children"),
    Output("swing-status", "children"),
    Output("btn-run-swing", "disabled"),
    Output("btn-stop-swing", "disabled"),
    Output("swing-last-run", "children"),
    Input("swing-job-poll", "n_intervals"),
)
def _poll_swing(_n):
    """Interval tick: refresh swing-job status/last-run label and render cached results once done."""
    status = universe_jobs.job_status(CFG, universe_jobs.SWING)
    state = status.get("state", "idle")
    running = universe_jobs.is_running(universe_jobs.SWING)

    # Status bar (only visible while running / after error or stop)
    status_alert = ""
    if state == "running":
        status_alert = _status_alert(status, "Running swing screener…")
    elif state in ("error", "stopped"):
        status_alert = _status_alert(status, "")

    # Last-run timestamp
    last_run = ""
    if state == "done" and status.get("updated_at"):
        try:
            ts = datetime.fromisoformat(status["updated_at"])
            last_run = f"Last run: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        except (ValueError, TypeError):
            last_run = ""

    # Results panel — render from cached pickle when done
    results = no_update
    result_path = universe_jobs.swing_result_path(CFG)
    if state == "done" and result_path.exists():
        try:
            result = pickle.loads(result_path.read_bytes())
            results = _render_swing_from_result(result)
        except Exception as exc:  # noqa: BLE001
            results = dbc.Alert(f"Could not load cached results: {exc}", color="danger")

    return (
        results,
        status_alert,
        running,         # disable Run while running
        not running,     # disable Stop while NOT running
        last_run,
    )


@app.callback(
    Output("store-selected-stock", "data", allow_duplicate=True),
    Input({"type": "swing-analyze", "ticker": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _analyze_pick(clicks):
    """Select a stock for Stock Details when its "Analyze →" button is clicked (pattern-matched ID)."""
    if not ctx.triggered_id or not any(c for c in (clicks or []) if c):
        return no_update
    return {"ticker": ctx.triggered_id["ticker"], "source": "Swing"}


@app.callback(
    Output("store-selected-stock", "data", allow_duplicate=True),
    Input("table-swing", "selected_rows"),
    State("table-swing", "data"),
    prevent_initial_call=True,
)
def _select_from_table(selected_rows, data):
    """Select a stock for Stock Details from a row selection in the Swing screener table."""
    if not selected_rows or not data:
        return no_update
    row = data[selected_rows[0]]
    return {"ticker": row.get("SignalTicker"), "source": "Swing"}


@app.callback(
    Output("portfolio-results", "children"),
    Input("btn-run-portfolio", "n_clicks"),
    State("portfolio-thresholds", "value"),
    State("store-active-universe", "data"),
    prevent_initial_call=True,
)
def _run_portfolio(_n, thresholds, active_path):
    """Run and render the Portfolio tracker when "Run tracker" is clicked."""
    sell, buy = thresholds
    return _render_portfolio_results(active_path, buy, sell)


@app.callback(
    Output("store-selected-stock", "data", allow_duplicate=True),
    Input("table-portfolio", "selected_rows"),
    State("table-portfolio", "data"),
    prevent_initial_call=True,
)
def _select_from_portfolio(selected_rows, data):
    """Select a stock for Stock Details from a row selection in the Portfolio table."""
    if not selected_rows or not data:
        return no_update
    return {"ticker": data[selected_rows[0]].get("SignalTicker"), "source": "Portfolio"}


@app.callback(
    Output("details-content", "children"),
    Input("store-selected-stock", "data"),
    Input("sd-thresholds", "value"),
    Input("sd-atr-mult", "value"),
    Input("sd-lookback", "value"),
    State("store-active-universe", "data"),
)
def _render_details(selected, thresholds, atr_mult, lookback, active_path):
    """Re-render the Stock Details tab whenever the selected stock or its input controls change."""
    return _render_stock_details_content(selected, active_path, thresholds, atr_mult, lookback)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)
