"""Pure Plotly figure builders for the Dash UI.

Framework-agnostic (no Dash or Streamlit imports) so they stay unit-testable.
``app.py`` renders the returned figures inside ``dcc.Graph``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# TradingView-style gauge zones: Strong Sell -> Strong Buy over the score range.
_GAUGE_STEPS = [
    (-1.0, -0.6, "#E24B4A"),
    (-0.6, -0.2, "#D85A30"),
    (-0.2, 0.2, "#888780"),
    (0.2, 0.6, "#97C459"),
    (0.6, 1.0, "#639922"),
]


def rating_gauge_figure(score: float, height: int = 160) -> go.Figure:
    """Semicircular gauge with a needle at ``score`` (clamped to [-1, 1])."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if pd.isna(value):
        value = 0.0
    value = max(-1.0, min(1.0, value))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": "+.2f", "font": {"size": 22}},
            gauge={
                "shape": "angular",
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [{"range": [lo, hi], "color": color} for lo, hi, color in _GAUGE_STEPS],
                "threshold": {
                    "line": {"color": "#111", "width": 4},
                    "thickness": 0.85,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 20, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def lifecycle_score_figure(
    dates: pd.Series,
    scores: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    score_name: str = "Score",
) -> go.Figure:
    """Score-over-time line with buy/sell threshold guides."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=scores, mode="lines", name=score_name, line={"color": "#4C78A8"})
    )
    for value, color, name in (
        (buy_threshold, "#2ca02c", "Buy threshold"),
        (sell_threshold, "#d62728", "Sell threshold"),
    ):
        fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=name)
    fig.update_layout(
        height=280,
        margin={"l": 40, "r": 20, "t": 20, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=score_name,
        showlegend=False,
    )
    return fig


def lifecycle_with_price_figure(
    dates: pd.Series,
    scores: pd.Series,
    prices: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    score_name: str = "Production Score",
    price_name: str = "Price",
) -> go.Figure:
    """Score line (left axis) overlaid with the stock's price line (right axis)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=scores, mode="lines", name=score_name, line={"color": "#4C78A8", "width": 2}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name=price_name,
            line={"color": "#E45756", "width": 1.5},
            yaxis="y2",
        )
    )
    for value, color in ((buy_threshold, "#2ca02c"), (sell_threshold, "#d62728")):
        fig.add_hline(y=value, line_dash="dash", line_color=color)
    fig.update_layout(
        height=320,
        margin={"l": 45, "r": 55, "t": 20, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": score_name, "range": [-1.05, 1.05], "zeroline": True},
        yaxis2={"title": price_name, "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    return fig
