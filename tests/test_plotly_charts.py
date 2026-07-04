"""Offline tests for the pure Plotly figure builders."""

from __future__ import annotations

import pandas as pd

from src.plotly_charts import (
    lifecycle_score_figure,
    lifecycle_with_price_figure,
    rating_gauge_figure,
)


def test_gauge_value_clamped_and_zoned():
    fig = rating_gauge_figure(0.81)
    ind = fig.data[0]
    assert ind.value == 0.81
    assert ind.gauge.threshold.value == 0.81
    assert len(ind.gauge.steps) == 5  # five rating zones

    assert rating_gauge_figure(5.0).data[0].value == 1.0  # clamped
    assert rating_gauge_figure(-5.0).data[0].value == -1.0
    assert rating_gauge_figure(float("nan")).data[0].value == 0.0  # no crash


def test_lifecycle_figure_has_line_and_thresholds():
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    scores = pd.Series([0.1, 0.2, -0.1, 0.4, 0.3])
    fig = lifecycle_score_figure(dates, scores, buy_threshold=0.3, sell_threshold=-0.2)
    assert fig.data[0].mode == "lines"
    assert list(fig.data[0].y) == [0.1, 0.2, -0.1, 0.4, 0.3]
    # two hline shapes for the thresholds
    assert len(fig.layout.shapes) == 2


def test_lifecycle_with_price_overlay_has_two_axes():
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    scores = pd.Series([0.1, 0.2, -0.1, 0.4, 0.3])
    prices = pd.Series([10.0, 11.0, 10.5, 12.0, 12.5])
    fig = lifecycle_with_price_figure(dates, scores, prices, 0.3, -0.2)

    assert len(fig.data) == 2
    price_trace = next(t for t in fig.data if t.name == "Price")
    score_trace = next(t for t in fig.data if t.name == "Production Score")
    assert price_trace.yaxis == "y2"  # price on the secondary (right) axis
    assert list(score_trace.y) == [0.1, 0.2, -0.1, 0.4, 0.3]
    assert fig.layout.yaxis.range == (-1.05, 1.05)
    assert len(fig.layout.shapes) == 2  # buy/sell threshold lines
