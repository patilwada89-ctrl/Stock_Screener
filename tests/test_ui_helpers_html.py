"""Pure HTML/SVG builders for the TradingView-style swing view."""

from __future__ import annotations

import numpy as np

from src.ui_helpers import (
    funnel_strip_html,
    rating_gauge_svg,
    screener_heat_table_html,
    swing_pick_card_html,
)


def test_rating_gauge_needle_tracks_score_direction():
    # +1 points right (strong buy), -1 left (strong sell), 0 straight up.
    assert 'x2="98.0"' in rating_gauge_svg(1.0)
    assert 'x2="22.0"' in rating_gauge_svg(-1.0)
    up = rating_gauge_svg(0.0)
    assert 'x2="60.0"' in up and 'y2="22.0"' in up


def test_rating_gauge_clamps_out_of_range_and_handles_bad_input():
    assert 'x2="98.0"' in rating_gauge_svg(5.0)  # clamps to +1
    assert "<svg" in rating_gauge_svg(float("nan"))  # no crash
    assert "<svg" in rating_gauge_svg(None)  # type: ignore[arg-type]


def test_funnel_strip_contains_all_stages():
    html = funnel_strip_html(universe=34, qualified=14, buy=6, watch=5, avoid=3)
    for token in ["Universe 34", "Qualified 14", "Buy 6", "Watch 5", "Avoid 3"]:
        assert token in html


def test_pick_card_has_core_fields_and_levels():
    html = swing_pick_card_html(
        ticker="ASML.AS",
        name="ASML Holding",
        region="EU",
        decision="Buy",
        prod_score=0.81,
        tv_rating="Strong Buy",
        setup="Breakout",
        risk_flag="OK",
        entry=612.4,
        stop=588.0,
        target_2r=661.2,
        featured=True,
    )
    assert "ASML.AS" in html
    assert "+0.81" in html
    assert "Strong Buy" in html
    assert "Stop 588.00" in html
    assert "2R 661.20" in html
    assert "<svg" in html


def test_pick_card_without_levels_omits_levels_row():
    html = swing_pick_card_html(
        ticker="X",
        name="X Co",
        region="US",
        decision="Hold",
        prod_score=0.1,
        tv_rating="Neutral",
        setup="Trend",
        risk_flag="Watch",
    )
    assert "Stop" not in html
    assert "2R" not in html


def test_heat_table_renders_rows_and_handles_nan_score():
    rows = [
        {
            "symbol": "ASML.AS",
            "rating": "Strong Buy",
            "score": 0.81,
            "setup": "Breakout",
            "decision": "Buy",
            "price": 612.4,
        },
        {
            "symbol": "BMW.DE",
            "rating": "Sell",
            "score": np.nan,
            "setup": "",
            "decision": "Sell",
            "price": 92.4,
        },
    ]
    html = screener_heat_table_html(rows)
    assert "ASML.AS" in html and "BMW.DE" in html
    assert "Strong Buy" in html
    assert "n/a" in html  # NaN score row
    assert "<table" in html
