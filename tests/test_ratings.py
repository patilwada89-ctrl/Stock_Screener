import numpy as np
import pandas as pd

from src.ratings import technical_ratings


def test_technical_ratings_returns_expected_blocks():
    idx = pd.date_range("2021-01-01", periods=260, freq="B")
    daily = pd.DataFrame(
        {
            "Close": np.linspace(50, 140, len(idx)),
            "High": np.linspace(51, 141, len(idx)),
            "Low": np.linspace(49, 139, len(idx)),
            "Volume": np.linspace(1000, 4000, len(idx)),
        },
        index=idx,
    )

    out = technical_ratings(daily)

    assert out["status"] == "OK"
    assert set(out.keys()) >= {"oscillators", "summary", "moving_averages"}
    for key in ["oscillators", "summary", "moving_averages"]:
        assert {"label", "buy", "neutral", "sell", "score"}.issubset(out[key].keys())
