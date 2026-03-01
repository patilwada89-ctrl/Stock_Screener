import pandas as pd

from src.ui_helpers import prepare_lifecycle_frame


def test_prepare_lifecycle_frame_sorts_dates_and_uses_datetime_dtype():
    idx = pd.to_datetime(["2025-03-01", "2024-01-01", "2024-12-31"])
    lifecycle = pd.DataFrame(
        {
            "Production Score": [0.2, -0.1, 0.4],
            "Decision": ["Hold", "Sell", "Buy"],
        },
        index=idx,
    )

    out = prepare_lifecycle_frame(
        lifecycle=lifecycle,
        score_col="Production Score",
        decision_col="Decision",
        window=104,
    )

    assert str(out["Date"].dtype).startswith("datetime64")
    assert out["Date"].is_monotonic_increasing
