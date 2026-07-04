"""Pure display helpers shared by the UI (framework-agnostic; no Dash/Streamlit)."""

from __future__ import annotations

import pandas as pd


def prepare_lifecycle_frame(
    lifecycle: pd.DataFrame,
    score_col: str,
    decision_col: str,
    window: int = 104,
) -> pd.DataFrame:
    """Tidy a lifecycle frame for charting: last ``window`` rows, a datetime
    ``Date`` column, sorted ascending, with a decision column guaranteed."""
    if lifecycle.empty or score_col not in lifecycle.columns:
        return pd.DataFrame(columns=["Date", score_col, decision_col])

    out = lifecycle.copy()
    out = out.sort_index()
    out = out.tail(window)
    out = out.reset_index().rename(columns={out.index.name or "index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    if decision_col not in out.columns:
        out[decision_col] = ""
    return out
