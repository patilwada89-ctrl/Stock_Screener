"""Pure display/chart helper utilities."""

from __future__ import annotations

import pandas as pd

try:
    import altair as alt
except Exception:  # pragma: no cover - optional fallback in test runtime
    alt = None


def clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in {"f"}:
            out[col] = out[col].round(4)
    return out


def prepare_lifecycle_frame(
    lifecycle: pd.DataFrame,
    score_col: str,
    decision_col: str,
    window: int = 104,
) -> pd.DataFrame:
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


def decision_change_points(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    if df.empty or decision_col not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["_prev_decision"] = out[decision_col].shift(1).fillna(out[decision_col])
    changes = out[out[decision_col] != out["_prev_decision"]].drop(columns=["_prev_decision"])
    return changes


def lifecycle_score_chart(
    lifecycle_df: pd.DataFrame,
    score_col: str,
    decision_col: str,
    buy_threshold: float,
    sell_threshold: float,
) -> object | None:
    if alt is None:
        return None
    base = prepare_lifecycle_frame(
        lifecycle_df, score_col=score_col, decision_col=decision_col, window=104
    )
    if base.empty:
        return alt.Chart(pd.DataFrame({"Date": [], score_col: []}))

    line = (
        alt.Chart(base)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(f"{score_col}:Q", title=score_col),
            tooltip=["Date:T", f"{score_col}:Q", f"{decision_col}:N"],
        )
    )

    thresholds = pd.DataFrame(
        {
            "Date": [
                base["Date"].min(),
                base["Date"].max(),
                base["Date"].min(),
                base["Date"].max(),
            ],
            "Line": ["Buy Threshold", "Buy Threshold", "Sell Threshold", "Sell Threshold"],
            "Value": [buy_threshold, buy_threshold, sell_threshold, sell_threshold],
        }
    )
    threshold_line = (
        alt.Chart(thresholds)
        .mark_line(strokeDash=[5, 5])
        .encode(
            x="Date:T",
            y="Value:Q",
            color=alt.Color(
                "Line:N",
                scale=alt.Scale(
                    domain=["Buy Threshold", "Sell Threshold"], range=["#2ca02c", "#d62728"]
                ),
            ),
        )
    )

    changes = decision_change_points(base, decision_col=decision_col)
    points = (
        alt.Chart(changes)
        .mark_point(size=75, color="#ff7f0e")
        .encode(
            x="Date:T",
            y=f"{score_col}:Q",
            tooltip=["Date:T", f"{decision_col}:N", f"{score_col}:Q"],
        )
    )

    return (line + threshold_line + points).interactive()
