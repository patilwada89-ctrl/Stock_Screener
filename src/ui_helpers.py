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


def price_decision_chart(
    price_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    decision_col: str,
    window: int = 104,
) -> object | None:
    if alt is None:
        return None
    if price_df.empty or "Close" not in price_df.columns:
        return alt.Chart(pd.DataFrame({"Date": [], "Price": []}))

    prices = price_df.copy().sort_index().tail(window).reset_index()
    date_col = prices.columns[0]
    prices = prices.rename(columns={date_col: "Date", "Close": "Price"})
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"])

    if prices.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "Price": []}))

    line = (
        alt.Chart(prices)
        .mark_line(color="#4C78A8", strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title="Price"),
            tooltip=["Date:T", "Price:Q"],
        )
    )

    if lifecycle_df.empty or decision_col not in lifecycle_df.columns:
        return line.interactive()

    lifecycle = lifecycle_df.copy()
    if "Date" not in lifecycle.columns:
        lifecycle = lifecycle.reset_index().rename(
            columns={lifecycle.index.name or "index": "Date"}
        )
    lifecycle["Date"] = pd.to_datetime(lifecycle["Date"], errors="coerce")
    lifecycle = lifecycle.dropna(subset=["Date"])
    if lifecycle.empty:
        return line.interactive()

    lifecycle = lifecycle.sort_values("Date")
    lifecycle["_prev_decision"] = lifecycle[decision_col].shift(1).fillna(lifecycle[decision_col])
    change_rows = lifecycle[lifecycle[decision_col] != lifecycle["_prev_decision"]].copy()
    change_rows = change_rows[change_rows[decision_col].isin(["Buy", "Sell"])]
    if change_rows.empty:
        return line.interactive()

    markers = change_rows.merge(prices[["Date", "Price"]], on="Date", how="left").dropna(
        subset=["Price"]
    )
    if markers.empty:
        return line.interactive()

    marker_points = (
        alt.Chart(markers)
        .mark_point(size=90)
        .encode(
            x="Date:T",
            y="Price:Q",
            color=alt.Color(
                f"{decision_col}:N",
                scale=alt.Scale(domain=["Buy", "Sell"], range=["#2ca02c", "#d62728"]),
            ),
            shape=alt.Shape(
                f"{decision_col}:N",
                scale=alt.Scale(domain=["Buy", "Sell"], range=["triangle-up", "triangle-down"]),
            ),
            tooltip=["Date:T", "Price:Q", f"{decision_col}:N"],
        )
    )
    return (line + marker_points).interactive()
