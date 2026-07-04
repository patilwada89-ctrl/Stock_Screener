"""Pure display/chart helper utilities."""

from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# TradingView-style swing decision view (pure HTML/SVG string builders).
#
# These return markup that app.py renders via st.markdown(..., unsafe_allow_html=True).
# Surfaces use neutral rgba fills + inherited text so they adapt to Streamlit's
# light or dark theme; pills are opaque light fills with same-family dark text
# (readable on either theme). No Streamlit import here -- these stay pure.
# ---------------------------------------------------------------------------

_RATING_COLORS = {
    "Strong Buy": ("#97C459", "#173404"),
    "Buy": ("#C0DD97", "#173404"),
    "Neutral": ("#E7E5DD", "#2C2C2A"),
    "Sell": ("#F0997B", "#4A1B0C"),
    "Strong Sell": ("#E24B4A", "#501313"),
}
_DECISION_COLORS = {
    "Buy": ("#97C459", "#173404"),
    "Hold": ("#FAC775", "#412402"),
    "Sell": ("#F7C1C1", "#501313"),
}
_RISK_COLORS = {
    "OK": ("#C0DD97", "#173404"),
    "Watch": ("#FAC775", "#412402"),
    "Breakdown": ("#F7C1C1", "#501313"),
}
_NEUTRAL_PILL = ("#D3D1C7", "#2C2C2A")


def _pill(label: object, palette: dict[str, tuple[str, str]], *, size: int = 11) -> str:
    text = "n/a" if label is None or str(label).strip() == "" else str(label)
    bg, fg = palette.get(text, _NEUTRAL_PILL)
    return (
        f'<span style="font-size:{size}px;padding:2px 8px;border-radius:6px;'
        f'background:{bg};color:{fg};white-space:nowrap;">{text}</span>'
    )


def _fmt(value: object, nd: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def rating_gauge_svg(score: float) -> str:
    """Semicircular TradingView-style gauge; needle maps score in [-1, 1]."""
    try:
        s = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        s = 0.0
    theta = math.radians(90.0 - s * 90.0)
    x2 = 60.0 + 38.0 * math.cos(theta)
    y2 = 60.0 - 38.0 * math.sin(theta)
    arcs = (
        '<path d="M14,60 A46,46 0 0,1 22.8,33" fill="none" stroke="#E24B4A" stroke-width="11"/>'
        '<path d="M22.8,33 A46,46 0 0,1 45.8,16.3" fill="none" stroke="#D85A30" stroke-width="11"/>'
        '<path d="M45.8,16.3 A46,46 0 0,1 74.2,16.3" fill="none" stroke="#888780" stroke-width="11"/>'
        '<path d="M74.2,16.3 A46,46 0 0,1 97.2,33" fill="none" stroke="#97C459" stroke-width="11"/>'
        '<path d="M97.2,33 A46,46 0 0,1 106,60" fill="none" stroke="#639922" stroke-width="11"/>'
    )
    return (
        '<svg viewBox="0 0 120 70" style="width:150px;height:88px;display:block;'
        'margin:2px auto;color:inherit;" role="img" aria-label="Rating gauge">'
        f"{arcs}"
        f'<line x1="60" y1="60" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="currentColor" stroke-width="2.5"/>'
        '<circle cx="60" cy="60" r="4" fill="currentColor"/></svg>'
    )


def funnel_strip_html(universe: int, qualified: int, buy: int, watch: int, avoid: int) -> str:
    """Universe -> Qualified -> Buy/Watch/Avoid count chips."""

    def chip(text: str, bg: str, fg: str) -> str:
        return (
            f'<span style="font-size:12px;padding:5px 10px;border-radius:8px;'
            f'background:{bg};color:{fg};">{text}</span>'
        )

    arrow = '<span style="opacity:0.4;">›</span>'
    parts = [
        chip(f"Universe {int(universe)}", "rgba(128,128,128,0.15)", "inherit"),
        arrow,
        chip(f"Qualified {int(qualified)}", "rgba(55,138,221,0.22)", "inherit"),
        arrow,
        chip(f"Buy {int(buy)}", "#C0DD97", "#173404"),
        chip(f"Watch {int(watch)}", "#FAC775", "#412402"),
        chip(f"Avoid {int(avoid)}", "#F7C1C1", "#501313"),
    ]
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;'
        'margin:4px 0 14px;">' + "".join(parts) + "</div>"
    )


def swing_pick_card_html(
    *,
    ticker: str,
    name: str,
    region: str,
    decision: str,
    prod_score: float,
    tv_rating: str,
    setup: str,
    risk_flag: str,
    entry: float | None = None,
    stop: float | None = None,
    target_2r: float | None = None,
    featured: bool = False,
) -> str:
    """One top-pick card: gauge (Production Score) + TV rating pill + trade levels."""
    border = "2px solid rgba(55,138,221,0.85)" if featured else "1px solid rgba(128,128,128,0.25)"
    try:
        score_txt = f"{float(prod_score):+.2f}"
    except (TypeError, ValueError):
        score_txt = "—"

    levels = ""
    if entry is not None and stop is not None:
        levels = (
            '<div style="border-top:1px solid rgba(128,128,128,0.25);padding-top:6px;'
            'font-size:12px;display:flex;justify-content:space-between;gap:6px;">'
            f'<span><span style="opacity:0.65;">Entry</span> {_fmt(entry)}</span>'
            f'<span style="color:#d9534f;">Stop {_fmt(stop)}</span>'
            f'<span style="color:#5c9a2e;">2R {_fmt(target_2r)}</span></div>'
        )

    return (
        f'<div style="background:rgba(128,128,128,0.06);border:{border};border-radius:12px;'
        'padding:12px;height:100%;box-sizing:border-box;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:15px;font-weight:500;">{ticker}</span>'
        f"{_pill(decision, _DECISION_COLORS)}</div>"
        f'<div style="font-size:12px;opacity:0.65;margin-bottom:2px;">{name} · {region}</div>'
        f"{rating_gauge_svg(prod_score)}"
        f'<div style="text-align:center;margin:2px 0 8px;">{_pill(tv_rating, _RATING_COLORS)}</div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin:3px 0;">'
        f'<span style="opacity:0.65;">Prod score</span><span style="font-weight:500;">{score_txt}</span></div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">'
        f'<span style="opacity:0.65;">Setup</span><span>{setup or "—"}</span></div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:8px;'
        'align-items:center;">'
        f'<span style="opacity:0.65;">Risk</span>{_pill(risk_flag, _RISK_COLORS)}</div>'
        f"{levels}</div>"
    )


def screener_heat_table_html(rows: list[dict]) -> str:
    """Display-only color-coded screener table (rating pills + score bars)."""
    head = (
        '<thead><tr style="text-align:left;opacity:0.7;font-size:12px;">'
        '<th style="padding:8px 10px;">Symbol</th>'
        '<th style="padding:8px 10px;">Rating</th>'
        '<th style="padding:8px 10px;min-width:90px;">Score</th>'
        '<th style="padding:8px 10px;">Setup</th>'
        '<th style="padding:8px 10px;">Decision</th>'
        '<th style="padding:8px 10px;text-align:right;">Price</th></tr></thead>'
    )
    body = []
    for r in rows:
        try:
            s = float(r.get("score"))
        except (TypeError, ValueError):
            s = float("nan")
        if pd.isna(s):
            bar = '<span style="opacity:0.4;">n/a</span>'
        else:
            pct = max(0.0, min(100.0, (s + 1.0) / 2.0 * 100.0))
            color = "#639922" if s > 0 else ("#888780" if s == 0 else "#E24B4A")
            bar = (
                '<div style="background:rgba(128,128,128,0.18);border-radius:3px;height:7px;'
                f'width:100%;" title="{s:+.2f}"><div style="background:{color};height:7px;'
                f'width:{pct:.0f}%;border-radius:3px;"></div></div>'
            )
        body.append(
            '<tr style="border-top:1px solid rgba(128,128,128,0.18);font-size:12px;">'
            f'<td style="padding:7px 10px;font-weight:500;">{r.get("symbol", "")}</td>'
            f'<td style="padding:7px 10px;">{_pill(r.get("rating"), _RATING_COLORS)}</td>'
            f'<td style="padding:7px 10px;">{bar}</td>'
            f'<td style="padding:7px 10px;opacity:0.8;">{r.get("setup") or "—"}</td>'
            f'<td style="padding:7px 10px;">{_pill(r.get("decision"), _DECISION_COLORS)}</td>'
            f'<td style="padding:7px 10px;text-align:right;">{_fmt(r.get("price"))}</td></tr>'
        )
    return (
        '<div style="border:1px solid rgba(128,128,128,0.25);border-radius:12px;overflow:hidden;">'
        '<table style="width:100%;border-collapse:collapse;">'
        f'{head}<tbody>{"".join(body)}</tbody></table></div>'
    )
