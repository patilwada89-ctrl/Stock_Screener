"""Small Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_universe_csv


def pick_universe_df(title: str, example_path: Path, key_prefix: str) -> tuple[pd.DataFrame | None, str | None]:
    st.subheader(title)
    source = st.radio(
        "Source",
        ["Use example CSV", "Upload CSV"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    try:
        if source == "Use example CSV":
            df = load_universe_csv(example_path)
            st.caption(f"Using example file: `{example_path}`")
            return df, None

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key=f"{key_prefix}_upload",
            help="CSV schema: Name, Region, SignalTicker (+ optional TradeTicker_DE, Benchmark)",
        )
        if uploaded is None:
            return None, "Upload a CSV to continue."

        df = load_universe_csv(uploaded)
        return df, None
    except Exception as exc:  # pragma: no cover - UI display branch
        return None, str(exc)


def clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in {"f"}:
            out[col] = out[col].round(4)
    return out
