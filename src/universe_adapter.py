"""Adapt a screened universe CSV into the app's screener universe schema.

The Frankfurt pipeline exports ``yahoo_ticker, isin, name, frankfurt_line, ...``;
the screener tabs consume the ``load_universe_csv`` contract of
``Name, Region, SignalTicker, TradeTicker_DE, Benchmark``. This maps between them
so the existing screener runs only over the screened tickers, unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

_UNIVERSE_COLUMNS = ["Name", "Region", "SignalTicker", "TradeTicker_DE", "Benchmark"]


def _region_from_isin(isin: object) -> str:
    """ "US" for an ISIN starting with the US country code, else "EU" (the app's only two regions)."""
    return "US" if str(isin or "").strip().upper().startswith("US") else "EU"


def screened_csv_to_universe(path: str | Path) -> pd.DataFrame:
    """Return a universe DataFrame (Name/Region/SignalTicker/TradeTicker_DE/Benchmark)."""
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS)

    out = pd.DataFrame(index=df.index)
    if "yahoo_ticker" in df.columns:
        out["SignalTicker"] = df["yahoo_ticker"].astype(str).str.strip()
    else:
        out["SignalTicker"] = ""

    if "name" in df.columns:
        out["Name"] = df["name"].astype(str)
    elif "longName" in df.columns:
        out["Name"] = df["longName"].astype(str)
    else:
        out["Name"] = out["SignalTicker"]

    if "isin" in df.columns:
        out["Region"] = df["isin"].map(_region_from_isin)
    else:
        out["Region"] = "EU"

    if "frankfurt_line" in df.columns:
        out["TradeTicker_DE"] = df["frankfurt_line"].fillna("").astype(str).str.strip()
    else:
        out["TradeTicker_DE"] = ""

    out["Benchmark"] = out["Region"].map(config.REGION_TO_BENCHMARK).fillna(config.EU_BENCHMARK)

    # Drop rows without a usable signal ticker.
    out = out[out["SignalTicker"].ne("") & out["SignalTicker"].ne("nan")]
    return out[_UNIVERSE_COLUMNS].reset_index(drop=True)
