from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_ohlcv_fixture(name: str = "sample_ohlcv.csv") -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_DIR / name)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]
