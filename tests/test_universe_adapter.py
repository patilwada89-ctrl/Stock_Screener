"""Offline tests for the screened-CSV -> app-universe adapter."""

from __future__ import annotations

import pandas as pd

from src.universe_adapter import screened_csv_to_universe

_UNIVERSE_COLUMNS = ["Name", "Region", "SignalTicker", "TradeTicker_DE", "Benchmark"]


def test_maps_screened_csv_to_universe_schema(tmp_path):
    csv = tmp_path / "screened.csv"
    pd.DataFrame(
        {
            "yahoo_ticker": ["AAPL", "SAP.DE", ""],
            "isin": ["US0378331005", "DE0007164600", "NL0000000000"],
            "name": ["Apple", "SAP SE", "No Ticker"],
            "frankfurt_line": ["APC.F", "SAP.F", "XX.F"],
        }
    ).to_csv(csv, index=False)

    uni = screened_csv_to_universe(csv)

    assert list(uni.columns) == _UNIVERSE_COLUMNS
    assert len(uni) == 2  # the blank-ticker row is dropped

    apple = uni[uni["SignalTicker"] == "AAPL"].iloc[0]
    assert apple["Region"] == "US"
    assert apple["Benchmark"] == "SPY"
    assert apple["TradeTicker_DE"] == "APC.F"
    assert apple["Name"] == "Apple"

    sap = uni[uni["SignalTicker"] == "SAP.DE"].iloc[0]
    assert sap["Region"] == "EU"
    assert sap["Benchmark"] == "EXSA.DE"


def test_empty_csv_returns_empty_universe(tmp_path):
    csv = tmp_path / "empty.csv"
    pd.DataFrame({"yahoo_ticker": [], "isin": [], "name": []}).to_csv(csv, index=False)
    uni = screened_csv_to_universe(csv)
    assert uni.empty
    assert list(uni.columns) == _UNIVERSE_COLUMNS
