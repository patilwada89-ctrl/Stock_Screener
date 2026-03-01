from io import StringIO

from src.data import load_universe_csv


def test_load_universe_csv_accepts_semicolon_delimited_input():
    raw = StringIO(
        "Name;Region;SignalTicker;TradeTicker_DE;Benchmark\n"
        "APPLE INC.;US;APC.F;APC;SPY\n"
        "SIEMENS ENERGY AG NA O.N.;EU;ENR.F;ENR;EXSA.DE\n"
    )

    df = load_universe_csv(raw)

    assert list(df.columns) == ["Name", "Region", "SignalTicker", "TradeTicker_DE", "Benchmark"]
    assert len(df) == 2
    assert df.loc[0, "SignalTicker"] == "APC.F"
    assert df.loc[1, "Benchmark"] == "EXSA.DE"


def test_load_universe_csv_maps_row_region_alias():
    raw = StringIO(
        "Name,Region,SignalTicker,TradeTicker_DE,Benchmark\n"
        '"ORIENT OVERS. NEW  DL-,10",ROW,ORI1.F,ORI1,EXSA.DE\n'
        '"ALPHABET INC.CL.A DL-,001",ROW,ABEA.F,ABEA,SPY\n'
    )

    df = load_universe_csv(raw)

    assert df.loc[0, "Region"] == "EU"
    assert df.loc[1, "Region"] == "US"
