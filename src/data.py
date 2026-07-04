"""Data loading, validation, downloading, and resampling."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src import config


@dataclass
class TickerData:
    """Daily/weekly/monthly OHLCV frames for one ticker plus a fetch ``status``."""

    ticker: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    status: str


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column labels (defends against messy CSV exports)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_csv_autodetect(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, sniffing the delimiter.

    Handles Dash upload buffers (bytes, BOM-prefixed) the same way as plain
    file paths so callers don't need to special-case the upload source.
    """
    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            text = str(raw)
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return pd.read_csv(source, sep=None, engine="python")


def _normalize_region(raw_region: str, benchmark: str) -> str:
    """Map free-form region labels (ROW, EMEA, USA, ...) to the canonical ``EU``/``US``.

    ``ROW``/``WORLD``/``GLOBAL`` are disambiguated using the row's benchmark:
    US-benchmarked rows become ``US``, everything else defaults to ``EU``.
    """
    region = str(raw_region or "").upper().strip()
    bench = str(benchmark or "").upper().strip()

    if region in {"EU", "US"}:
        return region

    if region in {"ROW", "WORLD", "GLOBAL"}:
        if bench == config.US_BENCHMARK.upper():
            return "US"
        return "EU"

    if region in {"DE", "GERMANY", "EUROPE", "EMEA"}:
        return "EU"
    if region in {"USA", "NA", "NORTH AMERICA", "NORTHAMERICA"}:
        return "US"

    return region


def load_universe_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load, validate, and normalize an uploaded/screened universe CSV.

    Enforces ``config.REQUIRED_CSV_COLUMNS``, fills optional columns, normalizes
    ``Region`` to EU/US, drops rows with no ``SignalTicker``, and resolves each
    row's benchmark. Raises ``ValueError`` on missing or invalid columns.
    """
    df = _read_csv_autodetect(source)
    df = _strip_columns(df)

    missing = [c for c in config.REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for col in config.OPTIONAL_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[config.REQUIRED_CSV_COLUMNS + config.OPTIONAL_CSV_COLUMNS].copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["SignalTicker"] = df["SignalTicker"].astype(str).str.strip()
    df["TradeTicker_DE"] = df["TradeTicker_DE"].fillna("").astype(str).str.strip()
    df["Benchmark"] = df["Benchmark"].fillna("").astype(str).str.strip()
    df["Region"] = df.apply(lambda row: _normalize_region(row["Region"], row["Benchmark"]), axis=1)

    invalid_regions = sorted(set(df.loc[~df["Region"].isin(["US", "EU"]), "Region"].tolist()))
    if invalid_regions:
        raise ValueError(
            f"Invalid Region values: {', '.join(invalid_regions)}. Use EU/US (aliases like ROW are auto-mapped)."
        )

    df = df[df["SignalTicker"] != ""].reset_index(drop=True)
    df["Benchmark"] = df.apply(resolve_benchmark, axis=1)
    return df


def resolve_benchmark(row: pd.Series) -> str:
    """Return the row's explicit ``Benchmark`` override, else the region default."""
    override = str(row.get("Benchmark", "") or "").strip()
    if override:
        return override
    region = str(row.get("Region", "")).upper().strip()
    return config.REGION_TO_BENCHMARK.get(region, config.US_BENCHMARK)


def download_history(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV history for ``ticker`` via yfinance.

    Prefers ``Adj Close`` for the ``Close`` column when available. Returns an
    empty frame (never raises) if yfinance is unavailable or the ticker has no data.
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    raw = yf.download(
        ticker,
        period=config.YF_LOOKBACK_PERIOD,
        interval=config.YF_INTERVAL,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    price_col = (
        "Adj Close" if "Adj Close" in raw.columns and raw["Adj Close"].notna().any() else "Close"
    )

    out = pd.DataFrame(index=raw.index)
    out["Open"] = raw["Open"] if "Open" in raw.columns else pd.NA
    out["High"] = raw["High"] if "High" in raw.columns else pd.NA
    out["Low"] = raw["Low"] if "Low" in raw.columns else pd.NA
    out["Close"] = raw[price_col]
    out["Volume"] = raw["Volume"] if "Volume" in raw.columns else pd.NA
    out = out.dropna(subset=["Close"])
    return out


def _drop_incomplete_last_period(
    resampled: pd.DataFrame,
    source_last_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Drop the trailing resampled bar if the source data doesn't yet cover its full period.

    Prevents an in-progress week/month from being scored as if it were a
    completed candle.
    """
    if resampled.empty:
        return resampled

    last_period_end = resampled.index[-1]
    if source_last_timestamp < last_period_end:
        return resampled.iloc[:-1]
    return resampled


def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to Friday-ending weekly bars, dropping any incomplete final week."""
    if df_daily.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    weekly = df_daily.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    weekly = weekly.dropna(subset=["Close"])
    weekly = _drop_incomplete_last_period(weekly, df_daily.index.max())
    return weekly


def to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to calendar-month-end bars, dropping any incomplete final month."""
    if df_daily.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    monthly = df_daily.resample(pd.offsets.MonthEnd()).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    monthly = monthly.dropna(subset=["Close"])
    monthly = _drop_incomplete_last_period(monthly, df_daily.index.max())
    return monthly


def fetch_ticker_data(ticker: str) -> TickerData:
    """Download and resample one ticker's history, gating on minimum bar counts.

    Requires >=210 weekly bars and >=24 monthly bars (the longest lookbacks
    used by the scoring rules); short-circuits with a descriptive ``status``
    otherwise so callers can surface why a ticker was skipped.
    """
    daily = download_history(ticker)
    if daily.empty:
        return TickerData(
            ticker=ticker,
            daily=daily,
            weekly=pd.DataFrame(),
            monthly=pd.DataFrame(),
            status="Download failed or empty data",
        )

    weekly = to_weekly(daily)
    monthly = to_monthly(daily)

    if len(weekly) < 210:
        return TickerData(
            ticker=ticker,
            daily=daily,
            weekly=weekly,
            monthly=monthly,
            status="Insufficient weekly history (<210 bars)",
        )

    if len(monthly) < 24:
        return TickerData(
            ticker=ticker,
            daily=daily,
            weekly=weekly,
            monthly=monthly,
            status="Insufficient monthly history",
        )

    return TickerData(ticker=ticker, daily=daily, weekly=weekly, monthly=monthly, status="OK")
