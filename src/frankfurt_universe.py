"""
frankfurt_universe.py
=====================

Builds a screener-ready equity universe from the Frankfurt Stock Exchange.

Pipeline
--------
1. Download the latest "all tradable instruments" CSVs directly from
   Deutsche Boerse (Xetra + Boerse Frankfurt). The download function scrapes
   the live page for the current blob link, so it always grabs the newest file
   even though the URL hash changes on every update.
2. Parse the T7 reference-data CSV (semicolon-delimited, 2 metadata lines,
   ~140 columns) and keep only active shares (Instrument Type == "CS").
3. Merge the two universes with the scope you configure (Xetra = all shares;
   Frankfurt = US-only / all-foreign / all).
4. Resolve each ISIN to a Yahoo Finance ticker (cached to disk, with a
   mnemonic-based fallback).
5. Pull fundamentals from yfinance (cached to disk).
6. Filter on the fundamental thresholds in CONFIG.
7. Export a tidy CSV your screener can read.

The two data sources are separate because the CSVs carry NO fundamentals and
NO Yahoo tickers - only ISIN / WKN / Xetra mnemonic. Steps 4-6 add everything
your strategy needs.

Requires: requests, pandas, yfinance   (pip install requests pandas yfinance)

Scope and thresholds are externalized to config.yaml (loaded by
``src.universe_config``); both this CLI and the Dash Universe tab read them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # yfinance only needed for the fundamentals stage
    yf = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frankfurt_universe")


class JobCancelled(Exception):
    """Raised by the resolve / fundamentals loops when a stop is requested."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Live pages that host the "all tradable instruments" download links.
# The download step scrapes these for the current CSV blob URL.
XETRA_PAGE = (
    "https://www.cashmarket.deutsche-boerse.com/cash-en/trading/Tradable-Instruments-Xetra/xetra"
)
FRANKFURT_PAGE = "https://www.cashmarket.deutsche-boerse.com/cash-en/trading/Tradable-Instruments-Xetra/boersefrankfurt"

# Yahoo exchange codes used to pick the right listing during ISIN resolution.
US_YAHOO_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BATS"}
XETRA_YAHOO_EXCHANGE = "GER"  # -> ".DE" tickers
FRANKFURT_YAHOO_EXCHANGE = "FRA"  # -> ".F" tickers


@dataclass
class Config:
    """All tunables for the universe-builder pipeline (paths, scope, and fundamental thresholds).

    Loaded from ``config.yaml`` via ``src.universe_config.load_universe_config``;
    a threshold of ``None`` means that fundamental gate is skipped entirely.
    """

    # --- where files live -------------------------------------------------
    data_dir: Path = Path("./data")  # raw CSVs + caches
    out_path: Path = Path("./data/universe_screener.csv")  # CLI default output
    # Two distinct outputs for the Dash workflow: the downloaded universe and
    # the screened result. Parameterized via export(..., out_path=...).
    universe_out_path: Path = Path("./data/universe_downloaded.csv")
    universe_fundamentals_out_path: Path = Path("./data/universe_fundamentals.csv")
    screened_out_path: Path = Path("./data/universe_screened.csv")

    # --- universe scope ---------------------------------------------------
    include_xetra: bool = True
    include_frankfurt: bool = True
    # Frankfurt shares subset: "us" (ISIN starts US), "foreign" (non-DE), "all".
    frankfurt_scope: str = "us"
    share_type_codes: tuple[str, ...] = ("CS",)  # T7 Instrument Type for common shares
    active_only: bool = True

    # --- Yahoo ticker resolution -----------------------------------------
    # For US-domiciled names: "us_primary" -> AAPL (best liquidity/signal),
    # "frankfurt_line" -> AAPL.F / .DE (what a German broker actually trades),
    # "both" -> keep US primary but also record the Frankfurt line.
    us_ticker_mode: str = "us_primary"
    yahoo_pause_s: float = 0.3  # politeness delay between search calls
    yahoo_max_retries: int = 4

    # --- fundamentals (yfinance) -----------------------------------------
    fetch_fundamentals: bool = True
    yf_pause_s: float = 0.4

    # --- fundamental filter thresholds (None = ignore that gate) ----------
    min_market_cap: float | None = 300_000_000  # in the stock's own currency
    max_market_cap: float | None = None
    min_price: float | None = 2.0
    min_avg_volume: float | None = 100_000  # shares/day (yfinance averageVolume)
    min_turnover: float | None = 1_000_000
    min_current_ratio: float | None = 1.0
    max_debt_to_equity: float | None = 150.0
    require_positive_eps: bool = False
    min_trailing_pe: float | None = None
    max_trailing_pe: float | None = None
    max_price_to_book: float | None = None
    excluded_sectors: tuple[str, ...] = ()
    allowed_sectors: tuple[str, ...] = ()  # empty = allow all

    # --- misc -------------------------------------------------------------
    request_timeout_s: int = 120
    csv_encodings: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    def __post_init__(self):
        """Coerce path fields to ``Path`` and ensure ``data_dir`` exists."""
        self.data_dir = Path(self.data_dir)
        self.out_path = Path(self.out_path)
        self.universe_out_path = Path(self.universe_out_path)
        self.universe_fundamentals_out_path = Path(self.universe_fundamentals_out_path)
        self.screened_out_path = Path(self.screened_out_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Download the latest CSV
# ---------------------------------------------------------------------------

_BLOB_RE = re.compile(
    r"""["'](?P<url>(?:https?://[^"']+)?/resource/blob/\d+/[a-f0-9]+/data/"""
    r"""[^"']*allTradableInstruments\.csv)["']""",
    re.IGNORECASE,
)


def find_latest_csv_url(page_url: str, cfg: Config) -> str:
    """Scrape a Deutsche Boerse page for the current instrument-CSV blob URL."""
    log.info("Locating latest CSV link on %s", page_url)
    r = requests.get(
        page_url, headers={"User-Agent": cfg.user_agent}, timeout=cfg.request_timeout_s
    )
    r.raise_for_status()
    m = _BLOB_RE.search(r.text)
    if not m:
        raise RuntimeError(
            f"Could not find an allTradableInstruments.csv link on {page_url}. "
            "The page layout may have changed - inspect it and update _BLOB_RE."
        )
    return urljoin(page_url, m.group("url"))


def download_file(
    url: str, dest: Path, cfg: Config, should_stop: Callable[[], bool] | None = None
) -> Path:
    """Stream a (potentially huge) file to disk in chunks."""
    log.info("Downloading %s", url)
    with requests.get(
        url,
        headers={"User-Agent": cfg.user_agent},
        stream=True,
        timeout=cfg.request_timeout_s,
    ) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        size = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                if should_stop is not None and should_stop():
                    raise JobCancelled("download stopped")
                if chunk:
                    fh.write(chunk)
                    size += len(chunk)
        tmp.replace(dest)
    log.info("Saved %s (%.1f MB)", dest.name, size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# 2. Parse the T7 reference CSV
# ---------------------------------------------------------------------------

# Raw T7 column name -> normalized name we keep.
COLUMN_MAP = {
    "Instrument": "name",
    "ISIN": "isin",
    "WKN": "wkn",
    "Mnemonic": "mnemonic",
    "MIC Code": "mic",
    "Primary Market MIC Code": "primary_market_mic",
    "Currency": "currency",
    "Instrument Type": "instrument_type",
    "Instrument Status": "status",
}


def _detect_header_row(path: Path, encoding: str) -> int:
    """Return the 0-based index of the real header line (starts 'Product Status')."""
    with open(path, encoding=encoding, errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.lstrip().lower().startswith("product status"):
                return i
            if i > 15:
                break
    raise RuntimeError(f"Header row not found in {path}")


def parse_t7_csv(
    path: Path, cfg: Config, should_stop: Callable[[], bool] | None = None
) -> pd.DataFrame:
    """Stream-parse a T7 instrument CSV, keeping only active shares.

    Handles the 1.6M-row Frankfurt file by reading in chunks and filtering
    each chunk immediately, so memory stays flat.
    """
    last_err: Exception | None = None
    for enc in cfg.csv_encodings:
        try:
            header_row = _detect_header_row(path, enc)
            log.info("Parsing %s (encoding=%s, header at line %d)", path.name, enc, header_row)
            keep_frames: list[pd.DataFrame] = []
            reader = pd.read_csv(
                path,
                sep=";",
                skiprows=header_row,
                dtype=str,
                encoding=enc,
                chunksize=200_000,
                on_bad_lines="skip",
                low_memory=False,
            )
            wanted_raw = [c for c in COLUMN_MAP if c]  # raw names we need
            for chunk in reader:
                if should_stop is not None and should_stop():
                    raise JobCancelled("parse stopped")
                chunk.columns = [c.strip() for c in chunk.columns]
                missing = [c for c in wanted_raw if c not in chunk.columns]
                if missing:
                    raise KeyError(f"Missing expected columns: {missing}")
                sub = chunk[wanted_raw].rename(columns=COLUMN_MAP)
                sub = sub[sub["instrument_type"].isin(cfg.share_type_codes)]
                if cfg.active_only:
                    sub = sub[sub["status"].str.strip().str.lower() == "active"]
                if not sub.empty:
                    keep_frames.append(sub)
            df = (
                pd.concat(keep_frames, ignore_index=True)
                if keep_frames
                else pd.DataFrame(columns=list(COLUMN_MAP.values()))
            )
            for col in (
                "isin",
                "wkn",
                "mnemonic",
                "mic",
                "primary_market_mic",
                "currency",
                "name",
            ):
                df[col] = df[col].astype(str).str.strip()
            df = df.drop_duplicates(subset="isin")
            log.info("  -> %d active shares", len(df))
            return df
        except (UnicodeDecodeError, KeyError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to parse {path}: {last_err}")


# ---------------------------------------------------------------------------
# 3. Build the merged universe
# ---------------------------------------------------------------------------


def _scope_frankfurt(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Filter Frankfurt-listed shares by ISIN scope: "all", "foreign" (non-DE), or "us"."""
    if scope == "all":
        return df
    if scope == "foreign":
        return df[~df["isin"].str.startswith("DE")]
    if scope == "us":
        return df[df["isin"].str.startswith("US")]
    raise ValueError(f"Unknown frankfurt_scope: {scope!r}")


def build_universe(
    cfg: Config, refresh: bool = True, should_stop: Callable[[], bool] | None = None
) -> pd.DataFrame:
    """Download (if ``refresh``) and merge the Xetra + Frankfurt instrument CSVs into one universe.

    On duplicate ISIN across the two sources, the Xetra listing wins. Raises
    if neither ``cfg.include_xetra`` nor ``cfg.include_frankfurt`` is enabled.
    """
    frames: list[pd.DataFrame] = []

    if cfg.include_xetra:
        xetra_csv = cfg.data_dir / "xetra_all.csv"
        if refresh or not xetra_csv.exists():
            download_file(
                find_latest_csv_url(XETRA_PAGE, cfg), xetra_csv, cfg, should_stop=should_stop
            )
        xe = parse_t7_csv(xetra_csv, cfg, should_stop=should_stop)
        xe["source"] = "XETR"
        frames.append(xe)

    if cfg.include_frankfurt:
        fra_csv = cfg.data_dir / "frankfurt_all.csv"
        if refresh or not fra_csv.exists():
            download_file(
                find_latest_csv_url(FRANKFURT_PAGE, cfg), fra_csv, cfg, should_stop=should_stop
            )
        fr = parse_t7_csv(fra_csv, cfg, should_stop=should_stop)
        fr = _scope_frankfurt(fr, cfg.frankfurt_scope)
        fr["source"] = "XFRA"
        log.info("Frankfurt scope=%s -> %d shares", cfg.frankfurt_scope, len(fr))
        frames.append(fr)

    if not frames:
        raise RuntimeError("No sources enabled.")

    universe = pd.concat(frames, ignore_index=True)
    # On duplicate ISIN, prefer the Xetra listing.
    universe["_prio"] = (universe["source"] == "XETR").astype(int)
    universe = (
        universe.sort_values("_prio", ascending=False)
        .drop_duplicates(subset="isin", keep="first")
        .drop(columns="_prio")
        .reset_index(drop=True)
    )
    log.info("Merged universe: %d unique shares", len(universe))
    return universe


# ---------------------------------------------------------------------------
# 4. Resolve ISIN -> Yahoo ticker (cached)
# ---------------------------------------------------------------------------


class YahooResolver:
    """Resolve ISIN -> Yahoo symbol via Yahoo's search endpoint, with a
    disk cache and a mnemonic-based fallback."""

    SEARCH_URLS = (
        "https://query2.finance.yahoo.com/v1/finance/search",
        "https://query1.finance.yahoo.com/v1/finance/search",
    )

    def __init__(self, cfg: Config):
        """Load the ISIN->ticker disk cache (if present) and set up a shared requests session."""
        self.cfg = cfg
        self.cache_path = cfg.data_dir / "isin_to_yahoo.json"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent

    def _save(self):
        """Persist the ISIN->ticker cache to disk."""
        self.cache_path.write_text(json.dumps(self.cache, indent=0))

    def _search(self, isin: str) -> list[dict]:
        """Query Yahoo's search endpoint for an ISIN, retrying with backoff on 429/errors."""
        for url in self.SEARCH_URLS:
            for attempt in range(self.cfg.yahoo_max_retries):
                try:
                    r = self.session.get(
                        url,
                        params={"q": isin, "quotesCount": 10, "newsCount": 0},
                        timeout=self.cfg.request_timeout_s,
                    )
                    if r.status_code == 429:
                        time.sleep(2**attempt)
                        continue
                    r.raise_for_status()
                    return r.json().get("quotes", [])
                except requests.RequestException:
                    time.sleep(2**attempt)
        return []

    def _pick(self, isin: str, quotes: list[dict]) -> tuple[str | None, str | None]:
        """Return (chosen_ticker, frankfurt_line_ticker)."""
        equities = [q for q in quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
        if not equities:
            return None, None

        def by_exchange(codes):
            """First equity quote symbol whose exchange code is in ``codes``, else None."""
            for q in equities:
                if q.get("exchange") in codes:
                    return q["symbol"]
            return None

        fra_line = by_exchange({FRANKFURT_YAHOO_EXCHANGE}) or by_exchange({XETRA_YAHOO_EXCHANGE})

        if isin.startswith("US"):
            us_sym = by_exchange(US_YAHOO_EXCHANGES)
            mode = self.cfg.us_ticker_mode
            if mode == "frankfurt_line":
                return (fra_line or us_sym or equities[0]["symbol"]), fra_line
            # us_primary or both
            return (us_sym or equities[0]["symbol"]), fra_line

        # Non-US: prefer the German listing, else the first equity.
        return (fra_line or equities[0]["symbol"]), fra_line

    def resolve(self, isin: str, mnemonic: str, source: str) -> tuple[str | None, str | None]:
        """Resolve one ISIN to ``(ticker, frankfurt_line_ticker)``, using and updating the disk cache.

        Falls back to a guessed ``{mnemonic}.DE``/``.F`` ticker (flagged
        ``guessed: True`` in the cache) when Yahoo's search returns no match.
        """
        if isin in self.cache:
            c = self.cache[isin]
            return c.get("ticker"), c.get("fra_line")

        quotes = self._search(isin)
        ticker, fra_line = self._pick(isin, quotes)

        if ticker is None and mnemonic:
            # Fallback: construct from the Xetra mnemonic. Unvalidated, so flag it.
            suffix = ".DE" if source == "XETR" else ".F"
            ticker = f"{mnemonic}{suffix}"
            fra_line = ticker
            self.cache[isin] = {"ticker": ticker, "fra_line": fra_line, "guessed": True}
        else:
            self.cache[isin] = {"ticker": ticker, "fra_line": fra_line, "guessed": False}

        self._save()
        time.sleep(self.cfg.yahoo_pause_s)
        return ticker, fra_line


def resolve_tickers(
    universe: pd.DataFrame,
    cfg: Config,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Resolve every row's ISIN to a Yahoo ticker via ``YahooResolver``, adding
    ``yahoo_ticker``/``frankfurt_line`` columns. Reports progress every 25 rows
    and raises ``JobCancelled`` if ``should_stop`` fires mid-loop.
    """
    resolver = YahooResolver(cfg)
    tickers, fra_lines = [], []
    n = len(universe)
    for i, row in enumerate(universe.itertuples(index=False), 1):
        if should_stop is not None and should_stop():
            raise JobCancelled(f"resolve stopped at {i}/{n}")
        t, f = resolver.resolve(row.isin, row.mnemonic, row.source)
        tickers.append(t)
        fra_lines.append(f)
        if i % 100 == 0:
            log.info("  resolved %d/%d tickers", i, n)
        if progress is not None and i % 25 == 0:
            progress(i, n)
    universe = universe.copy()
    universe["yahoo_ticker"] = tickers
    universe["frankfurt_line"] = fra_lines
    unresolved = universe["yahoo_ticker"].isna().sum()
    log.info("Ticker resolution done. Unresolved: %d", unresolved)
    if progress is not None:
        progress(n, n)
    return universe


# ---------------------------------------------------------------------------
# 5. Fundamentals (yfinance, cached)
# ---------------------------------------------------------------------------

_FUND_FIELDS = (
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "dividendYield",
    "trailingEps",
    "profitMargins",
    "returnOnEquity",
    "sector",
    "industry",
    "averageVolume",
    "currentPrice",
    "regularMarketPrice",
    "currency",
    "longName",
    "currentRatio",
    "debtToEquity",
)


def fetch_fundamentals(
    universe: pd.DataFrame,
    cfg: Config,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Fetch and merge yfinance fundamentals (``_FUND_FIELDS``) for each unique ``yahoo_ticker``.

    Caches per-ticker results to ``fundamentals.json`` (flushed every 25
    tickers) so a cancelled/resumed run doesn't re-fetch already-seen tickers.
    """
    if yf is None:
        raise ImportError("yfinance not installed. pip install yfinance")

    cache_path = cfg.data_dir / "fundamentals.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    rows: list[dict] = []
    tickers = list(universe["yahoo_ticker"].dropna().unique())
    n = len(tickers)
    for i, tkr in enumerate(tickers, 1):
        if should_stop is not None and should_stop():
            raise JobCancelled(f"fundamentals stopped at {i}/{n}")
        if progress is not None and i % 10 == 0:
            progress(i, n)
        if tkr not in cache:
            info = {}
            try:
                info = yf.Ticker(tkr).get_info() or {}
            except Exception as e:  # noqa: BLE001 - yfinance raises many things
                log.debug("info failed for %s: %s", tkr, e)
            cache[tkr] = {k: info.get(k) for k in _FUND_FIELDS}
            if i % 25 == 0:
                cache_path.write_text(json.dumps(cache))
                log.info("  fundamentals %d/%d", i, n)
            time.sleep(cfg.yf_pause_s)
        rec = dict(cache[tkr])
        rec["yahoo_ticker"] = tkr
        rec["price"] = rec.get("currentPrice") or rec.get("regularMarketPrice")
        rows.append(rec)

    cache_path.write_text(json.dumps(cache))
    fund = pd.DataFrame(rows)
    merged = universe.merge(fund, on="yahoo_ticker", how="left", suffixes=("", "_yf"))
    log.info("Fundamentals fetched for %d tickers", n)
    return merged


# ---------------------------------------------------------------------------
# 6. Fundamental filter
# ---------------------------------------------------------------------------


def apply_filter(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the fundamental screener thresholds from ``cfg`` (see ADR 0005), AND-ing all enabled gates."""
    before = len(df)
    m = pd.Series(True, index=df.index)

    def num(col):
        """Coerce a fundamentals column to numeric, turning bad/missing values into NaN."""
        return pd.to_numeric(df.get(col), errors="coerce")

    if cfg.min_market_cap is not None:
        m &= num("marketCap") >= cfg.min_market_cap
    if cfg.max_market_cap is not None:
        m &= num("marketCap") <= cfg.max_market_cap
    if cfg.min_price is not None:
        m &= num("price") >= cfg.min_price
    if cfg.min_avg_volume is not None:
        m &= num("averageVolume") >= cfg.min_avg_volume
    if cfg.min_turnover is not None:
        m &= (num("price") * num("averageVolume")) >= cfg.min_turnover
    if cfg.min_current_ratio is not None:
        m &= num("currentRatio") >= cfg.min_current_ratio
    if cfg.max_debt_to_equity is not None:
        m &= num("debtToEquity") <= cfg.max_debt_to_equity
    if cfg.require_positive_eps:
        m &= num("trailingEps") > 0
    if cfg.min_trailing_pe is not None:
        m &= num("trailingPE") >= cfg.min_trailing_pe
    if cfg.max_trailing_pe is not None:
        m &= num("trailingPE") <= cfg.max_trailing_pe
    if cfg.max_price_to_book is not None:
        m &= num("priceToBook") <= cfg.max_price_to_book
    if cfg.excluded_sectors:
        m &= ~df.get("sector").isin(cfg.excluded_sectors)
    if cfg.allowed_sectors:
        m &= df.get("sector").isin(cfg.allowed_sectors)

    out = df[m].copy()
    log.info("Filter: %d -> %d passed", before, len(out))
    return out


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

EXPORT_COLUMNS = [
    "yahoo_ticker",
    "isin",
    "wkn",
    "name",
    "source",
    "mic",
    "primary_market_mic",
    "currency",
    "sector",
    "industry",
    "marketCap",
    "price",
    "averageVolume",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "dividendYield",
    "currentRatio",
    "debtToEquity",
    "frankfurt_line",
]


def export(df: pd.DataFrame, cfg: Config, out_path: Path | str | None = None) -> Path:
    """Write the tidy CSV. ``out_path`` overrides ``cfg.out_path`` so the Dash
    workflow can target distinct downloaded / screened files."""
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    out = df[cols].rename(
        columns={
            "marketCap": "market_cap",
            "averageVolume": "avg_volume",
            "trailingPE": "trailing_pe",
            "forwardPE": "forward_pe",
            "priceToBook": "price_to_book",
            "dividendYield": "dividend_yield",
        }
    )
    out["last_updated"] = pd.Timestamp.utcnow().isoformat()
    target = Path(out_path) if out_path is not None else cfg.out_path
    out.to_csv(target, index=False)
    log.info("Wrote %d rows -> %s", len(out), target)
    return target


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(cfg: Config, refresh: bool = True) -> pd.DataFrame:
    """CLI/library entry point: build, resolve, (optionally) screen, and export the universe."""
    universe = build_universe(cfg, refresh=refresh)
    universe = resolve_tickers(universe, cfg)
    if cfg.fetch_fundamentals:
        universe = fetch_fundamentals(universe, cfg)
        universe = apply_filter(universe, cfg)
    export(universe, cfg)
    return universe


if __name__ == "__main__":
    import argparse

    try:
        from src.universe_config import load_universe_config

        config = load_universe_config()
    except Exception:  # pragma: no cover - fall back to built-in defaults
        config = Config()

    p = argparse.ArgumentParser(description="Build a Frankfurt/Xetra screener universe.")
    p.add_argument(
        "--no-refresh",
        action="store_true",
        help="reuse already-downloaded CSVs instead of fetching fresh ones",
    )
    p.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="stop after building + resolving the universe (skip yfinance)",
    )
    p.add_argument("--frankfurt-scope", choices=["us", "foreign", "all"])
    p.add_argument("--us-ticker-mode", choices=["us_primary", "frankfurt_line", "both"])
    p.add_argument("--out")
    args = p.parse_args()

    if args.frankfurt_scope:
        config.frankfurt_scope = args.frankfurt_scope
    if args.us_ticker_mode:
        config.us_ticker_mode = args.us_ticker_mode
    if args.no_fundamentals:
        config.fetch_fundamentals = False
    if args.out:
        config.out_path = Path(args.out)

    run(config, refresh=not args.no_refresh)
