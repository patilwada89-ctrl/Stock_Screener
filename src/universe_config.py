"""Load the Frankfurt universe :class:`Config` from ``config.yaml``.

Shared by the CLI (``python -m src.frankfurt_universe``) and the Dash Universe
tab so scope and thresholds have a single source of truth on disk.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.frankfurt_universe import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# Only these YAML keys are applied to Config; anything else is ignored.
_ALLOWED_KEYS = {
    "data_dir",
    "out_path",
    "universe_out_path",
    "universe_fundamentals_out_path",
    "screened_out_path",
    "include_xetra",
    "include_frankfurt",
    "frankfurt_scope",
    "us_ticker_mode",
    "fetch_fundamentals",
    "min_market_cap",
    "max_market_cap",
    "min_price",
    "min_avg_volume",
    "min_turnover",
    "min_current_ratio",
    "max_debt_to_equity",
    "require_positive_eps",
    "min_trailing_pe",
    "max_trailing_pe",
    "max_price_to_book",
    "excluded_sectors",
    "allowed_sectors",
}
_TUPLE_KEYS = ("excluded_sectors", "allowed_sectors")


def load_universe_config(path: str | Path | None = None) -> Config:
    """Build a :class:`Config` from ``config.yaml`` (missing file -> defaults)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}

    kwargs = {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
    for key in _TUPLE_KEYS:
        if kwargs.get(key) is not None:
            kwargs[key] = tuple(kwargs[key])
    return Config(**kwargs)
