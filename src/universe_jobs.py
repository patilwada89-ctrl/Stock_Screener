"""Background jobs for the two-step universe workflow (download / screen).

Each job runs in a daemon thread and reports to disk so a Dash ``dcc.Interval``
can poll it without blocking the server:

- ``data/job_status.json`` — per-job ``state``, ``percent``, ``message``, ``output_path``.
- ``data/<job>.log`` — a live, human-readable feed (the reused module's log lines
  plus high-level step messages), tailed by the UI like a terminal.

Jobs are cooperatively cancellable: ``stop_job`` sets an Event that the resolve /
fundamentals loops check each iteration (raising ``JobCancelled``). The underlying
module's disk caches (``isin_to_yahoo.json``, ``fundamentals.json``) are preserved,
so a stopped job resumes cheaply.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import frankfurt_universe as fu

DOWNLOAD = "download"
SCREEN = "screen"
SWING = "swing"
_STATUS_FILENAME = "job_status.json"

_threads: dict[str, threading.Thread] = {}
_cancel: dict[str, threading.Event] = {
    DOWNLOAD: threading.Event(),
    SCREEN: threading.Event(),
    SWING: threading.Event(),
}
_start_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Status + log files
# ---------------------------------------------------------------------------


def _status_path(cfg: fu.Config) -> Path:
    """Path to the shared ``job_status.json`` file for this config's ``data_dir``."""
    return Path(cfg.data_dir) / _STATUS_FILENAME


def _log_path(cfg: fu.Config, job: str) -> Path:
    """Path to the live log file (``<job>.log``) for a given job name."""
    return Path(cfg.data_dir) / f"{job}.log"


def _now() -> str:
    """Current UTC timestamp as an ISO 8601 string, for status/log entries."""
    return datetime.now(timezone.utc).isoformat()


def read_status(cfg: fu.Config) -> dict[str, Any]:
    """Read the full job-status JSON (all jobs), or ``{}`` if missing/corrupt."""
    path = _status_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def is_running(job: str) -> bool:
    """True only if this process has a live thread for ``job`` (source of truth)."""
    thread = _threads.get(job)
    return thread is not None and thread.is_alive()


def job_status(cfg: fu.Config, job: str) -> dict[str, Any]:
    """Current status for one job, reconciling a stale on-disk "running" state
    (e.g. after a server restart) to "interrupted" using ``is_running`` as the
    source of truth.
    """
    default = {"state": "idle", "message": "", "percent": 0, "output_path": None}
    status = {**default, **read_status(cfg).get(job, {})}
    # Reconcile a stale "running" left by a crashed/restarted process.
    if status["state"] == "running" and not is_running(job):
        status["state"] = "interrupted"
        status["message"] = "Interrupted (server restarted before it finished)."
    return status


def _set_status(
    cfg: fu.Config,
    job: str,
    *,
    state: str | None = None,
    message: str | None = None,
    percent: int | None = None,
    output_path: Path | str | None = None,
    set_output: bool = False,
) -> None:
    """Merge the given fields into ``job``'s status entry and write it back atomically."""
    path = _status_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_status(cfg)
    entry = dict(data.get(job, {}))
    if state is not None:
        entry["state"] = state
    if message is not None:
        entry["message"] = message
    if percent is not None:
        entry["percent"] = int(percent)
    if set_output:
        entry["output_path"] = str(output_path) if output_path else None
    entry["updated_at"] = _now()
    data[job] = entry
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic


def _reset_log(cfg: fu.Config, job: str) -> None:
    """Truncate ``job``'s log file at the start of a fresh run."""
    path = _log_path(cfg, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _log(cfg: fu.Config, job: str, msg: str) -> None:
    """Append one timestamped line to ``job``'s log file."""
    path = _log_path(cfg, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H:%M:%S")
    with open(path, "a") as fh:
        fh.write(f"{stamp}  {msg}\n")


def read_log(cfg: fu.Config, job: str, max_lines: int = 40) -> str:
    """Tail the last ``max_lines`` of ``job``'s log file, for the Dash terminal view."""
    path = _log_path(cfg, job)
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    return "\n".join(lines[-max_lines:])


class _JobLogHandler(logging.Handler):
    """Mirror the reused module's log records into the job's live log file."""

    def __init__(self, cfg: fu.Config, job: str):
        """Bind this handler to a specific job's log file."""
        super().__init__(level=logging.INFO)
        self.cfg = cfg
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        """Write the record's message to the job log; swallows errors so logging can't crash the job."""
        try:
            _log(self.cfg, self.job, record.getMessage())
        except Exception:  # noqa: BLE001 - logging must never crash the job
            pass


def mark_interrupted(cfg: fu.Config) -> None:
    """Persist any stale 'running' status as 'interrupted' (call on app startup)."""
    data = read_status(cfg)
    changed = False
    for job in (DOWNLOAD, SCREEN, SWING):
        if data.get(job, {}).get("state") == "running" and not is_running(job):
            data[job]["state"] = "interrupted"
            data[job]["message"] = "Interrupted (server restarted before it finished)."
            data[job]["updated_at"] = _now()
            changed = True
    if changed:
        path = _status_path(cfg)
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _run_download(cfg: fu.Config) -> None:
    """Thread target for the download job: build the universe and resolve tickers, reporting progress."""
    handler = _JobLogHandler(cfg, DOWNLOAD)
    fu.log.addHandler(handler)
    try:
        _reset_log(cfg, DOWNLOAD)
        _set_status(
            cfg,
            DOWNLOAD,
            state="running",
            message="Downloading exchange CSVs…",
            percent=2,
            output_path=None,
            set_output=True,
        )

        def should_stop() -> bool:
            """True once ``stop_job(DOWNLOAD)`` has requested cancellation."""
            return _cancel[DOWNLOAD].is_set()

        _log(cfg, DOWNLOAD, "Starting universe download…")
        universe = fu.build_universe(cfg, refresh=True, should_stop=should_stop)
        n_total = len(universe)
        _set_status(cfg, DOWNLOAD, message=f"Resolving {n_total} tickers…", percent=15)

        def progress(i: int, n: int) -> None:
            """Update DOWNLOAD status percent (mapped to the 15-95% ticker-resolution range)."""
            _set_status(
                cfg,
                DOWNLOAD,
                message=f"Resolving {i}/{n} tickers",
                percent=15 + int(80 * i / max(1, n)),
            )

        universe = fu.resolve_tickers(universe, cfg, progress=progress, should_stop=should_stop)
        out = fu.export(universe, cfg, out_path=cfg.universe_out_path)
        _set_status(
            cfg,
            DOWNLOAD,
            state="done",
            message=f"Universe ready: {len(universe)} rows.",
            percent=100,
            output_path=out,
            set_output=True,
        )
        _log(cfg, DOWNLOAD, f"Done. Wrote {len(universe)} rows to {out}")
    except fu.JobCancelled as exc:
        _set_status(cfg, DOWNLOAD, state="stopped", message=f"Stopped ({exc}). Cache preserved.")
        _log(cfg, DOWNLOAD, f"Stopped by user ({exc}).")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _set_status(cfg, DOWNLOAD, state="error", message=f"{type(exc).__name__}: {exc}")
        _log(cfg, DOWNLOAD, f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        fu.log.removeHandler(handler)


def _run_screen(cfg: fu.Config, input_csv: str | Path) -> None:
    """Thread target for the screen job: resolve tickers (if needed), fetch fundamentals, filter, export."""
    handler = _JobLogHandler(cfg, SCREEN)
    fu.log.addHandler(handler)
    try:
        _reset_log(cfg, SCREEN)
        _set_status(
            cfg,
            SCREEN,
            state="running",
            message=f"Loading {Path(input_csv).name}…",
            percent=3,
            output_path=None,
            set_output=True,
        )
        df = pd.read_csv(input_csv)

        def should_stop() -> bool:
            """True once ``stop_job(SCREEN)`` has requested cancellation."""
            return _cancel[SCREEN].is_set()

        if "yahoo_ticker" not in df.columns or df["yahoo_ticker"].isna().all():
            if "isin" not in df.columns:
                raise ValueError("Loaded CSV has neither 'yahoo_ticker' nor 'isin' columns.")
            for col in ("mnemonic", "source"):
                if col not in df.columns:
                    df[col] = ""
            _set_status(cfg, SCREEN, message="Resolving ISIN → Yahoo tickers…", percent=8)

            def res_progress(i: int, n: int) -> None:
                """Update SCREEN status percent during the ISIN-resolution sub-step (8-28%)."""
                _set_status(
                    cfg,
                    SCREEN,
                    message=f"Resolving {i}/{n} tickers",
                    percent=8 + int(20 * i / max(1, n)),
                )

            df = fu.resolve_tickers(df, cfg, progress=res_progress, should_stop=should_stop)

        _set_status(cfg, SCREEN, message="Fetching fundamentals…", percent=30)

        def fund_progress(i: int, n: int) -> None:
            """Update SCREEN status percent during the fundamentals-fetch sub-step (30-95%)."""
            _set_status(
                cfg, SCREEN, message=f"Fundamentals {i}/{n}", percent=30 + int(65 * i / max(1, n))
            )

        df = fu.fetch_fundamentals(df, cfg, progress=fund_progress, should_stop=should_stop)

        if getattr(cfg, "universe_fundamentals_out_path", None):
            fund_out = fu.export(df, cfg, out_path=cfg.universe_fundamentals_out_path)
            _log(cfg, SCREEN, f"Saved full fundamentals universe to {fund_out}")

        df = fu.apply_filter(df, cfg)
        out = fu.export(df, cfg, out_path=cfg.screened_out_path)
        _set_status(
            cfg,
            SCREEN,
            state="done",
            message=f"Screened set: {len(df)} rows.",
            percent=100,
            output_path=out,
            set_output=True,
        )
        _log(cfg, SCREEN, f"Done. Wrote {len(df)} rows to {out}")
    except fu.JobCancelled as exc:
        _set_status(cfg, SCREEN, state="stopped", message=f"Stopped ({exc}). Cache preserved.")
        _log(cfg, SCREEN, f"Stopped by user ({exc}).")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _set_status(cfg, SCREEN, state="error", message=f"{type(exc).__name__}: {exc}")
        _log(cfg, SCREEN, f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        fu.log.removeHandler(handler)


def _start(cfg: fu.Config, job: str, target, args: tuple) -> bool:
    """Spawn ``target`` as a daemon thread for ``job`` if it isn't already running.

    Guarded by ``_start_lock`` so two near-simultaneous Dash callbacks can't
    both pass the ``is_running`` check and start duplicate threads.
    """
    with _start_lock:
        if is_running(job):
            return False
        _cancel[job].clear()
        thread = threading.Thread(target=target, args=args, daemon=True)
        _threads[job] = thread
        thread.start()
    return True


def start_download_job(cfg: fu.Config) -> bool:
    """Spawn the download job. Returns False if it is already running."""
    return _start(cfg, DOWNLOAD, _run_download, (cfg,))


def start_screen_job(cfg: fu.Config, input_csv: str | Path) -> bool:
    """Spawn the screen job over ``input_csv``. Returns False if already running."""
    return _start(cfg, SCREEN, _run_screen, (cfg, input_csv))


def stop_job(cfg: fu.Config, job: str) -> None:
    """Request cooperative cancellation of ``job``."""
    _cancel[job].set()
    if is_running(job):
        _set_status(cfg, job, message="Stop requested — finishing current step…")


# ---------------------------------------------------------------------------
# Swing background job
# ---------------------------------------------------------------------------

import pickle  # noqa: E402 — grouped with job helpers intentionally


def swing_result_path(cfg: fu.Config) -> Path:
    """Path where the most recent swing result dict is cached."""
    return Path(cfg.data_dir) / "swing_result.pkl"


def _run_swing_job(
    cfg: fu.Config,
    compute_fn,
    active_path: str | None,
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    """Background thread target that runs the swing screener via *compute_fn*.

    *compute_fn* is ``app.compute_swing`` — passed as a callable to avoid a
    circular import from ``src/`` back into ``app``.
    """
    try:
        _reset_log(cfg, SWING)
        _set_status(
            cfg,
            SWING,
            state="running",
            message="Starting swing screener…",
            percent=5,
            output_path=None,
            set_output=True,
        )

        def should_stop() -> bool:
            """True once ``stop_job(SWING)`` has requested cancellation."""
            return _cancel[SWING].is_set()

        def progress(i: int, n: int) -> None:
            """Update SWING status percent as stocks are evaluated (5-95%)."""
            _set_status(
                cfg,
                SWING,
                message=f"Evaluating {i}/{n} stocks",
                percent=5 + int(90 * i / max(1, n)),
            )

        _log(cfg, SWING, "Starting swing screener…")
        result = compute_fn(
            active_path,
            buy_threshold,
            sell_threshold,
            should_stop=should_stop,
            progress=progress,
        )
        # Persist the result dict so the polling callback can render it.
        out = swing_result_path(cfg)
        out.write_bytes(pickle.dumps(result))
        n_qualified = len(result.get("qualified", []))
        _set_status(
            cfg,
            SWING,
            state="done",
            message=f"Done — {n_qualified} qualified stocks.",
            percent=100,
            output_path=str(out),
            set_output=True,
        )
        _log(cfg, SWING, f"Done. {n_qualified} qualified, result cached.")
    except fu.JobCancelled as exc:
        _set_status(cfg, SWING, state="stopped", message=f"Stopped ({exc}).")
        _log(cfg, SWING, f"Stopped by user ({exc}).")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _set_status(cfg, SWING, state="error", message=f"{type(exc).__name__}: {exc}")
        _log(cfg, SWING, f"ERROR: {type(exc).__name__}: {exc}")


def start_swing_job(
    cfg: fu.Config,
    compute_fn,
    active_path: str | None,
    buy_threshold: float,
    sell_threshold: float,
) -> bool:
    """Spawn the swing screener job. Returns False if already running."""
    return _start(
        cfg,
        SWING,
        _run_swing_job,
        (cfg, compute_fn, active_path, buy_threshold, sell_threshold),
    )
