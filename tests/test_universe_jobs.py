"""Offline tests for the background job runner (network functions monkeypatched)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src import frankfurt_universe as fu
from src import universe_jobs as jobs

_TERMINAL = ("done", "error", "stopped")


def _cfg(tmp_path):
    return fu.Config(
        data_dir=tmp_path,
        universe_out_path=tmp_path / "downloaded.csv",
        screened_out_path=tmp_path / "screened.csv",
    )


def _wait_terminal(cfg, job, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jobs.job_status(cfg, job)
        if status["state"] in _TERMINAL:
            return status
        time.sleep(0.02)
    return jobs.job_status(cfg, job)


def _fake_export(df, cfg, out_path=None):
    Path(out_path).write_text("ok")
    return out_path


def test_idle_status_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert jobs.job_status(cfg, jobs.SCREEN)["state"] == "idle"


def test_download_job_success_reports_percent(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    universe = pd.DataFrame({"isin": ["US1"], "mnemonic": ["A"], "source": ["XETR"]})

    monkeypatch.setattr(fu, "build_universe", lambda cfg, refresh=True, **kw: universe)
    monkeypatch.setattr(fu, "resolve_tickers", lambda u, cfg, **kw: u.assign(yahoo_ticker=["AAPL"]))
    monkeypatch.setattr(fu, "export", _fake_export)

    assert jobs.start_download_job(cfg) is True
    status = _wait_terminal(cfg, jobs.DOWNLOAD)
    assert status["state"] == "done"
    assert status["percent"] == 100
    assert Path(status["output_path"]).exists()
    assert status["output_path"] == str(cfg.universe_out_path)


def test_screen_job_success(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"yahoo_ticker": ["AAPL"], "isin": ["US1"], "name": ["Apple"]}).to_csv(
        input_csv, index=False
    )

    monkeypatch.setattr(fu, "fetch_fundamentals", lambda df, cfg, **kw: df.assign(price=[100.0]))
    monkeypatch.setattr(fu, "apply_filter", lambda df, cfg: df)
    monkeypatch.setattr(fu, "export", _fake_export)

    assert jobs.start_screen_job(cfg, input_csv) is True
    status = _wait_terminal(cfg, jobs.SCREEN)
    assert status["state"] == "done"
    assert Path(status["output_path"]).exists()


def test_download_job_error_is_captured(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def boom(cfg, refresh=True, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(fu, "build_universe", boom)

    assert jobs.start_download_job(cfg) is True
    status = _wait_terminal(cfg, jobs.DOWNLOAD)
    assert status["state"] == "error"
    assert "network down" in status["message"]


def test_stop_job_cancels_and_marks_stopped(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    universe = pd.DataFrame({"isin": ["US1"], "mnemonic": ["A"], "source": ["XETR"]})

    def slow_resolve(u, cfg, progress=None, should_stop=None):
        for _ in range(500):
            if should_stop is not None and should_stop():
                raise fu.JobCancelled("stopped in test")
            time.sleep(0.01)
        return u.assign(yahoo_ticker=["X"] * len(u))

    monkeypatch.setattr(fu, "build_universe", lambda cfg, refresh=True, **kw: universe)
    monkeypatch.setattr(fu, "resolve_tickers", slow_resolve)
    monkeypatch.setattr(fu, "export", _fake_export)

    assert jobs.start_download_job(cfg) is True
    time.sleep(0.1)  # let it enter the resolve loop
    assert jobs.is_running(jobs.DOWNLOAD) is True
    jobs.stop_job(cfg, jobs.DOWNLOAD)

    status = _wait_terminal(cfg, jobs.DOWNLOAD)
    assert status["state"] == "stopped"
    assert jobs.is_running(jobs.DOWNLOAD) is False


def test_read_log_returns_recent_lines(tmp_path):
    cfg = _cfg(tmp_path)
    jobs._reset_log(cfg, jobs.DOWNLOAD)
    for i in range(3):
        jobs._log(cfg, jobs.DOWNLOAD, f"line {i}")
    out = jobs.read_log(cfg, jobs.DOWNLOAD)
    assert "line 0" in out and "line 2" in out
