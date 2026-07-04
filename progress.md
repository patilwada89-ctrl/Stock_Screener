# Progress — Dash Migration + Universe Builder

_Last updated: 2026-07-04 (later same day), branch `feat/dash-migration-universe`_

## What we've accomplished

- **Migrated the app from Streamlit to Dash + Plotly.** `app.py` is now a
  Dash app (`dbc.Container`, `dcc.Tabs`, `dash_table.DataTable`) with four
  fully working tabs: **Universe**, **Swing**, **Portfolio**, **Stock Details**.
  All charting goes through the new `src/plotly_charts.py` (pure functions,
  no Dash imports) instead of the old Altair/Streamlit chart helpers.
- **Universe builder tab is functional end-to-end.** Two-step background job
  workflow (download → screen) runs in daemon threads
  (`src/universe_jobs.py`), reporting progress/logs to disk
  (`data/job_status.json`, `data/<job>.log`) so the Dash `dcc.Interval`
  callbacks can poll without blocking the server. Jobs are cooperatively
  cancellable.
- **Added a fundamental screener** on top of the existing technical universe
  builder (`src/frankfurt_universe.py`, `docs/decisions/0005-fundamental-swing-screener.md`).
  Gates on turnover ≥ $1M, current ratio ≥ 1.0, debt/equity ≤ 150%, price ≥ $5,
  market cap ≥ $300M — thresholds centralized in `config.yaml`
  (`src/universe_config.py`). Outputs both `universe_fundamentals.csv` (raw)
  and `universe_screened.csv` (filtered).
- **New adapter/config layer**: `src/universe_adapter.py` and
  `src/universe_config.py` bridge the universe-builder module and the Dash UI
  without leaking framework code into `src/`.
- **`src/ui_helpers.py` slimmed down** (289 lines removed) — dropped
  Streamlit/HTML-specific helpers no longer needed post-migration; the old
  `tests/test_ui_helpers_html.py` was deleted accordingly.
- **Docs and tooling refreshed** to match: `AGENTS.md`, `CLAUDE.md`, `CODEX.md`,
  `README.md`, `docs/overview.md`, `docs/api.md`, `docs/context.md`,
  `docs/debugging.md`, `docs/logic.md`, `Makefile`, and dependency locks
  (`requirements.in/.txt`, `requirements-dev.txt`) all updated for Dash/Plotly.
- **New test coverage**: `tests/test_plotly_charts.py`,
  `tests/test_universe_adapter.py`, `tests/test_universe_jobs.py`.
- **Full test suite passes (39 tests)** when run with the project's `.venv`
  (`./.venv/bin/python -m pytest -q`) — the system/anaconda Python does not
  have `dash`/`plotly` installed, so always use `.venv` for this repo.
- **Full-codebase documentation pass completed.** Every function/class in
  `src/` (11 files) and `app.py` (49 total) now has a concise, accurate
  docstring — previously most of the codebase, including long-standing files
  like `signals.py` and `data.py`, had none. Style: plain prose, 1-3 lines,
  matching the handful of docstrings that already existed in the repo (no XML
  tags, no forced Args/Returns blocks).
- **Docs audit found and fixed 4 stale entries** left over from the Streamlit
  era: README's "Debug mode" section described a sidebar toggle that no
  longer exists; `docs/debugging.md` was entirely about that same toggle;
  README's Stock Details workflow described `Prev`/`Next` buttons that aren't
  in the Dash UI; and README's CSV-schema section described `portfolio.csv`/
  `watchlist.csv` as separately-read files when the app actually runs all
  tabs against one "active universe" CSV. All four fixed to match current
  `app.py` behavior.
- **Added `docs/decisions/0006-streamlit-to-dash-migration.md`** — the
  Streamlit→Dash migration itself had no ADR (0005 only covers the
  fundamental screener feature that shipped alongside it).
- **Universe tab: added CSV creation timestamps.** Each of the three CSVs
  (downloaded universe, fundamentals, screened) now shows a "Created:
  YYYY-MM-DD HH:MM:SS" label next to its step, read from the file's mtime and
  refreshed on the existing 2s poll (`_csv_timestamp_label` in `app.py`).
- **Manually smoke-tested the Dash app in-browser** (`make dev` equivalent,
  Chrome), all four tabs:
  - **Universe**: CSV creation timestamps render correctly for all three
    files (download/fundamentals/screened).
  - **Swing**: Stop button and "Last run" label (see bug fix below).
  - **Portfolio**: table renders with heat-map coloring, pagination works,
    lifecycle chart renders for the top-ranked stock, row selection works.
  - **Stock Details**: selecting a Portfolio row correctly loads decision
    card, trade levels, technical ratings, "Why this decision" rule/component
    tables, and the swing lifecycle (score vs price) chart — all populated
    correctly. No console errors, no Dash dev-tools errors.
  Found and fixed a real bug in the process — see below.
- **Fixed: Swing "Stop" button didn't actually stop promptly.**
  `compute_swing` built its entire ticker-data cache in one upfront dict
  comprehension (`{t: get_ticker_data(t) for t in sorted(tickers)}`) before
  the per-row loop where `should_stop` was checked — so clicking Stop while
  that comprehension was still fetching (the common case for a large
  universe: thousands of network calls) had no effect until it finished.
  Rewrote the cache-building step as an explicit loop that checks
  `should_stop` every ticker, so Stop now takes effect within a couple of
  ticker-fetches instead of waiting for the whole universe. Verified live:
  clicking Stop mid-run now reports e.g. "Stopped (swing stopped while
  fetching data (85/2773 tickers))" within ~2 seconds.

## Current state

- On branch `feat/dash-migration-universe`, all changes are **unstaged** in
  the working tree (nothing committed yet since `a0a3fa7`).
- Untracked new files: `.claude/`, `config.yaml`,
  `docs/decisions/0005-fundamental-swing-screener.md`,
  `docs/decisions/0006-streamlit-to-dash-migration.md`,
  `src/frankfurt_universe.py`, `src/plotly_charts.py`, `src/universe_adapter.py`,
  `src/universe_config.py`, `src/universe_jobs.py`, and their tests.
- Deleted: `tests/test_ui_helpers_html.py` (superseded by the Dash migration).
- Test suite is green; no known failing tests.
- All four tabs manually smoke-tested live in-browser this session (see
  above). Note: Portfolio/Stock Details were tested against the small bundled
  example universe (35 tickers) rather than the full 2771-row screened set,
  since Portfolio's tracker run is a synchronous per-request fetch (no
  background job/progress/cancel like Universe/Swing have) and would take a
  long time over the full screened universe — worth keeping in mind if the
  screened universe grows much larger.

## Next steps

1. **Decide on commit strategy** — this is a large, multi-concern diff
   (migration + new universe/fundamental feature + full docstring pass).
   Consider whether to split into logical commits (e.g. Dash migration vs.
   fundamental screener vs. docs/docstrings) or land as one PR; get user
   sign-off before committing/pushing.
2. **Review `.claude/` directory** (untracked) — confirm whether it's meant
   to be committed or should be gitignored.
3. Double check `config.yaml` (untracked, new) is the intended single source
   of truth for fundamental thresholds and is referenced correctly from
   `src/universe_config.py`.
4. `make lint` and the full test suite (39 tests) both currently pass on the
   full diff, including the new docstrings — re-verify after any further edits.
5. `swing_technical_snapshot` (`src/signals.py`) is tested but not currently
   called from `app.py` — it looks like a leftover from the old README's
   "Latest indicators" panel, which isn't wired into the Dash Stock Details
   tab. Worth a decision: wire it in, or drop the dead reference/test.
6. Portfolio tab's tracker run is synchronous (no background job/progress/
   Stop, unlike Universe/Swing) — fine at 35-2771 rows in a threaded dev
   server, but worth a decision if the universe grows much larger or the app
   moves to a non-threaded/production WSGI server.
