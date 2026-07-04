# 0006 - Streamlit to Dash + Plotly Migration

## Problem
The app was originally a Streamlit app. Streamlit's rerun-on-every-interaction
model made background work (the Frankfurt/Xetra universe download and the
fundamental screening pass, both of which take minutes and hit the network)
awkward to run without blocking the UI or duplicating work on every widget
interaction. There was no clean way to poll a long-running job's progress and
let the user keep using the rest of the app while it ran.

## Decision
- Rewrite the UI layer in **Dash + Plotly** (`app.py`): tabs (`dcc.Tabs`),
  callbacks, `dcc.Store` for cross-tab state, and `dash_table.DataTable` for
  all tables. `src/` stays Streamlit/Dash-agnostic — pure functions only.
- Long-running work (universe download, screening, the swing screener) runs
  in background daemon threads (`src/universe_jobs.py`), reporting progress
  to disk (`data/job_status.json`, `data/<job>.log`) that a `dcc.Interval`
  polls. This makes the two-step Universe workflow (and the Swing "Run
  screener" action) genuinely non-blocking, which Streamlit's execution model
  couldn't offer without extra process/thread plumbing of its own.
- Charting moved from Altair to Plotly (`src/plotly_charts.py`): pure figure
  builders (rating gauge, lifecycle score line, lifecycle-with-price overlay)
  with no Dash imports, matching the existing "pure `src/`" convention.
- All four tabs (Universe, Swing, Portfolio, Stock Details) were ported to
  full functional parity in this migration — none are stubs; each reuses the
  same `src/signals.py` / `src/ratings.py` / `src/decision_trace.py` logic the
  Streamlit version used.

## Consequences
- `requirements.in`/`requirements.txt` now pin `dash`, `dash-bootstrap-components`,
  and `plotly` instead of `streamlit`/`altair`; the dev venv must be rebuilt
  (`pip install -r requirements-dev.txt`) after pulling this change.
- Background jobs are cooperatively cancellable (`stop_job` sets a
  `threading.Event`) and survive a server restart cleanly: a stale "running"
  status is reconciled to "interrupted" via `mark_interrupted`/`job_status`,
  and the underlying resolve/fundamentals disk caches are preserved so a
  stopped job resumes cheaply rather than refetching everything.
- Row-level UI interactions (selecting a stock to open in Stock Details) now
  go through Dash pattern-matching IDs (`{"type": "swing-analyze", "ticker": ...}`)
  and `dcc.Store`, replacing Streamlit's session-state-based navigation.
- Existing `src/` unit tests were unaffected — this migration only touched
  `app.py` and added new `src/` modules (`universe_jobs.py`,
  `universe_adapter.py`, `universe_config.py`, `plotly_charts.py`,
  `frankfurt_universe.py`); it did not change scoring/rule behavior.
