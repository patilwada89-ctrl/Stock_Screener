"""Fail CI when code changes are missing required docs updates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RELEVANT_CODE_PATHS = (
    "app.py",
    "src/",
    "pyproject.toml",
)
DOC_PATHS = (
    "README.md",
    "AGENTS.md",
    "docs/",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _load_event_payload() -> dict:
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    path = Path(event_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_range() -> tuple[str, str]:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    event = _load_event_payload()

    if event_name == "pull_request":
        base_sha = event.get("pull_request", {}).get("base", {}).get("sha")
        head_sha = event.get("pull_request", {}).get("head", {}).get("sha")
        if base_sha and head_sha:
            return base_sha, head_sha

    if event_name == "push":
        before = event.get("before")
        after = event.get("after") or os.getenv("GITHUB_SHA")
        if before and after and before != "0" * 40:
            return before, after

    # Fallback for local runs / unusual CI contexts.
    try:
        base = _git("merge-base", "HEAD", "origin/main")
        return base, "HEAD"
    except subprocess.CalledProcessError:
        return "HEAD~1", "HEAD"


def _changed_files(base: str, head: str) -> list[str]:
    out = _git("diff", "--name-only", base, head)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_relevant_code_change(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in RELEVANT_CODE_PATHS)


def _is_doc_change(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in DOC_PATHS)


def main() -> int:
    base, head = _sha_range()
    changed = _changed_files(base, head)

    relevant_changed = [p for p in changed if _is_relevant_code_change(p)]
    doc_changed = [p for p in changed if _is_doc_change(p)]

    if not relevant_changed:
        print("[docs-check] No app/src/config changes detected; docs update not required.")
        return 0

    if doc_changed:
        print("[docs-check] Relevant code changed and docs were updated.")
        print("[docs-check] Updated docs files:")
        for p in doc_changed:
            print(f"  - {p}")
        return 0

    print("[docs-check] ERROR: Relevant code changed but no docs updates detected.")
    print("[docs-check] Changed code files:")
    for p in relevant_changed:
        print(f"  - {p}")
    print(
        "[docs-check] Please update at least one of: README.md, AGENTS.md, or docs/** "
        "to document behavior/API/contract changes."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
