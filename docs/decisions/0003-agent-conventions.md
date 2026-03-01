# 0003 - Agent Conventions In-Repo

## Problem
Agent context is lost across sessions/tools.

## Decision
Maintain an in-repo context pack (`AGENTS.md`, `docs/context.md`, API docs, decision logs) as source of truth.

## Consequences
- Better continuity across Claude/Cursor/Codex.
- Reduced onboarding time.
- Requires contributors to update docs when behavior/contracts change.
