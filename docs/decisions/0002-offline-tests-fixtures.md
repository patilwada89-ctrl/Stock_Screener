# 0002 - Offline Tests With Fixtures

## Problem
Network-dependent tests are flaky and non-reproducible in CI.

## Decision
Default unit tests must use local deterministic fixtures/synthetic data. Integration/network tests are optional and marked.

## Consequences
- Stable CI.
- Faster local test runs.
- Clear separation between unit and integration concerns.
