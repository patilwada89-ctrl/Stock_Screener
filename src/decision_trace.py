"""Typed decision trace objects for debuggable stock evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RuleTrace:
    """One pass/fail hard-filter rule (e.g. a weekly EMA gating condition) with its evaluated value."""

    name: str
    passed: bool
    value: str


@dataclass
class ComponentTrace:
    """One weighted score component: its +1/0/-1 ``signal``, the underlying value, and its weight."""

    name: str
    signal: int
    value: str
    weight: float = 1.0


@dataclass
class DecisionTrace:
    """Full auditable record of a Buy/Sell/Hold decision: rules, weighted components, and notes.

    Built by ``src.signals.build_swing_decision_trace`` and rendered by the
    Stock Details "why" panel so every decision is traceable back to its inputs.
    """

    name: str
    signal_ticker: str
    benchmark: str
    score: float
    score_name: str
    decision: str
    qualified: bool
    setup_type: str
    risk_flag: str
    risk_reason: str
    rules: list[RuleTrace] = field(default_factory=list)
    components: list[ComponentTrace] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert this trace (and its nested rule/component traces) to plain dicts."""
        return asdict(self)
