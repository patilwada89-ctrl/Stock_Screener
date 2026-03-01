"""Typed decision trace objects for debuggable stock evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RuleTrace:
    name: str
    passed: bool
    value: str


@dataclass
class ComponentTrace:
    name: str
    signal: int
    value: str
    weight: float = 1.0


@dataclass
class DecisionTrace:
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
        return asdict(self)
