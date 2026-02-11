"""Audit trail — immutable records of token consumption and agent actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass(frozen=True)
class Proof:
    """Immutable record of a token being consumed."""

    token_name: str
    scope: str
    args: dict
    timestamp: str
    result_summary: str
    duration_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApprovalRecord:
    """Record of a human approval decision."""

    token_name: str
    message: str
    approved: bool
    approver: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ErrorRecord:
    """Record of an error during agent execution."""

    token_name: str
    error_type: str
    message: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NegotiationRecord:
    """Record of a capability negotiation attempt."""

    requested_scope: str
    justification: str
    granted_token: str | None  # token name if granted, None if denied
    provider_type: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditTrail:
    """Collects all proofs, approvals, and errors from an agent run."""

    proofs: list[Proof] = field(default_factory=list)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    negotiations: list[NegotiationRecord] = field(default_factory=list)

    def record_proof(self, proof: Proof) -> None:
        self.proofs.append(proof)

    def record_approval(self, record: ApprovalRecord) -> None:
        self.approvals.append(record)

    def record_negotiation(self, record: NegotiationRecord) -> None:
        self.negotiations.append(record)

    def record_error(self, token_name: str, error: Exception) -> None:
        self.errors.append(
            ErrorRecord(
                token_name=token_name,
                error_type=type(error).__name__,
                message=str(error),
                timestamp=_now_iso(),
            )
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "proofs": [p.to_dict() for p in self.proofs],
                "approvals": [a.to_dict() for a in self.approvals],
                "errors": [e.to_dict() for e in self.errors],
                "negotiations": [n.to_dict() for n in self.negotiations],
            },
            indent=indent,
        )

    def to_table(self) -> str:
        lines = []
        lines.append(f"{'#':<4} {'Token':<25} {'Scope':<20} {'Duration':>10}  Result")
        lines.append("-" * 80)
        for i, p in enumerate(self.proofs, 1):
            dur = f"{p.duration_ms:.0f}ms"
            summary = p.result_summary[:30] if p.result_summary else ""
            lines.append(
                f"{i:<4} {p.token_name:<25} {p.scope:<20} {dur:>10}  {summary}"
            )
        if self.negotiations:
            lines.append("")
            lines.append("Negotiations:")
            for n in self.negotiations:
                status = f"granted:{n.granted_token}" if n.granted_token else "denied"
                lines.append(f"  [{status}] {n.requested_scope}: {n.justification}")
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  [{e.error_type}] {e.token_name}: {e.message}")
        return "\n".join(lines)
