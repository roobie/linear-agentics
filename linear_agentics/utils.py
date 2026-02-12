"""Shared utilities for linear-agentics."""

from __future__ import annotations

from datetime import datetime, timezone


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
