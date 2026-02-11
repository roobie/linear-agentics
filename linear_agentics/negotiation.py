"""Capability negotiation — runtime token requests from providers."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Protocol

from .tokens import LinearToken

if TYPE_CHECKING:
    from .agent import Agent


class CapabilityProvider(Protocol):
    """Protocol for granting additional capabilities during agent execution."""

    async def request_capability(
        self,
        requested_scope: str,
        justification: str,
        candidates: list[LinearToken],
    ) -> LinearToken | None:
        """Request a capability matching the given scope.

        Args:
            requested_scope: The scope string the agent is requesting.
            justification: Why the agent needs this capability.
            candidates: Pre-configured tokens available for granting.

        Returns:
            A token from candidates if granted, None if denied.
        """
        ...


class HumanCapabilityProvider:
    """Human operator grants capabilities via CLI prompt."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def request_capability(
        self,
        requested_scope: str,
        justification: str,
        candidates: list[LinearToken],
    ) -> LinearToken | None:
        matches = [t for t in candidates if not t.consumed]
        if not matches:
            return None

        print(f"\n{'=' * 60}", file=sys.stderr)
        print("CAPABILITY REQUEST", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(f"Requested scope: {requested_scope}", file=sys.stderr)
        print(f"Justification: {justification}\n", file=sys.stderr)
        print("Available candidates:", file=sys.stderr)
        for i, token in enumerate(matches, 1):
            print(f"  {i}. {token.name} ({token.scope})", file=sys.stderr)
        print("  0. Deny request", file=sys.stderr)
        print("\nEnter number to grant (0 to deny):", file=sys.stderr)

        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, input),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return None

        try:
            choice = int(response.strip())
            if 1 <= choice <= len(matches):
                return matches[choice - 1]
        except (ValueError, IndexError):
            pass

        return None


class SupervisorAgentProvider:
    """Delegate capability requests to a supervisor agent.

    The supervisor agent evaluates the request and decides whether to
    grant a capability from the candidate list.
    """

    def __init__(self, supervisor: Agent) -> None:
        self.supervisor = supervisor

    async def request_capability(
        self,
        requested_scope: str,
        justification: str,
        candidates: list[LinearToken],
    ) -> LinearToken | None:
        raise NotImplementedError(
            "SupervisorAgentProvider is not yet implemented. "
            "Use HumanCapabilityProvider for now."
        )
