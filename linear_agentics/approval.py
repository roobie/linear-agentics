"""Human-in-the-loop approval gates."""

from __future__ import annotations

import asyncio

from .audit import ApprovalRecord
from .logger import get_logger
from .utils import _now_iso

log = get_logger(__name__)


class ApprovalDeniedError(Exception):
    pass


class ApprovalTimeoutError(Exception):
    pass


class ApprovalGate:
    """Manages human approval for actions that require it."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def prompt_cli(self, token_name: str, message: str) -> ApprovalRecord:
        """Prompt the user for approval via CLI stdin.

        Blocks until the user responds or times out.
        """
        log.info(
            "approval_required",
            token_name=token_name,
            message=message,
        )

        loop = asyncio.get_running_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, input),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise ApprovalTimeoutError(f"Approval for '{token_name}' timed out after {self.timeout_seconds}s")

        approved = response.strip().lower() in ("yes", "y")
        return ApprovalRecord(
            token_name=token_name,
            message=message,
            approved=approved,
            approver="cli-user",
            timestamp=_now_iso(),
        )
