"""Linear capability tokens — use-once (or use-N) scoped action permits."""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timezone

from .actions import shell_exec, http_request
from .audit import Proof


class TokenError(Exception):
    pass


class TokenReusedError(TokenError):
    pass


class TokenScopeError(TokenError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LinearToken:
    """Base class for linear (use-once) capability tokens.

    Each token represents a scoped, audited, one-time-use capability.
    Consuming a token returns a Proof recording what happened.
    """

    def __init__(
        self,
        name: str,
        scope: str,
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.scope = scope
        self.requires_approval = requires_approval
        self._consumed = False
        self._proof: Proof | None = None

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def proof(self) -> Proof | None:
        return self._proof

    def _mark_consumed(self, args: dict, result_summary: str, duration_ms: float) -> Proof:
        self._consumed = True
        self._proof = Proof(
            token_name=self.name,
            scope=self.scope,
            args=args,
            timestamp=_now_iso(),
            result_summary=result_summary,
            duration_ms=duration_ms,
        )
        return self._proof

    def _check_reuse(self) -> None:
        if self._consumed:
            raise TokenReusedError(f"Capability '{self.name}' already consumed.")

    async def consume(self, **kwargs) -> Proof:
        raise NotImplementedError("Subclasses must implement consume()")

    def to_tool_definition(self) -> dict:
        """Return an LLM-compatible tool definition dict."""
        raise NotImplementedError("Subclasses must implement to_tool_definition()")

    def __del__(self) -> None:
        if not self._consumed:
            warnings.warn(
                f"Token '{self.name}' was never consumed. "
                "This may indicate an unused capability.",
                stacklevel=1,
            )


class ShellToken(LinearToken):
    """Scoped shell command execution token."""

    def __init__(
        self,
        name: str,
        allowed: list[str],
        requires_approval: bool = False,
    ) -> None:
        scope = f"shell:{','.join(allowed)}"
        super().__init__(name, scope, requires_approval)
        self.allowed = allowed

    async def consume(self, command: str) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()
        result = await shell_exec(command, self.allowed)
        duration = (time.monotonic() - t0) * 1000
        summary = result[:200] if result else "(empty)"
        return self._mark_consumed({"command": command}, summary, duration)

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"shell_{self.name}",
            "description": (
                f"Run a shell command. Allowed prefixes: {self.allowed}. "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        }


class HttpToken(LinearToken):
    """Scoped HTTP request token."""

    def __init__(
        self,
        name: str,
        url: str,
        methods: list[str],
        requires_approval: bool = False,
    ) -> None:
        scope = f"http:{','.join(methods)}:{url}"
        super().__init__(name, scope, requires_approval)
        self.url = url
        self.methods = methods

    async def consume(self, method: str | None = None, body: dict | None = None) -> Proof:
        self._check_reuse()
        if method is None:
            method = self.methods[0]
        t0 = time.monotonic()
        result = await http_request(self.url, method, self.methods, body=body)
        duration = (time.monotonic() - t0) * 1000
        summary = f"HTTP {result['status']}: {result['body'][:150]}"
        return self._mark_consumed(
            {"method": method, "url": self.url, "body": body}, summary, duration
        )

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": self.name,
            "description": (
                f"{'/'.join(self.methods)} {self.url}. "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": self.methods,
                        "description": "HTTP method to use",
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional JSON request body",
                    },
                },
                "required": [],
            },
        }


class DeployToken(LinearToken):
    """One-time deployment action token."""

    def __init__(
        self,
        name: str,
        method: str,
        target: str,
        image: str,
        rollback_to: str | None = None,
        requires_approval: bool = False,
    ) -> None:
        scope = f"deploy:{method}:{target}:{image}"
        super().__init__(name, scope, requires_approval)
        self.method = method
        self.target = target
        self.image = image
        self.rollback_to = rollback_to
        self._deploy_command = (
            f"kubectl set image deployment/app app={image} "
            f"--namespace={target}"
        )
        self._rollback_command: str | None = None
        if rollback_to:
            self._rollback_command = (
                f"kubectl set image deployment/app app={rollback_to} "
                f"--namespace={target}"
            )

    async def consume(self) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()
        result = await shell_exec(
            self._deploy_command,
            allowed_prefixes=["kubectl set image", "kubectl rollout"],
        )
        duration = (time.monotonic() - t0) * 1000
        summary = f"Deployed {self.image} to {self.target}: {result[:100]}"
        return self._mark_consumed(
            {"method": self.method, "target": self.target, "image": self.image},
            summary,
            duration,
        )

    async def rollback(self) -> str:
        if not self._rollback_command:
            raise TokenError(f"No rollback target configured for '{self.name}'")
        return await shell_exec(
            self._rollback_command,
            allowed_prefixes=["kubectl set image", "kubectl rollout"],
        )

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"deploy_{self.name}",
            "description": (
                f"Deploy {self.image} to {self.target} via {self.method}. "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class MultiUseToken(LinearToken):
    """Token with N uses instead of 1. For read operations that need repetition."""

    def __init__(
        self,
        name: str,
        scope: str,
        max_uses: int,
        requires_approval: bool = False,
    ) -> None:
        super().__init__(name, scope, requires_approval)
        self.max_uses = max_uses
        self._use_count = 0
        self._proofs: list[Proof] = []
        # Override _consumed behavior — MultiUseToken is consumed after max_uses
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._use_count >= self.max_uses

    @property
    def uses_remaining(self) -> int:
        return max(0, self.max_uses - self._use_count)

    def _check_reuse(self) -> None:
        if self._use_count >= self.max_uses:
            raise TokenReusedError(
                f"Capability '{self.name}' exhausted ({self.max_uses}/{self.max_uses} uses)."
            )

    def _record_use(self, args: dict, result_summary: str, duration_ms: float) -> Proof:
        self._use_count += 1
        proof = Proof(
            token_name=self.name,
            scope=self.scope,
            args=args,
            timestamp=_now_iso(),
            result_summary=result_summary,
            duration_ms=duration_ms,
        )
        self._proofs.append(proof)
        self._proof = proof
        return proof

    async def consume(self, **kwargs) -> Proof:
        raise NotImplementedError("MultiUseToken subclasses must implement consume()")

    def to_tool_definition(self) -> dict:
        return {
            "name": self.name,
            "description": f"{self.scope}. {self.max_uses} uses remaining.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }

    def __del__(self) -> None:
        if self._use_count == 0:
            warnings.warn(
                f"MultiUseToken '{self.name}' was never used. "
                "This may indicate an unused capability.",
                stacklevel=1,
            )


class MultiUseShellToken(MultiUseToken):
    """Shell token with N uses. For read commands that may need repetition."""

    def __init__(
        self,
        name: str,
        allowed: list[str],
        max_uses: int,
        requires_approval: bool = False,
    ) -> None:
        scope = f"shell:{','.join(allowed)}"
        super().__init__(name, scope, max_uses, requires_approval)
        self.allowed = allowed

    async def consume(self, command: str) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()
        result = await shell_exec(command, self.allowed)
        duration = (time.monotonic() - t0) * 1000
        summary = result[:200] if result else "(empty)"
        return self._record_use({"command": command}, summary, duration)

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"shell_{self.name}",
            "description": (
                f"Run a shell command. Allowed prefixes: {self.allowed}. "
                f"{self.uses_remaining} uses remaining.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        }
