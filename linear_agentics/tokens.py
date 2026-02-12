"""Linear capability tokens — use-once (or use-N) scoped action permits."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import asyncio

from .actions import (
    db_query,
    file_read,
    file_write,
    http_request,
    shell_exec,
    shell_exec_with_env,
)
from .audit import Proof
from .utils import _now_iso


class TokenError(Exception):
    pass


class TokenReusedError(TokenError):
    pass


class TokenScopeError(TokenError):
    pass


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

    def _mark_consumed(
        self, args: dict, result_summary: str, duration_ms: float
    ) -> Proof:
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

    async def consume(
        self, method: str | None = None, body: dict | None = None
    ) -> Proof:
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
                f"{'/'.join(self.methods)} {self.url}. ONE USE ONLY.{approval_note}"
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
            f"kubectl set image deployment/app app={image} --namespace={target}"
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


# ---------------------------------------------------------------------------
# FileToken
# ---------------------------------------------------------------------------

_FILE_MODES = {"read", "write", "readwrite"}


class _BaseFileToken:
    """Base class for file tokens - shared logic for single and multi-use."""

    def __init__(
        self,
        name: str,
        allowed_paths: list[str],
        mode: str,
        requires_approval: bool,
    ) -> None:
        if mode not in _FILE_MODES:
            raise ValueError(f"mode must be one of {_FILE_MODES}, got {mode!r}")
        scope = f"file:{mode}:{','.join(allowed_paths)}"
        self.name = name
        self.scope = scope
        self.requires_approval = requires_approval
        self.allowed_paths = allowed_paths
        self.mode = mode

    def _allowed_operations(self) -> list[str]:
        if self.mode == "readwrite":
            return ["read", "write"]
        return [self.mode]

    def _validate_operation(self, operation: str) -> None:
        allowed_ops = self._allowed_operations()
        if operation not in allowed_ops:
            raise TokenScopeError(
                f"Operation {operation!r} not allowed by mode {self.mode!r}. "
                f"Allowed: {allowed_ops}"
            )

    def _build_tool_definition(self, uses_remaining: int | None) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        ops = self._allowed_operations()
        props: dict = {
            "operation": {
                "type": "string",
                "enum": ops,
                "description": "File operation to perform",
            },
            "path": {
                "type": "string",
                "description": "Absolute path to the file",
            },
        }
        if "write" in ops:
            props["content"] = {
                "type": "string",
                "description": "Content to write (required for write operations)",
            }
        if uses_remaining is not None:
            desc = (
                f"File access ({self.mode}) within {self.allowed_paths}. "
                f"{uses_remaining} uses remaining.{approval_note}"
            )
        else:
            desc = (
                f"File access ({self.mode}) within {self.allowed_paths}. "
                f"ONE USE ONLY.{approval_note}"
            )
        return {
            "name": f"file_{self.name}",
            "description": desc,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": ["operation", "path"],
            },
        }


class FileToken(LinearToken):
    """Scoped filesystem access token (single use)."""

    def __init__(
        self,
        name: str,
        allowed_paths: list[str],
        mode: str = "read",
        requires_approval: bool = False,
    ) -> None:
        base = _BaseFileToken(name, allowed_paths, mode, requires_approval)
        super().__init__(base.name, base.scope, requires_approval)
        self.allowed_paths = base.allowed_paths
        self.mode = base.mode
        self._base = base

    async def consume(
        self, operation: str, path: str, content: str | None = None
    ) -> Proof:
        self._check_reuse()
        self._base._validate_operation(operation)
        t0 = time.monotonic()
        if operation == "read":
            result = await file_read(path, self.allowed_paths)
        else:
            if content is None:
                raise ValueError("content is required for write operations")
            result = await file_write(path, content, self.allowed_paths)
        duration = (time.monotonic() - t0) * 1000
        summary = result[:200] if result else "(empty)"
        return self._mark_consumed(
            {"operation": operation, "path": path}, summary, duration
        )

    def to_tool_definition(self) -> dict:
        return self._base._build_tool_definition(None)


class MultiUseFileToken(MultiUseToken):
    """Scoped filesystem access token with N uses."""

    def __init__(
        self,
        name: str,
        allowed_paths: list[str],
        mode: str = "read",
        max_uses: int = 5,
        requires_approval: bool = False,
    ) -> None:
        base = _BaseFileToken(name, allowed_paths, mode, requires_approval)
        super().__init__(base.name, base.scope, max_uses, requires_approval)
        self.allowed_paths = base.allowed_paths
        self.mode = base.mode
        self._base = base

    async def consume(
        self, operation: str, path: str, content: str | None = None
    ) -> Proof:
        self._check_reuse()
        self._base._validate_operation(operation)
        t0 = time.monotonic()
        if operation == "read":
            result = await file_read(path, self.allowed_paths)
        else:
            if content is None:
                raise ValueError("content is required for write operations")
            result = await file_write(path, content, self.allowed_paths)
        duration = (time.monotonic() - t0) * 1000
        summary = result[:200] if result else "(empty)"
        return self._record_use(
            {"operation": operation, "path": path}, summary, duration
        )

    def to_tool_definition(self) -> dict:
        return self._base._build_tool_definition(self.uses_remaining)


# ---------------------------------------------------------------------------
# SecretToken (wrapper pattern)
# ---------------------------------------------------------------------------


@dataclass
class SecretInjection:
    """Describes how a secret is injected into an action."""

    kind: str  # "header" or "env"
    key: str  # header name or env var name


class SecretToken(LinearToken):
    """Injects a credential into a single action without exposing it to the LLM.

    Wraps an inner token (HttpToken or ShellToken). The LLM sees the same
    tool interface as the inner token, but the secret is injected
    transparently at execution time.
    """

    def __init__(
        self,
        name: str,
        secret_value: str,
        injection: SecretInjection,
        inner_token: LinearToken,
        requires_approval: bool = False,
    ) -> None:
        scope = f"secret:{injection.kind}:{injection.key}:{inner_token.scope}"
        super().__init__(name, scope, requires_approval)
        self._secret_value = secret_value
        self.injection = injection
        self.inner_token = inner_token

    async def consume(self, **kwargs) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()

        if self.injection.kind == "header":
            # Inner token is an HttpToken — call http_request directly with
            # the secret injected as a header.
            inner = self.inner_token
            if not isinstance(inner, HttpToken):
                raise TokenScopeError(
                    "header injection requires an HttpToken as inner_token"
                )
            method = kwargs.get("method") or inner.methods[0]
            body = kwargs.get("body")
            headers = {self.injection.key: self._secret_value}
            result = await http_request(
                inner.url,
                method,
                inner.methods,
                body=body,
                headers=headers,
            )
            result_summary = f"HTTP {result['status']}: {result['body'][:150]}"

        elif self.injection.kind == "env":
            # Inner token is a ShellToken — call shell_exec_with_env with
            # the secret as an environment variable.
            inner = self.inner_token
            if not isinstance(inner, ShellToken):
                raise TokenScopeError(
                    "env injection requires a ShellToken as inner_token"
                )
            command = kwargs["command"]
            env_vars = {self.injection.key: self._secret_value}
            result_str = await shell_exec_with_env(
                command,
                inner.allowed,
                env_vars,
            )
            result_summary = result_str[:200] if result_str else "(empty)"

        else:
            raise TokenScopeError(f"Unknown injection kind: {self.injection.kind!r}")

        duration = (time.monotonic() - t0) * 1000
        # Redact: never include secret in proof args
        redacted_args = {k: v for k, v in kwargs.items()}
        return self._mark_consumed(
            redacted_args,
            f"[secret injected as {self.injection.kind}:{self.injection.key}] "
            + result_summary,
            duration,
        )

    def to_tool_definition(self) -> dict:
        """Mirror the inner token's tool schema under a different name."""
        inner_def = self.inner_token.to_tool_definition()
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"secret_{self.name}",
            "description": (
                f"Execute with injected credential "
                f"({self.injection.kind}:{self.injection.key}). "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": inner_def["input_schema"],
        }


# ---------------------------------------------------------------------------
# DatabaseToken
# ---------------------------------------------------------------------------


class DatabaseToken(LinearToken):
    """Parameterised database query token (single use)."""

    def __init__(
        self,
        name: str,
        dsn: str,
        allowed_patterns: list[str],
        requires_approval: bool = False,
    ) -> None:
        scope = f"db:{','.join(allowed_patterns)}"
        super().__init__(name, scope, requires_approval)
        self._dsn = dsn
        self.allowed_patterns = allowed_patterns

    async def consume(self, query: str, params: list | None = None) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()
        rows = await db_query(self._dsn, query, params, self.allowed_patterns)
        duration = (time.monotonic() - t0) * 1000
        summary = str(rows)[:200]
        return self._mark_consumed(
            {"query": query, "params": params}, summary, duration
        )

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"db_{self.name}",
            "description": (
                f"Execute a parameterised SQL query. "
                f"Allowed patterns: {self.allowed_patterns}. "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query with $1, $2, ... placeholders",
                    },
                    "params": {
                        "type": "array",
                        "items": {},
                        "description": "Positional parameter values",
                    },
                },
                "required": ["query"],
            },
        }


class MultiUseDatabaseToken(MultiUseToken):
    """Parameterised database query token with N uses."""

    def __init__(
        self,
        name: str,
        dsn: str,
        allowed_patterns: list[str],
        max_uses: int = 5,
        requires_approval: bool = False,
    ) -> None:
        scope = f"db:{','.join(allowed_patterns)}"
        super().__init__(name, scope, max_uses, requires_approval)
        self._dsn = dsn
        self.allowed_patterns = allowed_patterns

    async def consume(self, query: str, params: list | None = None) -> Proof:
        self._check_reuse()
        t0 = time.monotonic()
        rows = await db_query(self._dsn, query, params, self.allowed_patterns)
        duration = (time.monotonic() - t0) * 1000
        summary = str(rows)[:200]
        return self._record_use({"query": query, "params": params}, summary, duration)

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"db_{self.name}",
            "description": (
                f"Execute a parameterised SQL query. "
                f"Allowed patterns: {self.allowed_patterns}. "
                f"{self.uses_remaining} uses remaining.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query with $1, $2, ... placeholders",
                    },
                    "params": {
                        "type": "array",
                        "items": {},
                        "description": "Positional parameter values",
                    },
                },
                "required": ["query"],
            },
        }


# ---------------------------------------------------------------------------
# WaitToken
# ---------------------------------------------------------------------------


class WaitToken(LinearToken):
    """Controlled delay token — asyncio.sleep with a max-seconds cap."""

    def __init__(
        self,
        name: str,
        max_seconds: int = 300,
        requires_approval: bool = False,
    ) -> None:
        scope = f"wait:{max_seconds}s"
        super().__init__(name, scope, requires_approval)
        self.max_seconds = max_seconds

    async def consume(self, seconds: int, reason: str = "") -> Proof:
        self._check_reuse()
        if seconds > self.max_seconds:
            raise ValueError(
                f"Requested {seconds}s exceeds maximum of {self.max_seconds}s"
            )
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        t0 = time.monotonic()
        await asyncio.sleep(seconds)
        duration = (time.monotonic() - t0) * 1000
        summary = f"Waited {seconds}s"
        if reason:
            summary += f" ({reason})"
        return self._mark_consumed(
            {"seconds": seconds, "reason": reason}, summary, duration
        )

    def to_tool_definition(self) -> dict:
        approval_note = " REQUIRES APPROVAL." if self.requires_approval else ""
        return {
            "name": f"wait_{self.name}",
            "description": (
                f"Wait/sleep for a specified duration (max {self.max_seconds}s). "
                f"ONE USE ONLY.{approval_note}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the wait is needed",
                    },
                },
                "required": ["seconds"],
            },
        }
