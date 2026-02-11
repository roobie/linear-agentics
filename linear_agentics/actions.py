"""Low-level action executors — shell commands and HTTP requests."""

from __future__ import annotations

import asyncio
import shlex

import httpx


class CommandNotAllowedError(Exception):
    pass


_SHELL_OPERATORS = frozenset(("||", "&&", ";", "|", "&"))


def _validate_and_split(command: str, allowed_prefixes: list[str]) -> list[str]:
    """Parse command into argv and check that it starts with an allowed prefix.

    Rejects commands containing shell composition operators (||, &&, ;, |, &)
    to prevent attempts at command chaining.
    """
    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise CommandNotAllowedError(f"Invalid command syntax: {e}")
    shell_ops = _SHELL_OPERATORS.intersection(argv)
    if shell_ops:
        raise CommandNotAllowedError(
            f"Command contains shell operators {shell_ops}: {command!r}"
        )
    for prefix in allowed_prefixes:
        prefix_parts = shlex.split(prefix)
        if argv[: len(prefix_parts)] == prefix_parts:
            return argv
    raise CommandNotAllowedError(
        f"Command {command!r} not allowed. Permitted prefixes: {allowed_prefixes}"
    )


async def shell_exec(
    command: str,
    allowed_prefixes: list[str],
    timeout_seconds: float = 30.0,
) -> str:
    """Execute a shell command after validating it against allowed prefixes.

    Returns stdout. Raises on validation failure or non-zero exit.
    """
    argv = _validate_and_split(command, allowed_prefixes)

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Command timed out after {timeout_seconds}s: {command}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {command}\n"
            f"stderr: {stderr.decode(errors='replace')}"
        )
    return stdout.decode(errors="replace")


async def http_request(
    url: str,
    method: str,
    allowed_methods: list[str],
    body: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> dict:
    """Make an HTTP request after validating the method is allowed.

    Returns {"status": int, "body": str}.
    """
    method_upper = method.upper()
    if method_upper not in [m.upper() for m in allowed_methods]:
        raise ValueError(
            f"HTTP method {method_upper!r} not allowed. Permitted: {allowed_methods}"
        )

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.request(
            method_upper,
            url,
            json=body,
            headers=headers,
        )
    return {
        "status": response.status_code,
        "body": response.text,
    }
