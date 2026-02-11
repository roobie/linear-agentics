"""Low-level action executors — shell commands and HTTP requests."""

from __future__ import annotations

import asyncio
import shlex

import httpx


class CommandNotAllowedError(Exception):
    pass


def _validate_command_prefix(command: str, allowed_prefixes: list[str]) -> None:
    """Check that command starts with one of the allowed prefixes."""
    for prefix in allowed_prefixes:
        if command == prefix or command.startswith(prefix + " "):
            return
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
    _validate_command_prefix(command, allowed_prefixes)

    proc = await asyncio.create_subprocess_shell(
        command,
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
