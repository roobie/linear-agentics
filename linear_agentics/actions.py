"""Low-level action executors — shell commands, HTTP requests, file I/O, and database queries."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path

import httpx


class CommandNotAllowedError(Exception):
    pass


class FileAccessError(Exception):
    pass


class QueryNotAllowedError(Exception):
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


# ---------------------------------------------------------------------------
# Shell with environment injection
# ---------------------------------------------------------------------------


async def shell_exec_with_env(
    command: str,
    allowed_prefixes: list[str],
    env_vars: dict[str, str],
    timeout_seconds: float = 30.0,
) -> str:
    """Like shell_exec but merges *env_vars* into the subprocess environment."""
    argv = _validate_and_split(command, allowed_prefixes)
    merged_env = {**os.environ, **env_vars}

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
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


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def _validate_file_path(path: str, allowed_paths: list[str]) -> str:
    """Resolve *path* and check it falls under one of *allowed_paths*.

    Returns the resolved absolute path.  Raises ``FileAccessError`` on
    traversal attempts or paths outside the allowed set.
    """
    resolved = os.path.realpath(path)
    for allowed in allowed_paths:
        allowed_resolved = os.path.realpath(allowed)
        if resolved == allowed_resolved:
            return resolved
        # Allow children of allowed directories
        if resolved.startswith(allowed_resolved + os.sep):
            return resolved
    raise FileAccessError(
        f"Path {path!r} (resolved: {resolved!r}) is not within "
        f"allowed paths: {allowed_paths}"
    )


async def file_read(path: str, allowed_paths: list[str]) -> str:
    """Read a file after validating its path against *allowed_paths*."""
    resolved = _validate_file_path(path, allowed_paths)
    return await asyncio.to_thread(Path(resolved).read_text)


async def file_write(path: str, content: str, allowed_paths: list[str]) -> str:
    """Write *content* to a file after validating its path."""
    resolved = _validate_file_path(path, allowed_paths)

    def _write() -> int:
        p = Path(resolved)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.write_text(content)

    n = await asyncio.to_thread(_write)
    return f"Wrote {n} bytes to {resolved}"


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

_SQL_COMMENT_RE = re.compile(r"--|/\*")


def _validate_query(query: str, allowed_patterns: list[str]) -> None:
    """Validate a SQL query against allowed pattern prefixes.

    Rejects multi-statement queries (containing ``;``) and SQL comments.
    """
    if ";" in query:
        raise QueryNotAllowedError(
            "Query contains ';' — multi-statement queries are not allowed"
        )
    if _SQL_COMMENT_RE.search(query):
        raise QueryNotAllowedError("Query contains SQL comments — not allowed")
    normalised = " ".join(query.split()).strip()
    for pattern in allowed_patterns:
        pattern_normalised = " ".join(pattern.split()).strip()
        if normalised.upper().startswith(pattern_normalised.upper()):
            return
    raise QueryNotAllowedError(
        f"Query not allowed. Permitted prefixes: {allowed_patterns}"
    )


async def db_query(
    dsn: str,
    query: str,
    params: list | None,
    allowed_patterns: list[str],
    timeout_seconds: float = 30.0,
) -> list[dict]:
    """Execute a parameterised query via asyncpg.

    Returns a list of row dicts.  Requires ``asyncpg`` to be installed
    (available via the ``linear-agentics[db]`` extra).
    """
    try:
        import asyncpg  # noqa: F811
    except ImportError:
        raise ImportError(
            "asyncpg is required for DatabaseToken. "
            "Install it with: pip install linear-agentics[db]"
        )

    _validate_query(query, allowed_patterns)

    conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout_seconds)
    try:
        params = params or []
        result = await asyncio.wait_for(
            conn.fetch(query, *params), timeout=timeout_seconds
        )
        return [dict(row) for row in result]
    finally:
        await conn.close()
