# Code Improvement Checklist

## Security

- [ ] **Secret handling** (`tokens.py:565`) - `SecretToken` stores secrets as plain strings in memory. Consider using a secrets manager or encrypting at rest.
- [ ] **Environment variable exposure** (`actions.py:129`) - `shell_exec_with_env` merges env vars but doesn't filter sensitive variables from being passed to subprocess.

## Code Quality

- [x] ~~Duplicate `_now_iso()` function~~ - defined in `audit.py`, `approval.py`, `tokens.py`. Extract to a shared utils module.
- [x] ~~Deprecated asyncio pattern~~ - `asyncio.get_event_loop()` is deprecated. Use `asyncio.get_running_loop()` instead.
- [x] ~~Missing `__init__.py` exports~~ - No public API defined in `linear_agentics/__init__.py`.
- [ ] No connection pooling - `actions.py:103` creates a new `httpx.AsyncClient` per request. Reuse clients for better performance.
- [ ] Hardcoded LLM provider (`agent.py:96`) - Only supports Anthropic. Consider an abstract `LLMClient` protocol for flexibility.
- [ ] Duplicated file token logic - `FileToken` and `MultiUseFileToken` share ~80% identical code. Extract to a base class.

## Testing

- [ ] No integration tests - Tests mock everything. Add tests that exercise actual shell/DB/HTTP execution with real constraints.
- [ ] Missing test coverage - No tests for `DeployToken`, `WaitToken`, `SecretToken`, `DatabaseToken`.

## Configuration & DX

- [ ] Add type hints for external deps - Install `anthropic-stubs` or add `type: ignore` comments for third-party APIs.
- [ ] Add linting/formatting config - `pyproject.toml` lacks `ruff` or `mypy` configuration.
- [ ] Add structured logging - Replace `print()` statements with `logging` module for production use.

## Reliability

- [ ] No retry logic - HTTP and DB operations have no retry mechanism for transient failures.
- [ ] No circuit breaker - A failing token can repeatedly fail without backoff.
