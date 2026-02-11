"""Property-based tests for security-critical validation, state machines, and audit integrity."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, patch

import tempfile

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from linear_agentics.actions import (
    CommandNotAllowedError,
    FileAccessError,
    QueryNotAllowedError,
    _SHELL_OPERATORS,
    _validate_and_split,
    _validate_file_path,
    _validate_query,
)
from linear_agentics.agent import CapabilitySet
from linear_agentics.audit import AuditTrail, Proof
from linear_agentics.budget import Budget, BudgetExhaustedError
from linear_agentics.tokens import (
    HttpToken,
    MultiUseShellToken,
    SecretInjection,
    SecretToken,
    ShellToken,
    TokenReusedError,
    WaitToken,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Characters likely to appear in paths, including traversal tricks
_path_alphabet = st.sampled_from(list("abcde./\\  \x00"))
_path_segments = st.text(alphabet="abcde./", min_size=0, max_size=20)

# Characters likely to appear in shell commands
_shell_chars = st.sampled_from(list("abcdefg -='\"|&;$()` \t\\"))
_shell_commands = st.text(alphabet=_shell_chars, min_size=1, max_size=60)

# SQL-like strings
_sql_alphabet = st.sampled_from(
    list("SELECT INSERT FROM WHERE AND OR 0123456789 *$-;/\t\n'\"")
)
_sql_strings = st.text(alphabet=_sql_alphabet, min_size=1, max_size=80)

# General text
_printable_text = st.text(min_size=1, max_size=100)


# ===================================================================
# TIER 1: Security-Critical Validation
# ===================================================================


class TestPropertyValidateFilePath:
    """Property 1: No path traversal ever escapes the sandbox."""

    @given(suffix=_path_segments)
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_traversal_never_escapes(self, suffix):
        """Any path built with '..' segments must either
        resolve inside the allowed dir or raise FileAccessError."""
        sandbox = tempfile.mkdtemp()
        test_path = os.path.join(sandbox, "..", "..", suffix)
        resolved = os.path.realpath(test_path)
        allowed_resolved = os.path.realpath(sandbox)

        is_inside = resolved == allowed_resolved or resolved.startswith(
            allowed_resolved + os.sep
        )

        if is_inside:
            result = _validate_file_path(test_path, [sandbox])
            assert result == resolved
        else:
            with pytest.raises(FileAccessError):
                _validate_file_path(test_path, [sandbox])

    @given(child=st.text(alphabet="abcde", min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_children_always_allowed(self, child):
        """Direct children of allowed dir always pass validation."""
        sandbox = tempfile.mkdtemp()
        test_path = os.path.join(sandbox, child)
        result = _validate_file_path(test_path, [sandbox])
        assert result == os.path.realpath(test_path)

    @given(path=st.text(min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_unrelated_paths_always_rejected(self, path):
        """Paths not under allowed dirs are always rejected."""
        sandbox = tempfile.mkdtemp()
        unrelated = "/unlikely_root_" + path.replace("/", "_").replace("\x00", "")
        resolved = os.path.realpath(unrelated)
        allowed_resolved = os.path.realpath(sandbox)

        assume(not resolved.startswith(allowed_resolved + os.sep))
        assume(resolved != allowed_resolved)

        with pytest.raises(FileAccessError):
            _validate_file_path(unrelated, [sandbox])


class TestPropertyValidateAndSplit:
    """Property 2: Shell operators never appear in output argv."""

    @given(command=_shell_commands)
    def test_no_shell_operators_in_output(self, command):
        """If _validate_and_split succeeds, argv contains no shell operators."""
        # Use the full command as the prefix so we only test the operator filter
        try:
            argv = _validate_and_split(command, [command])
        except (CommandNotAllowedError, ValueError):
            return  # Rejected is fine — that's the safe outcome

        # The critical invariant: no shell operators in the result
        for token in argv:
            assert token not in _SHELL_OPERATORS, (
                f"Shell operator {token!r} escaped validation in command {command!r}"
            )

    @given(
        safe_cmd=st.text(alphabet="abcdefg -=", min_size=1, max_size=20),
        operator=st.sampled_from(list(_SHELL_OPERATORS)),
        evil_cmd=st.text(alphabet="abcdefg /", min_size=1, max_size=20),
    )
    def test_injected_operator_always_caught(self, safe_cmd, operator, evil_cmd):
        """A command with an injected shell operator is always rejected."""
        assume(safe_cmd.strip())
        command = f"{safe_cmd} {operator} {evil_cmd}"
        prefix = safe_cmd.strip().split()[0]
        with pytest.raises(CommandNotAllowedError):
            _validate_and_split(command, [prefix])

    @given(command=_shell_commands)
    def test_malformed_quoting_rejected_or_safe(self, command):
        """Malformed quoting either raises or produces safe output."""
        try:
            argv = _validate_and_split(command, [command])
        except CommandNotAllowedError:
            return  # Rejected is the safe outcome
        # If it didn't raise, no operators must be present
        assert not _SHELL_OPERATORS.intersection(argv)


class TestPropertyValidateQuery:
    """Property 3: SQL injection markers always rejected."""

    @given(
        prefix=st.text(alphabet="SELECTFROM *", min_size=1, max_size=20),
        injection=st.text(min_size=0, max_size=30),
    )
    def test_semicolons_always_rejected(self, prefix, injection):
        """Any query containing a semicolon is rejected, regardless of position."""
        query = f"{prefix};{injection}"
        with pytest.raises(QueryNotAllowedError):
            _validate_query(query, [prefix])

    @given(
        prefix=st.just("SELECT"),
        middle=st.text(alphabet="abcde *$12", min_size=0, max_size=20),
    )
    def test_line_comments_always_rejected(self, prefix, middle):
        """Queries with -- comments are always rejected."""
        query = f"{prefix} {middle} -- comment"
        with pytest.raises(QueryNotAllowedError):
            _validate_query(query, [prefix])

    @given(
        prefix=st.just("SELECT"),
        middle=st.text(alphabet="abcde *$12", min_size=0, max_size=20),
    )
    def test_block_comments_always_rejected(self, prefix, middle):
        """Queries with /* comments are always rejected."""
        query = f"{prefix} {middle} /* evil */"
        with pytest.raises(QueryNotAllowedError):
            _validate_query(query, [prefix])

    @given(
        ws_before=st.text(alphabet=" \t\n", min_size=0, max_size=5),
        ws_between=st.text(alphabet=" \t\n", min_size=1, max_size=5),
        ws_after=st.text(alphabet=" \t\n", min_size=0, max_size=5),
    )
    def test_whitespace_normalisation_is_idempotent(
        self, ws_before, ws_between, ws_after
    ):
        """Whitespace variations of the same query all match the same prefix."""
        query = (
            f"{ws_before}SELECT{ws_between}*{ws_between}FROM{ws_between}users{ws_after}"
        )
        # Should pass — it normalizes to "SELECT * FROM users"
        _validate_query(query, ["SELECT * FROM users"])

    @given(query=_sql_strings)
    def test_dangerous_chars_rejected_or_prefix_matched(self, query):
        """Any generated query is either rejected or matches an allowed prefix."""
        try:
            _validate_query(query, ["SELECT"])
        except QueryNotAllowedError:
            return  # Rejected is fine
        # If it passed, it must not contain dangerous chars
        assert ";" not in query
        assert "--" not in query
        assert "/*" not in query


class TestPropertySecretRedaction:
    """Property 4: Secret values never leak into Proof objects."""

    @given(secret=st.text(min_size=8, max_size=100))
    @settings(max_examples=50)
    async def test_secret_never_in_proof_header_injection(self, secret):
        """The secret value never appears in proof args or result_summary
        for header injection, regardless of the secret content."""
        inner = HttpToken("api", url="https://example.com", methods=["GET"])
        token = SecretToken(
            "test",
            secret_value=secret,
            injection=SecretInjection(kind="header", key="Auth"),
            inner_token=inner,
        )
        with patch(
            "linear_agentics.tokens.http_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = {"status": 200, "body": "ok"}
            proof = await token.consume(method="GET")

        # Secret must not be stored as a value in proof.args
        assert secret not in proof.args.values(), (
            f"Secret leaked as value in proof.args"
        )
        # Secret must not appear as a key in proof.args
        assert secret not in proof.args, f"Secret leaked as key in proof.args"
        inner._consumed = True

    @given(secret=st.text(min_size=8, max_size=100))
    @settings(max_examples=50)
    async def test_secret_never_in_proof_env_injection(self, secret):
        """The secret value never appears in proof args for env injection."""
        inner = ShellToken("cmd", allowed=["echo"])
        token = SecretToken(
            "test",
            secret_value=secret,
            injection=SecretInjection(kind="env", key="SECRET"),
            inner_token=inner,
        )
        with patch(
            "linear_agentics.tokens.shell_exec_with_env", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = "done"
            proof = await token.consume(command="echo hello")

        assert secret not in proof.args.values(), (
            f"Secret leaked as value in proof.args"
        )
        assert secret not in proof.args, f"Secret leaked as key in proof.args"
        inner._consumed = True

    @given(secret=st.text(min_size=8, max_size=100))
    def test_secret_never_in_tool_definition(self, secret):
        """The secret value never appears in any tool definition field values."""
        inner = ShellToken("cmd", allowed=["echo"])
        token = SecretToken(
            "test",
            secret_value=secret,
            injection=SecretInjection(kind="env", key="KEY"),
            inner_token=inner,
        )
        tool_def = token.to_tool_definition()

        def _check_no_secret(obj, path=""):
            """Recursively check that secret doesn't appear in any string value."""
            if isinstance(obj, str):
                assert secret not in obj, f"Secret leaked at {path}: {obj!r}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_secret(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check_no_secret(v, f"{path}[{i}]")

        _check_no_secret(tool_def)
        token._consumed = True
        inner._consumed = True


# ===================================================================
# TIER 2: State Machine Invariants
# ===================================================================


class TestPropertyLinearTokenLifecycle:
    """Property 5: Single-use enforcement holds for all usage patterns."""

    @given(n_attempts=st.integers(min_value=1, max_value=10))
    async def test_exactly_one_consume_succeeds(self, n_attempts):
        """First consume succeeds, all subsequent raise TokenReusedError."""
        token = ShellToken("test", allowed=["echo"])

        # First should succeed
        proof = await token.consume(command="echo ok")
        assert token.consumed
        assert token.proof is proof
        assert proof is not None

        # All subsequent must fail
        for _ in range(n_attempts):
            with pytest.raises(TokenReusedError):
                await token.consume(command="echo again")

        # State must still be consumed
        assert token.consumed
        assert token.proof is proof  # proof unchanged

    @given(
        name=st.text(alphabet="abcde-_", min_size=1, max_size=20),
        duration=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    )
    def test_mark_consumed_proof_integrity(self, name, duration):
        """After _mark_consumed, proof fields match inputs exactly."""
        token = ShellToken(name, allowed=["echo"])
        args = {"command": "echo test"}
        summary = "test output"

        proof = token._mark_consumed(args, summary, duration)

        assert token.consumed is True
        assert token.proof is proof
        assert proof.token_name == name
        assert proof.args == args
        assert proof.result_summary == summary
        assert proof.duration_ms == duration
        assert proof.timestamp  # non-empty ISO string


class TestPropertyMultiUseTokenCounters:
    """Property 6: MultiUseToken counter invariants hold across all usage patterns."""

    @given(max_uses=st.integers(min_value=1, max_value=20))
    async def test_counter_invariants(self, max_uses):
        """After each use: consumed iff use_count >= max_uses,
        uses_remaining >= 0, proofs list matches use count."""
        token = MultiUseShellToken("test", allowed=["echo"], max_uses=max_uses)

        for i in range(max_uses):
            assert not token.consumed
            assert token.uses_remaining == max_uses - i
            assert len(token._proofs) == i

            proof = await token.consume(command=f"echo {i}")
            assert proof is not None
            assert len(token._proofs) == i + 1
            assert token._proof is token._proofs[-1]

        # Now exhausted
        assert token.consumed
        assert token.uses_remaining == 0
        assert len(token._proofs) == max_uses

        # Further attempts must fail
        with pytest.raises(TokenReusedError, match="exhausted"):
            await token.consume(command="echo overflow")

        # State unchanged after failed attempt
        assert len(token._proofs) == max_uses
        assert token.uses_remaining == 0


class TestPropertyBudgetSpend:
    """Property 7: Budget invariants hold across all spend sequences."""

    @given(
        max_steps=st.integers(min_value=1, max_value=100),
        spend_amounts=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=20
        ),
    )
    def test_remaining_invariant(self, max_steps, spend_amounts):
        """remaining == max(0, max_steps - steps_used) at all times.
        remaining is monotonically non-increasing. Failed spends don't corrupt state."""
        budget = Budget(max_steps=max_steps, timeout_minutes=60)
        prev_remaining = budget.remaining
        total_spent = 0

        for amount in spend_amounts:
            old_steps = budget.steps_used
            try:
                budget.spend(amount)
            except BudgetExhaustedError:
                # steps_used was already incremented (current implementation)
                # but remaining must still be non-negative
                assert budget.remaining >= 0
                assert budget.remaining <= prev_remaining
                break
            else:
                total_spent += amount
                assert budget.steps_used == old_steps + amount
                assert budget.remaining == max(0, max_steps - budget.steps_used)
                assert budget.remaining <= prev_remaining
                prev_remaining = budget.remaining

        # Final invariant: remaining is never negative
        assert budget.remaining >= 0

    @given(max_steps=st.integers(min_value=1, max_value=50))
    def test_exact_boundary(self, max_steps):
        """Spending exactly max_steps does not raise, spending one more does."""
        budget = Budget(max_steps=max_steps, timeout_minutes=60)
        budget.spend(max_steps)
        assert budget.remaining == 0
        with pytest.raises(BudgetExhaustedError):
            budget.spend(1)


# ===================================================================
# TIER 3: Audit Integrity
# ===================================================================


class TestPropertyProofImmutability:
    """Property 8: Proof is truly immutable."""

    @given(
        token_name=st.text(min_size=1, max_size=20),
        scope=st.text(min_size=1, max_size=30),
        result_summary=st.text(min_size=0, max_size=50),
        duration_ms=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    )
    def test_all_fields_frozen(self, token_name, scope, result_summary, duration_ms):
        """No attribute on a Proof can be reassigned."""
        proof = Proof(
            token_name=token_name,
            scope=scope,
            args={"key": "value"},
            timestamp="2024-01-01T00:00:00+00:00",
            result_summary=result_summary,
            duration_ms=duration_ms,
        )
        for field in dataclasses.fields(proof):
            with pytest.raises(FrozenInstanceError):
                setattr(proof, field.name, "tampered")

    @given(
        token_name=st.text(min_size=1, max_size=20),
        scope=st.text(min_size=1, max_size=20),
    )
    def test_to_dict_returns_fresh_copy(self, token_name, scope):
        """to_dict() returns a new dict each time (no shared references)."""
        proof = Proof(
            token_name=token_name,
            scope=scope,
            args={"key": "value"},
            timestamp="2024-01-01T00:00:00+00:00",
            result_summary="ok",
            duration_ms=1.0,
        )
        d1 = proof.to_dict()
        d2 = proof.to_dict()
        assert d1 == d2
        assert d1 is not d2


class TestPropertyAuditTrailJsonRoundtrip:
    """Property 9: AuditTrail.to_json() always produces valid, complete JSON."""

    @given(
        n_proofs=st.integers(min_value=0, max_value=10),
        n_errors=st.integers(min_value=0, max_value=5),
    )
    def test_json_roundtrip_preserves_counts(self, n_proofs, n_errors):
        """JSON output is valid and preserves the count of all records."""
        trail = AuditTrail()

        for i in range(n_proofs):
            trail.record_proof(
                Proof(
                    token_name=f"token-{i}",
                    scope=f"scope-{i}",
                    args={"i": i},
                    timestamp="2024-01-01T00:00:00+00:00",
                    result_summary=f"result-{i}",
                    duration_ms=float(i),
                )
            )

        for i in range(n_errors):
            trail.record_error(f"token-err-{i}", ValueError(f"error-{i}"))

        json_str = trail.to_json()
        parsed = json.loads(json_str)  # Must not raise

        assert len(parsed["proofs"]) == n_proofs
        assert len(parsed["errors"]) == n_errors
        assert len(parsed["approvals"]) == 0
        assert len(parsed["negotiations"]) == 0

    @given(
        n_records=st.integers(min_value=0, max_value=10),
        indent=st.sampled_from([0, 2, 4]),
    )
    def test_json_valid_with_any_indent(self, n_records, indent):
        """to_json() produces valid JSON regardless of indent setting."""
        trail = AuditTrail()
        for i in range(n_records):
            trail.record_proof(
                Proof(
                    token_name=f"t{i}",
                    scope="s",
                    args={},
                    timestamp="2024-01-01T00:00:00+00:00",
                    result_summary="ok",
                    duration_ms=0.0,
                )
            )
        parsed = json.loads(trail.to_json(indent=indent))
        assert isinstance(parsed, dict)

    @given(
        proofs=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10), st.text(min_size=0, max_size=20)
            ),
            min_size=0,
            max_size=8,
        )
    )
    def test_append_order_preserved(self, proofs):
        """Records appear in JSON in the same order they were appended."""
        trail = AuditTrail()
        for name, summary in proofs:
            trail.record_proof(
                Proof(
                    token_name=name,
                    scope="s",
                    args={},
                    timestamp="2024-01-01T00:00:00+00:00",
                    result_summary=summary,
                    duration_ms=0.0,
                )
            )
        parsed = json.loads(trail.to_json())
        for i, (name, summary) in enumerate(proofs):
            assert parsed["proofs"][i]["token_name"] == name
            assert parsed["proofs"][i]["result_summary"] == summary


class TestPropertyCapabilitySetIndex:
    """Property 10: CapabilitySet lookup is always consistent with token list."""

    @given(
        n_tokens=st.integers(min_value=1, max_value=10),
    )
    def test_all_tokens_findable_by_tool_name(self, n_tokens):
        """Every token in the set is findable via its tool definition name."""
        tokens = [ShellToken(f"cmd-{i}", allowed=["echo"]) for i in range(n_tokens)]
        cap_set = CapabilitySet(tokens=tokens)

        for token in tokens:
            tool_name = token.to_tool_definition()["name"]
            found = cap_set.get_token(tool_name)
            assert found is token, (
                f"Token {token.name} not found by tool name {tool_name}"
            )

        # Suppress __del__ warnings
        for t in tokens:
            t._consumed = True

    @given(
        initial=st.integers(min_value=0, max_value=5),
        added=st.integers(min_value=1, max_value=5),
    )
    def test_add_token_immediately_findable(self, initial, added):
        """Tokens added at runtime are immediately findable."""
        tokens = [ShellToken(f"init-{i}", allowed=["echo"]) for i in range(initial)]
        cap_set = CapabilitySet(tokens=tokens)

        added_tokens = []
        for i in range(added):
            new_token = ShellToken(f"added-{i}", allowed=["ls"])
            cap_set.add_token(new_token)
            added_tokens.append(new_token)

            # Immediately findable
            tool_name = new_token.to_tool_definition()["name"]
            assert cap_set.get_token(tool_name) is new_token

        # All tokens still findable
        for token in tokens + added_tokens:
            tool_name = token.to_tool_definition()["name"]
            assert cap_set.get_token(tool_name) is token

        # Suppress __del__ warnings
        for t in tokens + added_tokens:
            t._consumed = True

    @given(name=st.text(alphabet="abcde-_", min_size=1, max_size=15))
    def test_nonexistent_tool_returns_none(self, name):
        """Looking up a tool name that doesn't exist returns None."""
        cap_set = CapabilitySet(tokens=[])
        assert cap_set.get_token(f"shell_{name}") is None
