"""Tests for linear capability tokens."""

import asyncio
import warnings

import pytest

from linear_agentics.tokens import (
    DeployToken,
    HttpToken,
    LinearToken,
    MultiUseShellToken,
    ShellToken,
    TokenReusedError,
)
from linear_agentics.actions import CommandNotAllowedError


class TestShellToken:
    @pytest.fixture
    def echo_token(self):
        return ShellToken("echo-test", allowed=["echo"])

    async def test_consume_allowed_command(self, echo_token):
        proof = await echo_token.consume(command="echo hello")
        assert proof.token_name == "echo-test"
        assert "hello" in proof.result_summary
        assert proof.duration_ms >= 0

    async def test_consume_marks_token_consumed(self, echo_token):
        assert not echo_token.consumed
        await echo_token.consume(command="echo test")
        assert echo_token.consumed

    async def test_reuse_raises(self, echo_token):
        await echo_token.consume(command="echo first")
        with pytest.raises(TokenReusedError, match="already consumed"):
            await echo_token.consume(command="echo second")

    async def test_disallowed_command_raises(self, echo_token):
        with pytest.raises(CommandNotAllowedError):
            await echo_token.consume(command="rm -rf /")

    async def test_disallowed_command_composition_raises(self, echo_token):
        with pytest.raises(CommandNotAllowedError):
            await echo_token.consume(command="echo -n '' || rm -rf /")

    async def test_proof_recorded(self, echo_token):
        proof = await echo_token.consume(command="echo proof-test")
        assert echo_token.proof is proof
        assert proof.args == {"command": "echo proof-test"}

    def test_to_tool_definition(self, echo_token):
        tool_def = echo_token.to_tool_definition()
        assert tool_def["name"] == "shell_echo-test"
        assert "ONE USE ONLY" in tool_def["description"]
        assert "command" in tool_def["input_schema"]["properties"]

    def test_to_tool_definition_with_approval(self):
        token = ShellToken("sensitive", allowed=["ls"], requires_approval=True)
        tool_def = token.to_tool_definition()
        assert "REQUIRES APPROVAL" in tool_def["description"]
        # prevent __del__ warning
        token._consumed = True

    def test_unused_token_warns(self):
        token = ShellToken("temp", allowed=["echo"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del token
            assert any("never consumed" in str(warning.message) for warning in w)


class TestHttpToken:
    async def test_to_tool_definition(self):
        token = HttpToken("health", url="https://example.com/health", methods=["GET"])
        tool_def = token.to_tool_definition()
        assert tool_def["name"] == "health"
        assert "GET" in tool_def["description"]
        token._consumed = True

    async def test_reuse_raises(self):
        """HttpToken can't actually make requests in tests, but reuse check is sync."""
        token = HttpToken("test", url="https://example.com", methods=["GET"])
        # Manually mark consumed
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume(method="GET")


class TestDeployToken:
    def test_to_tool_definition(self):
        token = DeployToken(
            "staging",
            method="kubectl",
            target="staging",
            image="app:v1",
            requires_approval=True,
        )
        tool_def = token.to_tool_definition()
        assert tool_def["name"] == "deploy_staging"
        assert "REQUIRES APPROVAL" in tool_def["description"]
        assert "app:v1" in tool_def["description"]
        token._consumed = True

    async def test_reuse_raises(self):
        token = DeployToken("prod", method="kubectl", target="prod", image="app:v1")
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume()


class TestMultiUseShellToken:
    @pytest.fixture
    def multi_echo(self):
        return MultiUseShellToken("multi-echo", allowed=["echo"], max_uses=3)

    async def test_multiple_uses(self, multi_echo):
        proof1 = await multi_echo.consume(command="echo one")
        assert multi_echo.uses_remaining == 2
        assert not multi_echo.consumed

        proof2 = await multi_echo.consume(command="echo two")
        assert multi_echo.uses_remaining == 1

        proof3 = await multi_echo.consume(command="echo three")
        assert multi_echo.uses_remaining == 0
        assert multi_echo.consumed

    async def test_exhausted_raises(self, multi_echo):
        for i in range(3):
            await multi_echo.consume(command=f"echo {i}")
        with pytest.raises(TokenReusedError, match="exhausted"):
            await multi_echo.consume(command="echo four")

    def test_to_tool_definition(self, multi_echo):
        tool_def = multi_echo.to_tool_definition()
        assert "3 uses remaining" in tool_def["description"]
