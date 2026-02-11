"""Tests for the agent executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linear_agentics.agent import Agent, AgentResult, CapabilitySet
from linear_agentics.budget import Budget
from linear_agentics.tokens import ShellToken


class TestCapabilitySet:
    def test_get_token_by_tool_name(self):
        t1 = ShellToken("read-logs", allowed=["echo"])
        t2 = ShellToken("list-pods", allowed=["echo"])
        caps = CapabilitySet([t1, t2])

        assert caps.get_token("shell_read-logs") is t1
        assert caps.get_token("shell_list-pods") is t2
        assert caps.get_token("nonexistent") is None
        # cleanup
        t1._consumed = True
        t2._consumed = True

    def test_to_tool_definitions(self):
        t1 = ShellToken("t1", allowed=["echo"])
        caps = CapabilitySet([t1])
        defs = caps.to_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "shell_t1"
        t1._consumed = True

    def test_unused_tokens(self):
        t1 = ShellToken("used", allowed=["echo"])
        t2 = ShellToken("unused", allowed=["echo"])
        t1._consumed = True
        caps = CapabilitySet([t1, t2])
        unused = caps.unused_tokens
        assert len(unused) == 1
        assert unused[0] is t2
        t2._consumed = True


def _make_text_response(text: str):
    """Create a mock Anthropic response with just text."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_tool_response(tool_name: str, tool_input: dict, tool_id: str = "call_1"):
    """Create a mock Anthropic response with a tool use."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response = MagicMock()
    response.content = [block]
    return response


class TestAgent:
    async def test_simple_text_response(self):
        """Agent returns immediately when LLM gives text-only response."""
        t1 = ShellToken("t1", allowed=["echo"])
        caps = CapabilitySet([t1])
        budget = Budget(max_steps=5, timeout_minutes=5)

        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        mock_response = _make_text_response("All done!")

        with patch.object(
            agent._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert result.final_message == "All done!"
        assert result.steps_taken == 1
        assert "t1" in result.tokens_unused
        t1._consumed = True

    async def test_tool_call_consumes_token(self):
        """Agent correctly routes tool calls to tokens."""
        t1 = ShellToken("echo-cmd", allowed=["echo"])
        caps = CapabilitySet([t1])
        budget = Budget(max_steps=5, timeout_minutes=5)

        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        tool_response = _make_tool_response("shell_echo-cmd", {"command": "echo hello"})
        text_response = _make_text_response("Done.")

        with patch.object(
            agent._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [tool_response, text_response]
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert result.steps_taken == 2
        assert len(result.audit_trail.proofs) == 1
        assert result.audit_trail.proofs[0].token_name == "echo-cmd"
        assert result.tokens_unused == []

    async def test_reuse_returns_error_to_llm(self):
        """Second use of same token returns error message, doesn't crash."""
        t1 = ShellToken("once", allowed=["echo"])
        caps = CapabilitySet([t1])
        budget = Budget(max_steps=10, timeout_minutes=5)

        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        call1 = _make_tool_response("shell_once", {"command": "echo 1"}, "c1")
        call2 = _make_tool_response("shell_once", {"command": "echo 2"}, "c2")
        done = _make_text_response("Finished.")

        with patch.object(
            agent._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [call1, call2, done]
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert len(result.audit_trail.proofs) == 1
        assert len(result.audit_trail.errors) == 1
        assert "already consumed" in result.audit_trail.errors[0].message

    async def test_budget_exhaustion_stops_agent(self):
        """Agent stops when budget is exhausted."""
        t1 = ShellToken("t1", allowed=["echo"])
        caps = CapabilitySet([t1])
        budget = Budget(max_steps=1, timeout_minutes=5)

        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        tool_response = _make_tool_response("shell_t1", {"command": "echo hi"})

        with patch.object(
            agent._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = tool_response
            result = await agent.run()

        assert result.stop_reason == "budget_exhausted"

    async def test_unknown_tool_returns_error(self):
        """Unknown tool name returns error to LLM."""
        t1 = ShellToken("t1", allowed=["echo"])
        caps = CapabilitySet([t1])
        budget = Budget(max_steps=5, timeout_minutes=5)

        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        call = _make_tool_response("nonexistent_tool", {})
        done = _make_text_response("Ok.")

        with patch.object(
            agent._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [call, done]
            result = await agent.run()

        assert result.stop_reason == "complete"
        t1._consumed = True
