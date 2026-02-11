"""Tests for capability negotiation protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linear_agentics.agent import Agent, CapabilitySet
from linear_agentics.audit import NegotiationRecord
from linear_agentics.budget import Budget
from linear_agentics.negotiation import HumanCapabilityProvider
from linear_agentics.tokens import HttpToken, ShellToken


# --- Helpers ---


class StubProvider:
    """Test provider that returns a predetermined token."""

    def __init__(self, grant: ShellToken | HttpToken | None = None):
        self._grant = grant
        self.calls: list[dict] = []

    async def request_capability(self, requested_scope, justification, candidates):
        self.calls.append({
            "scope": requested_scope,
            "justification": justification,
            "candidates": candidates,
        })
        return self._grant


def _make_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_tool_response(tool_name: str, tool_input: dict, tool_id: str = "call_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response = MagicMock()
    response.content = [block]
    return response


# --- NegotiationRecord ---


class TestNegotiationRecord:
    def test_to_dict(self):
        rec = NegotiationRecord(
            requested_scope="http:GET:https://example.com",
            justification="need health check",
            granted_token="check-health",
            provider_type="StubProvider",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        d = rec.to_dict()
        assert d["granted_token"] == "check-health"
        assert d["justification"] == "need health check"

    def test_denied_record(self):
        rec = NegotiationRecord(
            requested_scope="shell:rm",
            justification="want to delete",
            granted_token=None,
            provider_type="StubProvider",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert rec.granted_token is None


# --- HumanCapabilityProvider ---


class TestHumanCapabilityProvider:
    async def test_grant_first_candidate(self):
        token = ShellToken("diag", allowed=["echo"])
        provider = HumanCapabilityProvider()

        with patch("linear_agentics.negotiation.input", return_value="1"):
            result = await provider.request_capability(
                "shell:echo", "need diagnostics", [token]
            )

        assert result is token
        token._consumed = True

    async def test_deny_request(self):
        token = ShellToken("diag", allowed=["echo"])
        provider = HumanCapabilityProvider()

        with patch("linear_agentics.negotiation.input", return_value="0"):
            result = await provider.request_capability(
                "shell:echo", "need diagnostics", [token]
            )

        assert result is None
        token._consumed = True

    async def test_invalid_input_returns_none(self):
        token = ShellToken("diag", allowed=["echo"])
        provider = HumanCapabilityProvider()

        with patch("linear_agentics.negotiation.input", return_value="abc"):
            result = await provider.request_capability(
                "shell:echo", "need diagnostics", [token]
            )

        assert result is None
        token._consumed = True

    async def test_no_candidates_returns_none(self):
        provider = HumanCapabilityProvider()
        result = await provider.request_capability("shell:echo", "need it", [])
        assert result is None

    async def test_skips_consumed_candidates(self):
        t1 = ShellToken("used", allowed=["echo"])
        t1._consumed = True
        t2 = ShellToken("fresh", allowed=["echo"])
        provider = HumanCapabilityProvider()

        with patch("linear_agentics.negotiation.input", return_value="1"):
            result = await provider.request_capability(
                "shell:echo", "need it", [t1, t2]
            )

        assert result is t2
        t2._consumed = True


# --- Agent integration ---


class TestAgentNegotiation:
    async def test_request_capability_granted(self):
        """Agent requests a capability, gets it, can use it."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        candidate = ShellToken("ls-cmd", allowed=["ls"])
        provider = StubProvider(grant=candidate)

        caps = CapabilitySet([initial])
        budget = Budget(max_steps=10, timeout_minutes=5)
        agent = Agent(
            capabilities=caps,
            budget=budget,
            system_prompt="test",
            capability_provider=provider,
            candidate_tokens=[candidate],
            max_negotiations=3,
        )

        # LLM requests capability, then uses it, then finishes
        req_call = _make_tool_response(
            "request_capability",
            {"scope": "shell:ls", "justification": "need to list files"},
            "c1",
        )
        use_call = _make_tool_response(
            "shell_ls-cmd", {"command": "ls"}, "c2"
        )
        done = _make_text_response("Done.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [req_call, use_call, done]
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert len(result.audit_trail.negotiations) == 1
        assert result.audit_trail.negotiations[0].granted_token == "ls-cmd"
        assert len(result.audit_trail.proofs) == 1
        assert result.audit_trail.proofs[0].token_name == "ls-cmd"

    async def test_request_capability_denied(self):
        """Agent requests capability, gets denied, continues."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        candidate = ShellToken("ls-cmd", allowed=["ls"])
        provider = StubProvider(grant=None)

        caps = CapabilitySet([initial])
        budget = Budget(max_steps=10, timeout_minutes=5)
        agent = Agent(
            capabilities=caps,
            budget=budget,
            system_prompt="test",
            capability_provider=provider,
            candidate_tokens=[candidate],
            max_negotiations=3,
        )

        req_call = _make_tool_response(
            "request_capability",
            {"scope": "shell:ls", "justification": "need to list files"},
        )
        done = _make_text_response("Ok, working without it.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [req_call, done]
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert len(result.audit_trail.negotiations) == 1
        assert result.audit_trail.negotiations[0].granted_token is None
        candidate._consumed = True

    async def test_max_negotiations_enforced(self):
        """Agent can't exceed max_negotiations."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        c1 = ShellToken("c1", allowed=["ls"])
        c2 = ShellToken("c2", allowed=["cat"])
        provider = StubProvider(grant=c1)

        caps = CapabilitySet([initial])
        budget = Budget(max_steps=10, timeout_minutes=5)
        agent = Agent(
            capabilities=caps,
            budget=budget,
            system_prompt="test",
            capability_provider=provider,
            candidate_tokens=[c1, c2],
            max_negotiations=1,
        )

        req1 = _make_tool_response(
            "request_capability",
            {"scope": "shell:ls", "justification": "first request"},
            "c1",
        )
        req2 = _make_tool_response(
            "request_capability",
            {"scope": "shell:cat", "justification": "second request"},
            "c2",
        )
        done = _make_text_response("Done.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [req1, req2, done]
            result = await agent.run()

        assert result.stop_reason == "complete"
        assert len(result.audit_trail.negotiations) == 2
        # First was granted
        assert result.audit_trail.negotiations[0].granted_token == "c1"
        # Second was denied (limit reached)
        assert result.audit_trail.negotiations[1].granted_token is None
        c2._consumed = True

    async def test_no_provider_returns_error(self):
        """Without a provider, request_capability returns error."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        caps = CapabilitySet([initial])
        budget = Budget(max_steps=10, timeout_minutes=5)

        # No provider, but we manually call the handler
        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        result = await agent._handle_capability_request({
            "scope": "shell:ls",
            "justification": "need it",
        })

        assert "not configured" in result
        initial._consumed = True

    async def test_negotiation_tool_not_added_without_provider(self):
        """request_capability tool not in tool list when no provider configured."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        caps = CapabilitySet([initial])
        budget = Budget(max_steps=5, timeout_minutes=5)
        agent = Agent(capabilities=caps, budget=budget, system_prompt="test")

        done = _make_text_response("Hello.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = done
            await agent.run()

        # Check tools passed to LLM
        call_args = mock_create.call_args
        tools = call_args.kwargs.get("tools", call_args[1].get("tools", []))
        tool_names = [t["name"] for t in tools]
        assert "request_capability" not in tool_names
        initial._consumed = True

    async def test_negotiation_tool_added_with_provider(self):
        """request_capability tool appears when provider is configured."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        candidate = ShellToken("ls-cmd", allowed=["ls"])
        provider = StubProvider()

        caps = CapabilitySet([initial])
        budget = Budget(max_steps=5, timeout_minutes=5)
        agent = Agent(
            capabilities=caps,
            budget=budget,
            system_prompt="test",
            capability_provider=provider,
            candidate_tokens=[candidate],
        )

        done = _make_text_response("Hello.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = done
            await agent.run()

        call_args = mock_create.call_args
        tools = call_args.kwargs.get("tools", call_args[1].get("tools", []))
        tool_names = [t["name"] for t in tools]
        assert "request_capability" in tool_names
        initial._consumed = True
        candidate._consumed = True

    async def test_audit_trail_includes_negotiations_in_json(self):
        """Negotiations appear in audit trail JSON output."""
        initial = ShellToken("echo-cmd", allowed=["echo"])
        candidate = ShellToken("ls-cmd", allowed=["ls"])
        provider = StubProvider(grant=candidate)

        caps = CapabilitySet([initial])
        budget = Budget(max_steps=10, timeout_minutes=5)
        agent = Agent(
            capabilities=caps,
            budget=budget,
            system_prompt="test",
            capability_provider=provider,
            candidate_tokens=[candidate],
        )

        req_call = _make_tool_response(
            "request_capability",
            {"scope": "shell:ls", "justification": "need listing"},
        )
        done = _make_text_response("Done.")

        with patch.object(agent._client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [req_call, done]
            result = await agent.run()

        import json
        trail = json.loads(result.audit_trail.to_json())
        assert "negotiations" in trail
        assert len(trail["negotiations"]) == 1
        assert trail["negotiations"][0]["granted_token"] == "ls-cmd"


class TestCapabilitySetAddToken:
    def test_add_token_at_runtime(self):
        t1 = ShellToken("initial", allowed=["echo"])
        caps = CapabilitySet([t1])

        assert caps.get_token("shell_new") is None

        t2 = ShellToken("new", allowed=["ls"])
        caps.add_token(t2)

        assert caps.get_token("shell_new") is t2
        assert len(caps.tokens) == 2
        t1._consumed = True
        t2._consumed = True
