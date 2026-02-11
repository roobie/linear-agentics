"""LLM Agent executor with capability-bounded tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic

from .approval import ApprovalDeniedError, ApprovalGate
from .audit import AuditTrail, NegotiationRecord, Proof, _now_iso
from .budget import Budget, BudgetExhaustedError, BudgetTimeoutError
from .negotiation import CapabilityProvider
from .tokens import LinearToken, TokenReusedError, TokenScopeError


@dataclass
class CapabilitySet:
    """A named collection of capability tokens available to an agent."""

    tokens: list[LinearToken]

    def __post_init__(self) -> None:
        self._by_tool_name: dict[str, LinearToken] = {}
        for token in self.tokens:
            tool_def = token.to_tool_definition()
            self._by_tool_name[tool_def["name"]] = token

    def get_token(self, tool_name: str) -> LinearToken | None:
        return self._by_tool_name.get(tool_name)

    def add_token(self, token: LinearToken) -> None:
        """Add a token to the capability set at runtime."""
        self.tokens.append(token)
        tool_def = token.to_tool_definition()
        self._by_tool_name[tool_def["name"]] = token

    def to_tool_definitions(self) -> list[dict]:
        return [t.to_tool_definition() for t in self.tokens]

    @property
    def unused_tokens(self) -> list[LinearToken]:
        return [t for t in self.tokens if not t.consumed]


@dataclass
class AgentResult:
    """Result of an agent run, containing audit trail and metadata."""

    audit_trail: AuditTrail
    tokens_unused: list[str]
    budget_remaining: int
    steps_taken: int
    final_message: str
    stop_reason: str  # "complete", "budget_exhausted", "timeout", "error"

    def to_dict(self) -> dict:
        return {
            "stop_reason": self.stop_reason,
            "steps_taken": self.steps_taken,
            "budget_remaining": self.budget_remaining,
            "tokens_unused": self.tokens_unused,
            "final_message": self.final_message,
            "audit_trail": json.loads(self.audit_trail.to_json()),
        }


class Agent:
    """LLM agent that operates within capability bounds.

    Converts tokens to LLM tool definitions, runs a tool-calling loop,
    and enforces token consumption, approval gates, and budget limits.
    """

    def __init__(
        self,
        capabilities: CapabilitySet,
        budget: Budget,
        llm_model: str = "claude-sonnet-4-5-20250929",
        system_prompt: str = "",
        approval_gate: ApprovalGate | None = None,
        capability_provider: CapabilityProvider | None = None,
        candidate_tokens: list[LinearToken] | None = None,
        max_negotiations: int = 3,
    ) -> None:
        self.capabilities = capabilities
        self.budget = budget
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.approval_gate = approval_gate or ApprovalGate()
        self.capability_provider = capability_provider
        self.candidate_tokens = candidate_tokens or []
        self.max_negotiations = max_negotiations
        self._negotiations_used = 0
        self._audit = AuditTrail()
        self._client = anthropic.AsyncAnthropic()

    async def run(self) -> AgentResult:
        self.budget.start()

        tools = self.capabilities.to_tool_definitions()

        # Add the request_approval meta-tool
        tools.append(
            {
                "name": "request_approval",
                "description": "Ask the human operator for approval before proceeding with a sensitive action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message explaining what you want to do and why",
                        }
                    },
                    "required": ["message"],
                },
            }
        )

        if self.capability_provider and self.candidate_tokens:
            tools.append(
                {
                    "name": "request_capability",
                    "description": (
                        "Request an additional capability you don't currently have. "
                        f"{self.max_negotiations} requests allowed per run. "
                        "Provide the scope you need and justify why."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "description": "The capability scope needed",
                            },
                            "justification": {
                                "type": "string",
                                "description": "Why you need this capability",
                            },
                        },
                        "required": ["scope", "justification"],
                    },
                }
            )

        messages: list[dict] = []
        final_message = ""
        stop_reason = "complete"

        try:
            while True:
                self.budget.check()

                response = await self._client.messages.create(
                    model=self.llm_model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=tools,
                    messages=messages,
                )

                self.budget.spend()

                # Collect text and tool uses from response
                text_parts = []
                tool_uses = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                if text_parts:
                    final_message = "\n".join(text_parts)

                # If no tool calls, agent is done
                if not tool_uses:
                    break

                # Add assistant message to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Process each tool call
                tool_results = []
                for tool_use in tool_uses:
                    result_text = await self._handle_tool_call(
                        tool_use.name, tool_use.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

        except BudgetExhaustedError:
            stop_reason = "budget_exhausted"
            final_message = "Agent stopped: budget exhausted."
        except BudgetTimeoutError:
            stop_reason = "timeout"
            final_message = "Agent stopped: timeout."
        except Exception as e:
            stop_reason = "error"
            final_message = f"Agent stopped with error: {e}"
            self._audit.record_error("agent", e)

        unused = [t.name for t in self.capabilities.unused_tokens]

        return AgentResult(
            audit_trail=self._audit,
            tokens_unused=unused,
            budget_remaining=self.budget.remaining,
            steps_taken=self.budget.steps_used,
            final_message=final_message,
            stop_reason=stop_reason,
        )

    async def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call against the capability set."""

        # Handle the meta request_approval tool
        if tool_name == "request_approval":
            message = tool_input.get("message", "Approval requested")
            record = await self.approval_gate.prompt_cli("agent_request", message)
            self._audit.record_approval(record)
            if record.approved:
                return "Approval granted. You may proceed."
            else:
                return "Approval denied. Do not proceed with this action."

        if tool_name == "request_capability":
            return await self._handle_capability_request(tool_input)

        token = self.capabilities.get_token(tool_name)
        if token is None:
            return f"Error: Unknown tool '{tool_name}'. Not in capability set."

        # Check approval requirement
        if token.requires_approval:
            record = await self.approval_gate.prompt_cli(
                token.name,
                f"Agent wants to use capability '{token.name}' (scope: {token.scope}). Allow?",
            )
            self._audit.record_approval(record)
            if not record.approved:
                self._audit.record_error(
                    token.name, ApprovalDeniedError("Human denied approval")
                )
                return f"Error: Approval denied for '{token.name}'."

        try:
            proof = await token.consume(**tool_input)
            self._audit.record_proof(proof)
            return proof.result_summary
        except TokenReusedError as e:
            self._audit.record_error(token.name, e)
            return f"Error: {e}"
        except Exception as e:
            self._audit.record_error(token.name, e)
            return f"Error executing '{tool_name}': {e}"

    async def _handle_capability_request(self, tool_input: dict) -> str:
        """Handle a request_capability meta-tool call."""
        scope = tool_input.get("scope", "")
        justification = tool_input.get("justification", "")

        def _make_record(granted_name: str | None) -> NegotiationRecord:
            return NegotiationRecord(
                requested_scope=scope,
                justification=justification,
                granted_token=granted_name,
                provider_type=type(self.capability_provider).__name__,
                timestamp=_now_iso(),
            )

        if not self.capability_provider:
            self._audit.record_negotiation(_make_record(None))
            return "Error: Capability negotiation not configured."

        if self._negotiations_used >= self.max_negotiations:
            self._audit.record_negotiation(_make_record(None))
            return (
                f"Error: Maximum negotiations ({self.max_negotiations}) exhausted. "
                "Work with your existing capabilities."
            )

        self._negotiations_used += 1

        granted = await self.capability_provider.request_capability(
            requested_scope=scope,
            justification=justification,
            candidates=self.candidate_tokens,
        )

        if granted is None:
            self._audit.record_negotiation(_make_record(None))
            return "Capability request denied. Work with your existing capabilities."

        # Add granted token to the live capability set
        self.capabilities.add_token(granted)
        self._audit.record_negotiation(_make_record(granted.name))
        return (
            f"Capability granted: {granted.name} ({granted.scope}). You can now use it."
        )
