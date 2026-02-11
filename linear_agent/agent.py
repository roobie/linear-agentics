"""LLM Agent executor with capability-bounded tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic

from .approval import ApprovalDeniedError, ApprovalGate
from .audit import AuditTrail, Proof
from .budget import Budget, BudgetExhaustedError, BudgetTimeoutError
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
    ) -> None:
        self.capabilities = capabilities
        self.budget = budget
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.approval_gate = approval_gate or ApprovalGate()
        self._audit = AuditTrail()
        self._client = anthropic.AsyncAnthropic()

    async def run(self) -> AgentResult:
        self.budget.start()

        tools = self.capabilities.to_tool_definitions()

        # Add the request_approval meta-tool
        tools.append({
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
        })

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
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                    })

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
