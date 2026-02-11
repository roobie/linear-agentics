#!/usr/bin/env python3
"""Example: Agent with capability negotiation.

The agent starts with read-only capabilities but can request
additional capabilities (like HTTP POST) from a human operator
if it discovers it needs them during execution.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python examples/capability_negotiation.py
"""

import asyncio

from linear_agentics import (
    Agent,
    Budget,
    CapabilitySet,
    HumanCapabilityProvider,
    HttpToken,
    ShellToken,
)
from linear_agentics.tokens import MultiUseShellToken


async def main():
    # Initial capabilities — read-only diagnostics
    capabilities = CapabilitySet([
        MultiUseShellToken(
            "read-logs",
            allowed=["kubectl logs", "kubectl get pods"],
            max_uses=3,
        ),
        HttpToken(
            "check-health",
            url="https://api.internal/health",
            methods=["GET"],
        ),
    ])

    # Candidate tokens — can be granted during execution if the agent requests them
    candidates = [
        HttpToken(
            "create-incident",
            url="https://api.internal/incidents",
            methods=["POST"],
            requires_approval=True,
        ),
        HttpToken(
            "update-status",
            url="https://api.internal/status",
            methods=["PUT"],
        ),
        # Same type as initial — agent can request another health check
        HttpToken(
            "check-metrics",
            url="https://api.internal/metrics",
            methods=["GET"],
        ),
    ]

    provider = HumanCapabilityProvider(timeout_seconds=120)
    budget = Budget(max_steps=15, timeout_minutes=10)

    agent = Agent(
        capabilities=capabilities,
        budget=budget,
        llm_model="claude-sonnet-4-5-20250929",
        system_prompt=(
            "You are an SRE monitoring agent. Check system health and logs.\n"
            "If you detect an issue, you may need to create an incident or\n"
            "update the status page. You start with read-only capabilities\n"
            "but can request additional ones if needed.\n\n"
            "Available candidate scopes you can request:\n"
            "- http:POST:https://api.internal/incidents (create incident)\n"
            "- http:PUT:https://api.internal/status (update status page)\n"
            "- http:GET:https://api.internal/metrics (check metrics)\n"
        ),
        capability_provider=provider,
        candidate_tokens=candidates,
        max_negotiations=2,
    )

    print("Starting monitoring agent with capability negotiation...")
    print(f"Initial capabilities: {[t.name for t in capabilities.tokens]}")
    print(f"Candidate tokens: {[t.name for t in candidates]}")
    print(f"Max negotiations: {agent.max_negotiations}")
    print()

    result = await agent.run()

    print("\n" + "=" * 60)
    print("AGENT RESULT")
    print("=" * 60)
    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Negotiations used: {agent._negotiations_used}/{agent.max_negotiations}")
    print(f"Unused tokens: {result.tokens_unused}")
    print()
    print("AUDIT TRAIL:")
    print(result.audit_trail.to_table())
    print()
    print("FULL AUDIT JSON:")
    print(result.audit_trail.to_json())


if __name__ == "__main__":
    asyncio.run(main())
