#!/usr/bin/env python3
"""Example: Incident triage agent with read-only capabilities.

This demo shows an agent that can ONLY read — no deployments, no mutations.
It reads pod status and logs, checks health, then reports its diagnosis.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python examples/sre_triage.py
"""

import asyncio

from linear_agentics import Agent, Budget, CapabilitySet, HttpToken
from linear_agentics.tokens import MultiUseShellToken


async def main():
    capabilities = CapabilitySet(
        [
            MultiUseShellToken(
                "read-pods",
                allowed=[
                    "kubectl get pods",
                    "kubectl describe pod",
                    "kubectl get events",
                ],
                max_uses=5,
            ),
            MultiUseShellToken(
                "read-logs",
                allowed=["kubectl logs"],
                max_uses=5,
            ),
            HttpToken(
                "check-health",
                url="https://api.internal/health",
                methods=["GET"],
            ),
        ]
    )

    budget = Budget(max_steps=15, timeout_minutes=10)

    agent = Agent(
        capabilities=capabilities,
        budget=budget,
        llm_model="claude-sonnet-4-5-20250929",
        system_prompt=(
            "You are an SRE triage agent investigating a reported issue.\n\n"
            "The alert says: 'myservice is returning 500 errors at elevated rate'\n\n"
            "Your job:\n"
            "1. Check pod status to see if pods are healthy\n"
            "2. Read recent logs to find error patterns\n"
            "3. Check the health endpoint\n"
            "4. Provide a diagnosis with:\n"
            "   - Root cause (or best guess)\n"
            "   - Severity (P1-P4)\n"
            "   - Recommended next steps\n\n"
            "You can ONLY read. You cannot deploy or modify anything."
        ),
    )

    print("Starting SRE triage agent...")
    print(f"Capabilities (read-only): {[t.name for t in capabilities.tokens]}")
    print()

    result = await agent.run()

    print("\n" + "=" * 60)
    print("TRIAGE RESULT")
    print("=" * 60)
    print(f"\n{result.final_message}")
    print()
    print("AUDIT TRAIL:")
    print(result.audit_trail.to_table())


if __name__ == "__main__":
    asyncio.run(main())
