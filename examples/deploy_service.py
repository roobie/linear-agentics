#!/usr/bin/env python3
"""Example: Safe deployment pipeline with capability-bounded agent.

This demo shows how to give an agent scoped deployment capabilities:
- Read pods and logs (multi-use for diagnostics)
- Health check endpoint
- Deploy to staging (no approval needed)
- Deploy to production (requires human approval)

Usage:
    export ANTHROPIC_API_KEY=your-key
    python examples/deploy_service.py
"""

import asyncio

from linear_agentics import (
    Agent,
    Budget,
    CapabilitySet,
    DeployToken,
    HttpToken,
    ShellToken,
)
from linear_agentics.tokens import MultiUseShellToken


async def main():
    capabilities = CapabilitySet([
        # Read-only diagnostics (3 uses each for iterative debugging)
        MultiUseShellToken(
            "read-pods",
            allowed=["kubectl get pods", "kubectl describe pod"],
            max_uses=3,
        ),
        MultiUseShellToken(
            "read-logs",
            allowed=["kubectl logs"],
            max_uses=3,
        ),
        # Health check — single use
        HttpToken(
            "check-health",
            url="https://api.internal/health",
            methods=["GET"],
        ),
        # Deployments
        DeployToken(
            "deploy-staging",
            method="kubectl",
            target="staging",
            image="myservice:v2.3.1",
            requires_approval=False,
        ),
        DeployToken(
            "deploy-prod",
            method="kubectl",
            target="production",
            image="myservice:v2.3.1",
            rollback_to="myservice:v2.3.0",
            requires_approval=True,  # Human must approve prod deploys
        ),
    ])

    budget = Budget(max_steps=20, timeout_minutes=15)

    agent = Agent(
        capabilities=capabilities,
        budget=budget,
        llm_model="claude-sonnet-4-5-20250929",
        system_prompt=(
            "You are an SRE agent. Your task is to safely deploy myservice v2.3.1.\n\n"
            "Steps:\n"
            "1. Check current pod status\n"
            "2. Deploy to staging first\n"
            "3. Verify the health endpoint\n"
            "4. If healthy, deploy to production (this requires approval)\n"
            "5. Report the deployment result\n\n"
            "Be cautious. If anything looks wrong, stop and explain."
        ),
    )

    print("Starting deployment agent...")
    print(f"Budget: {budget.max_steps} steps, {budget.timeout_minutes} min timeout")
    print(f"Capabilities: {[t.name for t in capabilities.tokens]}")
    print()

    result = await agent.run()

    print("\n" + "=" * 60)
    print("AGENT RESULT")
    print("=" * 60)
    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Budget remaining: {result.budget_remaining}")
    print(f"Unused tokens: {result.tokens_unused}")
    print()
    print("AUDIT TRAIL:")
    print(result.audit_trail.to_table())
    print()
    print("FULL AUDIT JSON:")
    print(result.audit_trail.to_json())


if __name__ == "__main__":
    asyncio.run(main())
