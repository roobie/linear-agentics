# Linear Agent Runtime

**Give your AI agent exactly the capabilities it needs, nothing more, and prove what it did.**

Linear Agent Runtime wraps infrastructure actions in use-once capability tokens. An LLM agent can only execute what you explicitly grant — every action is scoped, audited, and consumed on use. Destructive operations require human approval. Budget limits prevent runaway execution. When the agent finishes, you get a cryptographic-style audit trail of everything it did.

```python
from linear_agentics import Agent, CapabilitySet, Budget
from linear_agentics.tokens import ShellToken, HttpToken, DeployToken

capabilities = CapabilitySet([
    ShellToken("read-pods", allowed=["kubectl get pods"]),
    HttpToken("check-health", url="https://api.internal/health", methods=["GET"]),
    DeployToken("deploy-prod", method="kubectl", target="production",
                image="myservice:v2.3.1", requires_approval=True),
])

agent = Agent(
    capabilities=capabilities,
    budget=Budget(max_steps=20, timeout_minutes=15),
    system_prompt="Deploy myservice v2.3.1 safely.",
)

result = await agent.run()
print(result.audit_trail.to_json())
```

The agent sees these as LLM tools. It calls them. Each call consumes the token. Try to call it twice — `TokenReusedError`. Try a command outside scope — `CommandNotAllowedError`. Exceed the step budget — `BudgetExhaustedError`. Every action is recorded as an immutable `Proof`.

## How it works

```
You define CapabilitySet + Budget
  |
  v
Agent converts tokens to LLM tool definitions
  |
  v
LLM reasons, calls tools
  |
  v
Token layer validates scope, enforces one-time use, records proof
  |
  v
Action layer executes (subprocess, HTTP, kubectl)
  |
  v
Approval gate blocks if token requires human sign-off
  |
  v
Audit trail captures everything as structured JSON
```

## Library overview

### Tokens (`linear_agentics.tokens`)

Tokens are the core primitive. Each token is a scoped, auditable permit for a single action.

| Token | Purpose | Example |
|-------|---------|---------|
| `ShellToken` | Run a shell command matching allowed prefixes | `ShellToken("logs", allowed=["kubectl logs"])` |
| `HttpToken` | Make an HTTP request to a specific URL | `HttpToken("health", url="https://...", methods=["GET"])` |
| `DeployToken` | Execute a one-time deployment | `DeployToken("prod", method="kubectl", target="production", image="app:v2")` |
| `MultiUseShellToken` | Shell token with N uses instead of 1 | `MultiUseShellToken("read", allowed=["kubectl get"], max_uses=5)` |

All tokens share these properties:
- **Scoped**: a `ShellToken` with `allowed=["kubectl get pods"]` will reject `rm -rf /`
- **Linear**: consumed on use, cannot be reused (or consumed N times for `MultiUseToken`)
- **Audited**: every consumption produces an immutable `Proof` with timestamp, args, result, and duration
- **Approval-aware**: set `requires_approval=True` to block execution until a human approves
- **Self-reporting**: unused tokens emit a warning on garbage collection

### Budget (`linear_agentics.Budget`)

Prevents runaway agents with two hard limits:

- **Step budget**: maximum number of LLM tool calls before the agent is stopped
- **Timeout**: wall-clock time limit for the entire run

```python
budget = Budget(max_steps=20, timeout_minutes=15)
```

### Agent (`linear_agentics.Agent`)

The execution loop. Takes a `CapabilitySet`, a `Budget`, and a system prompt. Calls the Anthropic API, routes tool calls through the token layer, and returns an `AgentResult` with the full audit trail.

```python
result = await agent.run()

result.stop_reason       # "complete", "budget_exhausted", "timeout", "error"
result.audit_trail       # AuditTrail with .to_json() and .to_table()
result.tokens_unused     # names of tokens never consumed
result.budget_remaining  # steps left
```

### Approval (`linear_agentics.ApprovalGate`)

Human-in-the-loop for sensitive actions. When an agent tries to consume a token with `requires_approval=True`, execution pauses and prompts for confirmation via CLI. The agent cannot proceed until a human types `yes`.

### Audit (`linear_agentics.AuditTrail`)

Structured record of everything the agent did:

- `proofs` — immutable records of each token consumption (token name, scope, args, timestamp, result summary, duration)
- `approvals` — human approval decisions with approver identity
- `errors` — any failures (scope violations, reuse attempts, execution errors)

Output as JSON (`to_json()`) or a formatted CLI table (`to_table()`).

## Use cases

### Automated deployment pipelines

An agent that deploys a service through environments: staging first, then production after health checks pass. The agent gets `DeployToken`s for each environment — staging without approval, production with mandatory human sign-off. Read-only tokens let it check pod status and logs along the way. If the staging deploy looks bad, the agent stops. It physically cannot deploy to production without a human typing "yes".

```python
capabilities = CapabilitySet([
    MultiUseShellToken("diagnostics", allowed=["kubectl get pods", "kubectl logs"], max_uses=5),
    HttpToken("health-check", url="https://staging.internal/health", methods=["GET"]),
    DeployToken("staging", method="kubectl", target="staging", image="app:v2.3.1"),
    DeployToken("production", method="kubectl", target="production",
                image="app:v2.3.1", rollback_to="app:v2.3.0", requires_approval=True),
])
```

### Incident triage bots

An on-call agent that investigates alerts autonomously. It can read pod status, tail logs, and check health endpoints — but it cannot restart pods, scale deployments, or modify anything. The capability set is purely read-only. The agent gathers evidence and reports a diagnosis with severity and recommended actions. A human decides what to do next.

```python
capabilities = CapabilitySet([
    MultiUseShellToken("pods", allowed=["kubectl get pods", "kubectl describe pod"], max_uses=5),
    MultiUseShellToken("logs", allowed=["kubectl logs"], max_uses=5),
    MultiUseShellToken("events", allowed=["kubectl get events"], max_uses=3),
    HttpToken("health", url="https://api.internal/health", methods=["GET"]),
])
```

### Scoped database operations

An agent that can run read queries against a database for analysis, but write operations require approval. The token system prevents the agent from running arbitrary SQL — only the specific queries you permit.

```python
capabilities = CapabilitySet([
    MultiUseShellToken("read-db", allowed=["psql -c 'SELECT"], max_uses=10),
    ShellToken("run-migration", allowed=["flyway migrate"], requires_approval=True),
])
```

### CI/CD gate checks

An agent that validates a release candidate: runs tests, checks metrics, verifies canary health. Each check is a separate token. The audit trail becomes the release verification report — proof that every check was executed, what it returned, and how long it took.

### Multi-cloud operations

An agent managing resources across providers. Each provider interaction is a separate scoped token — the agent can read from AWS and GCP but only write to the one you're migrating to. Cross-provider blast radius is eliminated by construction.

## Installation

```
pip install linear-agent-runtime
```

Requires Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/).

```
export ANTHROPIC_API_KEY=your-key
```

## Running the examples

```bash
# Staged deployment with human approval gate
python examples/deploy_service.py

# Read-only incident triage
python examples/sre_triage.py
```

## Running the tests

```bash
uv sync --all-extras
uv run pytest tests/ -v
```
