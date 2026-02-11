# Linear Agentics

**Give your AI agent exactly the capabilities it needs, nothing more, and prove what it did.**

Linear Agentics wraps infrastructure actions in use-once capability tokens. An LLM agent can only execute what you explicitly grant — every action is scoped, audited, and consumed on use. Destructive operations require human approval. Budget limits prevent runaway execution. When the agent needs capabilities it wasn't granted, it can negotiate for additional tokens from a human or supervisor agent. When the agent finishes, you get an audit trail of everything it did.

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

The agent sees these as LLM tools. It calls them. Each call consumes the token. Try to call it twice — `TokenReusedError`. Try a command outside scope — `CommandNotAllowedError`. Exceed the step budget — `BudgetExhaustedError`. Need a capability you don't have? Call `request_capability` to negotiate for it. Every action is recorded as an immutable `Proof`.

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
Action layer executes (subprocess, HTTP, kubectl, SQL, file I/O)
  |
  v
Approval gate blocks if token requires human sign-off
  |
  v
Capability negotiation: agent requests additional tokens if needed
  |
  v
Audit trail captures everything as structured JSON
```

## Library overview

### Tokens (`linear_agentics.tokens`)

Tokens are the core primitive. Each token is a scoped, auditable permit for a single action (or N actions for multi-use tokens).

#### Shell tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `ShellToken` | Run a shell command matching allowed prefixes | `ShellToken("logs", allowed=["kubectl logs"])` |
| `MultiUseShellToken` | Shell token with N uses instead of 1 | `MultiUseShellToken("read", allowed=["kubectl get"], max_uses=5)` |

Shell commands are parsed with `shlex.split` and executed via `subprocess_exec` (never `shell=True`). Shell operators (`||`, `&&`, `;`, `|`, `&`) are rejected regardless of quoting.

#### HTTP tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `HttpToken` | Make an HTTP request to a specific URL | `HttpToken("health", url="https://...", methods=["GET"])` |

Scoped to a single URL and a set of allowed HTTP methods.

#### File tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `FileToken` | Read or write files within allowed paths | `FileToken("cfg", allowed_paths=["/etc/app"], mode="read")` |
| `MultiUseFileToken` | File access with N uses | `MultiUseFileToken("logs", allowed_paths=["/var/log"], mode="read", max_uses=10)` |

Modes: `"read"`, `"write"`, or `"readwrite"`. Path traversal attacks (`../`, symlink escapes) are blocked — all paths are resolved via `os.path.realpath()` and checked against the allowed list.

#### Database tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `DatabaseToken` | Execute a parameterised SQL query | `DatabaseToken("users", dsn="postgres://...", allowed_patterns=["SELECT"])` |
| `MultiUseDatabaseToken` | Database queries with N uses | `MultiUseDatabaseToken("reader", dsn="...", allowed_patterns=["SELECT"], max_uses=10)` |

Queries use asyncpg with `$1, $2, ...` parameterised placeholders — no string interpolation. Semicolons, `--` comments, and `/* */` comments are rejected. The DSN is never exposed to the LLM.

Requires the optional `db` dependency: `pip install linear-agentics[db]`

#### Deploy tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `DeployToken` | Execute a one-time deployment | `DeployToken("prod", method="kubectl", target="production", image="app:v2")` |

Generates a `kubectl set image` command scoped to the target namespace. Supports optional `rollback_to` for automated rollback.

#### Secret tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `SecretToken` | Inject a credential into an action without exposing it to the LLM | See below |

`SecretToken` wraps an inner token (`HttpToken` or `ShellToken`) and transparently injects a secret at execution time. The LLM sees the same tool interface as the inner token but never sees the secret value. The secret is also redacted from the `Proof` — it never appears in the audit trail.

```python
from linear_agentics.tokens import HttpToken, SecretToken, SecretInjection

inner = HttpToken("api", url="https://api.example.com/deploy", methods=["POST"])
token = SecretToken(
    "authed-api",
    secret_value="Bearer sk-secret-key",
    injection=SecretInjection(kind="header", key="Authorization"),
    inner_token=inner,
)
```

Injection kinds:
- `kind="header"` — injects as an HTTP header (requires `HttpToken` inner)
- `kind="env"` — injects as an environment variable (requires `ShellToken` inner)

#### Wait tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `WaitToken` | Controlled delay with a max-seconds cap | `WaitToken("pause", max_seconds=60)` |

For polling workflows where the agent needs to wait between checks. The `max_seconds` cap prevents the agent from sleeping indefinitely.

#### Token properties

All tokens share these properties:
- **Scoped**: a `ShellToken` with `allowed=["kubectl get pods"]` will reject `rm -rf /`
- **Linear**: consumed on use, cannot be reused (or consumed N times for multi-use variants)
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

### Capability Negotiation (`linear_agentics.negotiation`)

When an agent needs a capability it wasn't initially granted, it can request one at runtime via the `request_capability` meta-tool. Requests are evaluated by a `CapabilityProvider` — either a human operator (`HumanCapabilityProvider`) or a supervisor agent (`SupervisorAgentProvider`).

Negotiation is constrained by:
- **Candidate list**: only pre-configured tokens can be granted (candidates can overlap with initial tokens)
- **Negotiation budget**: a configurable `max_negotiations` limits how many requests per run

```python
from linear_agentics import Agent, CapabilitySet, Budget, HumanCapabilityProvider
from linear_agentics.tokens import ShellToken, HttpToken

agent = Agent(
    capabilities=CapabilitySet([
        ShellToken("read-logs", allowed=["kubectl logs"]),
    ]),
    budget=Budget(max_steps=15, timeout_minutes=10),
    system_prompt="Investigate the issue. Request write access if you need to fix it.",
    capability_provider=HumanCapabilityProvider(),
    candidate_tokens=[
        HttpToken("create-incident", url="https://api.internal/incidents", methods=["POST"]),
        HttpToken("check-metrics", url="https://api.internal/metrics", methods=["GET"]),
    ],
    max_negotiations=2,
)
```

Every negotiation attempt — granted or denied — is recorded as a `NegotiationRecord` in the audit trail.

### Audit (`linear_agentics.AuditTrail`)

Structured record of everything the agent did:

- `proofs` — immutable records of each token consumption (token name, scope, args, timestamp, result summary, duration)
- `approvals` — human approval decisions with approver identity
- `negotiations` — capability negotiation attempts (requested scope, justification, granted or denied, provider type)
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

### Incident triage with database access

An on-call agent that investigates alerts by reading logs, querying the database, and checking health endpoints — but cannot modify anything. The `DatabaseToken` uses parameterised queries to prevent SQL injection, and the DSN is never visible to the LLM.

```python
from linear_agentics.tokens import (
    MultiUseShellToken, MultiUseDatabaseToken, HttpToken, FileToken,
)

capabilities = CapabilitySet([
    MultiUseShellToken("pods", allowed=["kubectl get pods", "kubectl describe pod"], max_uses=5),
    MultiUseShellToken("logs", allowed=["kubectl logs"], max_uses=5),
    MultiUseDatabaseToken(
        "query-db",
        dsn="postgres://readonly:pass@db.internal/prod",
        allowed_patterns=["SELECT"],
        max_uses=10,
    ),
    HttpToken("health", url="https://api.internal/health", methods=["GET"]),
    FileToken("read-config", allowed_paths=["/etc/app/config"], mode="read"),
])
```

### Authenticated API calls with secret injection

An agent that interacts with an external API using credentials that are never exposed to the LLM. The `SecretToken` wraps an `HttpToken` and injects the `Authorization` header at execution time. The secret never appears in the tool definition, the proof, or the audit trail.

```python
from linear_agentics.tokens import HttpToken, SecretToken, SecretInjection

api_token = SecretToken(
    "github-api",
    secret_value=f"Bearer {os.environ['GITHUB_TOKEN']}",
    injection=SecretInjection(kind="header", key="Authorization"),
    inner_token=HttpToken("gh-deployments", url="https://api.github.com/repos/org/app/deployments", methods=["GET", "POST"]),
)

migration = SecretToken(
    "run-migration",
    secret_value=os.environ["DATABASE_URL"],
    injection=SecretInjection(kind="env", key="DATABASE_URL"),
    inner_token=ShellToken("migrate", allowed=["python manage.py migrate"]),
    requires_approval=True,
)
```

### Full-stack deployment with all token types

An agent that performs a complete deployment: reads config, runs database migrations, deploys the service, waits for health checks, and calls an external API — with every action scoped, audited, and the sensitive ones gated behind approval or secret injection.

```python
import os
from linear_agentics import Agent, CapabilitySet, Budget
from linear_agentics.tokens import (
    ShellToken, MultiUseShellToken, HttpToken, DeployToken,
    FileToken, DatabaseToken, SecretToken, SecretInjection, WaitToken,
)

capabilities = CapabilitySet([
    # Read deployment config from disk
    FileToken("read-config", allowed_paths=["/etc/app"], mode="read"),

    # Check current state
    MultiUseShellToken("diagnostics",
        allowed=["kubectl get pods", "kubectl logs", "kubectl describe pod"],
        max_uses=5,
    ),

    # Run database migration with injected credentials
    SecretToken(
        "migrate",
        secret_value=os.environ["DATABASE_URL"],
        injection=SecretInjection(kind="env", key="DATABASE_URL"),
        inner_token=ShellToken("migrate-cmd", allowed=["python manage.py migrate"]),
        requires_approval=True,
    ),

    # Read-only query to verify migration
    DatabaseToken(
        "verify-migration",
        dsn=os.environ["DATABASE_URL"],
        allowed_patterns=["SELECT"],
    ),

    # Deploy to staging (no approval needed)
    DeployToken("staging", method="kubectl", target="staging", image="app:v2.3.1"),

    # Wait for rollout, then check health
    WaitToken("wait-for-rollout", max_seconds=120),
    HttpToken("health-check", url="https://staging.internal/health", methods=["GET"]),

    # Deploy to production (requires human approval)
    DeployToken("production", method="kubectl", target="production",
                image="app:v2.3.1", rollback_to="app:v2.3.0",
                requires_approval=True),

    # Notify Slack via authenticated API
    SecretToken(
        "notify-slack",
        secret_value=os.environ["SLACK_TOKEN"],
        injection=SecretInjection(kind="header", key="Authorization"),
        inner_token=HttpToken("slack", url="https://slack.com/api/chat.postMessage", methods=["POST"]),
    ),
])

agent = Agent(
    capabilities=capabilities,
    budget=Budget(max_steps=30, timeout_minutes=20),
    system_prompt=(
        "Deploy app v2.3.1. Read the config, run migrations, deploy to staging, "
        "verify health, then deploy to production. Notify Slack when done."
    ),
)

result = await agent.run()
print(result.audit_trail.to_json())
```

### Scoped database operations

An agent that can run read queries against a database for analysis. The `DatabaseToken` enforces parameterised queries with prefix matching — the agent cannot run `DELETE` or `DROP` statements even if it tries.

```python
capabilities = CapabilitySet([
    MultiUseDatabaseToken(
        "analytics",
        dsn="postgres://readonly@db.internal/analytics",
        allowed_patterns=["SELECT"],
        max_uses=10,
    ),
    FileToken("write-report", allowed_paths=["/tmp/reports"], mode="write"),
])
```

### CI/CD gate checks

An agent that validates a release candidate: runs tests, checks metrics, verifies canary health. Each check is a separate token. The audit trail becomes the release verification report — proof that every check was executed, what it returned, and how long it took.

### Adaptive incident response

An agent that starts with read-only access but can request write capabilities when it determines action is needed. It begins by investigating — reading logs, checking health endpoints, inspecting pod status. If it detects a real incident, it calls `request_capability` to ask a human operator for permission to create an incident ticket or update the status page. The human sees the justification and the specific capability being requested, and grants or denies it. The agent never gets blanket write access — only the specific action it justified.

```python
capabilities = CapabilitySet([
    MultiUseShellToken("logs", allowed=["kubectl logs"], max_uses=5),
    HttpToken("health", url="https://api.internal/health", methods=["GET"]),
])

candidates = [
    HttpToken("create-incident", url="https://api.internal/incidents", methods=["POST"]),
    HttpToken("update-status", url="https://api.internal/status", methods=["PUT"]),
]

agent = Agent(
    capabilities=capabilities,
    budget=Budget(max_steps=15, timeout_minutes=10),
    system_prompt="Investigate the alert. If it's a real incident, request capabilities to respond.",
    capability_provider=HumanCapabilityProvider(),
    candidate_tokens=candidates,
    max_negotiations=2,
)
```

### Multi-cloud operations

An agent managing resources across providers. Each provider interaction is a separate scoped token — the agent can read from AWS and GCP but only write to the one you're migrating to. Cross-provider blast radius is eliminated by construction.

## Installation

```
pip install linear-agentics
```

Requires Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/).

```
export ANTHROPIC_API_KEY=your-key
```

For database token support:
```
pip install linear-agentics[db]
```

## Running the examples

```bash
# Staged deployment with human approval gate
python examples/deploy_service.py

# Read-only incident triage
python examples/sre_triage.py

# Agent with capability negotiation
python examples/capability_negotiation.py
```

## Running the tests

```bash
uv sync --all-extras
uv run pytest tests/ -v
```

### Mutation testing

```bash
uv run mutmut run
```

See [docs/MUTANT_HUNTING.md](docs/MUTANT_HUNTING.md) for strategies on analyzing surviving mutants.
