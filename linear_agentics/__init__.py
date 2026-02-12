"""Linear Agent Runtime — safety rails for AI agents doing infrastructure work."""

from .actions import (
    CommandNotAllowedError,
    FileAccessError,
    QueryNotAllowedError,
    close_http_client,
    get_http_client,
)
from .agent import Agent, AgentResult, CapabilitySet
from .approval import ApprovalDeniedError, ApprovalGate, ApprovalTimeoutError
from .audit import ApprovalRecord, AuditTrail, ErrorRecord, NegotiationRecord, Proof
from .budget import Budget, BudgetExhaustedError, BudgetTimeoutError
from .negotiation import HumanCapabilityProvider, SupervisorAgentProvider
from .tokens import (
    DatabaseToken,
    DeployToken,
    FileToken,
    HttpToken,
    LinearToken,
    MultiUseDatabaseToken,
    MultiUseFileToken,
    MultiUseShellToken,
    MultiUseToken,
    SecretInjection,
    SecretToken,
    ShellToken,
    TokenError,
    TokenReusedError,
    TokenScopeError,
    WaitToken,
)

__all__ = [
    "Agent",
    "AgentResult",
    "ApprovalDeniedError",
    "ApprovalGate",
    "ApprovalRecord",
    "ApprovalTimeoutError",
    "AuditTrail",
    "Budget",
    "BudgetExhaustedError",
    "BudgetTimeoutError",
    "CapabilitySet",
    "close_http_client",
    "CommandNotAllowedError",
    "DatabaseToken",
    "DeployToken",
    "ErrorRecord",
    "FileAccessError",
    "FileToken",
    "get_http_client",
    "HumanCapabilityProvider",
    "HttpToken",
    "LinearToken",
    "MultiUseDatabaseToken",
    "MultiUseFileToken",
    "MultiUseShellToken",
    "MultiUseToken",
    "NegotiationRecord",
    "Proof",
    "QueryNotAllowedError",
    "SecretInjection",
    "SecretToken",
    "ShellToken",
    "SupervisorAgentProvider",
    "TokenError",
    "TokenReusedError",
    "TokenScopeError",
    "WaitToken",
]
