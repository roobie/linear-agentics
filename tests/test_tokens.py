"""Tests for linear capability tokens."""

import asyncio
import warnings
from unittest.mock import AsyncMock, patch

import pytest

from linear_agentics.tokens import (
    DatabaseToken,
    DeployToken,
    FileToken,
    HttpToken,
    LinearToken,
    MultiUseDatabaseToken,
    MultiUseFileToken,
    MultiUseShellToken,
    SecretInjection,
    SecretToken,
    ShellToken,
    TokenReusedError,
    TokenScopeError,
    WaitToken,
)
from linear_agentics.actions import CommandNotAllowedError, FileAccessError


class TestShellToken:
    @pytest.fixture
    def echo_token(self):
        return ShellToken("echo-test", allowed=["echo"])

    async def test_consume_allowed_command(self, echo_token):
        proof = await echo_token.consume(command="echo hello")
        assert proof.token_name == "echo-test"
        assert "hello" in proof.result_summary
        assert proof.duration_ms >= 0

    async def test_consume_marks_token_consumed(self, echo_token):
        assert not echo_token.consumed
        await echo_token.consume(command="echo test")
        assert echo_token.consumed

    async def test_reuse_raises(self, echo_token):
        await echo_token.consume(command="echo first")
        with pytest.raises(TokenReusedError, match="already consumed"):
            await echo_token.consume(command="echo second")

    async def test_disallowed_command_raises(self, echo_token):
        with pytest.raises(CommandNotAllowedError):
            await echo_token.consume(command="rm -rf /")

    async def test_disallowed_command_composition_raises(self, echo_token):
        with pytest.raises(CommandNotAllowedError):
            await echo_token.consume(command="echo -n '' || rm -rf /")

    async def test_proof_recorded(self, echo_token):
        proof = await echo_token.consume(command="echo proof-test")
        assert echo_token.proof is proof
        assert proof.args == {"command": "echo proof-test"}

    def test_to_tool_definition(self, echo_token):
        tool_def = echo_token.to_tool_definition()
        assert tool_def["name"] == "shell_echo-test"
        assert "ONE USE ONLY" in tool_def["description"]
        assert "echo" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert schema["properties"]["command"]["type"] == "string"
        assert schema["required"] == ["command"]

    def test_to_tool_definition_with_approval(self):
        token = ShellToken("sensitive", allowed=["ls"], requires_approval=True)
        tool_def = token.to_tool_definition()
        assert "REQUIRES APPROVAL" in tool_def["description"]
        assert "ONE USE ONLY" in tool_def["description"]
        # prevent __del__ warning
        token._consumed = True

    def test_unused_token_warns(self):
        token = ShellToken("temp", allowed=["echo"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del token
            assert any("never consumed" in str(warning.message) for warning in w)


class TestHttpToken:
    async def test_consume_with_mock(self):
        token = HttpToken("health", url="https://example.com/health", methods=["GET"])
        mock_result = {"status": 200, "body": "ok"}
        with patch("linear_agentics.tokens.http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_result
            proof = await token.consume(method="GET")
            mock_req.assert_called_once_with(
                "https://example.com/health", "GET", ["GET"], body=None
            )
        assert proof.token_name == "health"
        assert proof.args == {"method": "GET", "url": "https://example.com/health", "body": None}
        assert "HTTP 200" in proof.result_summary
        assert token.consumed

    async def test_consume_default_method(self):
        token = HttpToken("api", url="https://example.com/api", methods=["POST", "GET"])
        mock_result = {"status": 201, "body": "created"}
        with patch("linear_agentics.tokens.http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_result
            proof = await token.consume()
            # Default method should be the first in the list
            mock_req.assert_called_once_with(
                "https://example.com/api", "POST", ["POST", "GET"], body=None
            )
        assert proof.args["method"] == "POST"

    async def test_consume_with_body(self):
        token = HttpToken("api", url="https://example.com/api", methods=["POST"])
        mock_result = {"status": 200, "body": "ok"}
        with patch("linear_agentics.tokens.http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_result
            proof = await token.consume(method="POST", body={"key": "value"})
            mock_req.assert_called_once_with(
                "https://example.com/api", "POST", ["POST"], body={"key": "value"}
            )
        assert proof.args["body"] == {"key": "value"}

    async def test_to_tool_definition(self):
        token = HttpToken("health", url="https://example.com/health", methods=["GET"])
        tool_def = token.to_tool_definition()
        assert tool_def["name"] == "health"
        assert "GET" in tool_def["description"]
        assert "ONE USE ONLY" in tool_def["description"]
        assert "https://example.com/health" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "method" in schema["properties"]
        assert schema["properties"]["method"]["enum"] == ["GET"]
        assert "body" in schema["properties"]
        token._consumed = True

    async def test_to_tool_definition_with_approval(self):
        token = HttpToken("api", url="https://example.com", methods=["POST"], requires_approval=True)
        tool_def = token.to_tool_definition()
        assert "REQUIRES APPROVAL" in tool_def["description"]
        token._consumed = True

    async def test_reuse_raises(self):
        """HttpToken can't actually make requests in tests, but reuse check is sync."""
        token = HttpToken("test", url="https://example.com", methods=["GET"])
        # Manually mark consumed
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume(method="GET")


class TestDeployToken:
    def test_init_stores_attributes(self):
        token = DeployToken(
            "staging",
            method="kubectl",
            target="staging",
            image="app:v1",
            rollback_to="app:v0",
            requires_approval=True,
        )
        assert token.method == "kubectl"
        assert token.target == "staging"
        assert token.image == "app:v1"
        assert token.rollback_to == "app:v0"
        assert token.requires_approval is True
        assert "kubectl set image" in token._deploy_command
        assert token._rollback_command is not None
        token._consumed = True

    def test_init_no_rollback(self):
        token = DeployToken("prod", method="kubectl", target="prod", image="app:v2")
        assert token.rollback_to is None
        assert token._rollback_command is None
        token._consumed = True

    async def test_consume_with_mock(self):
        token = DeployToken("staging", method="kubectl", target="staging", image="app:v1")
        with patch("linear_agentics.tokens.shell_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "deployment.apps/app image updated"
            proof = await token.consume()
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert "kubectl set image" in call_args[0][0]
            assert "app:v1" in call_args[0][0]
            assert "--namespace=staging" in call_args[0][0]
        assert proof.token_name == "staging"
        assert proof.args == {"method": "kubectl", "target": "staging", "image": "app:v1"}
        assert "Deployed app:v1 to staging" in proof.result_summary
        assert token.consumed

    def test_to_tool_definition(self):
        token = DeployToken(
            "staging",
            method="kubectl",
            target="staging",
            image="app:v1",
            requires_approval=True,
        )
        tool_def = token.to_tool_definition()
        assert tool_def["name"] == "deploy_staging"
        assert "REQUIRES APPROVAL" in tool_def["description"]
        assert "ONE USE ONLY" in tool_def["description"]
        assert "app:v1" in tool_def["description"]
        assert "staging" in tool_def["description"]
        assert "kubectl" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert schema["required"] == []
        token._consumed = True

    async def test_reuse_raises(self):
        token = DeployToken("prod", method="kubectl", target="prod", image="app:v1")
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume()


class TestMultiUseShellToken:
    @pytest.fixture
    def multi_echo(self):
        return MultiUseShellToken("multi-echo", allowed=["echo"], max_uses=3)

    async def test_multiple_uses(self, multi_echo):
        proof1 = await multi_echo.consume(command="echo one")
        assert multi_echo.uses_remaining == 2
        assert not multi_echo.consumed

        proof2 = await multi_echo.consume(command="echo two")
        assert multi_echo.uses_remaining == 1

        proof3 = await multi_echo.consume(command="echo three")
        assert multi_echo.uses_remaining == 0
        assert multi_echo.consumed

    async def test_exhausted_raises(self, multi_echo):
        for i in range(3):
            await multi_echo.consume(command=f"echo {i}")
        with pytest.raises(TokenReusedError, match="exhausted"):
            await multi_echo.consume(command="echo four")

    def test_to_tool_definition(self, multi_echo):
        tool_def = multi_echo.to_tool_definition()
        assert tool_def["name"] == "shell_multi-echo"
        assert "3 uses remaining" in tool_def["description"]
        assert "echo" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert schema["properties"]["command"]["type"] == "string"
        assert schema["required"] == ["command"]


class TestFileToken:
    @pytest.fixture
    def read_token(self, tmp_path):
        return FileToken("cfg", allowed_paths=[str(tmp_path)], mode="read")

    @pytest.fixture
    def write_token(self, tmp_path):
        return FileToken("out", allowed_paths=[str(tmp_path)], mode="write")

    @pytest.fixture
    def rw_token(self, tmp_path):
        return FileToken("rw", allowed_paths=[str(tmp_path)], mode="readwrite")

    async def test_read_allowed_path(self, tmp_path, read_token):
        f = tmp_path / "config.yaml"
        f.write_text("key: value")
        proof = await read_token.consume(operation="read", path=str(f))
        assert "key: value" in proof.result_summary
        assert proof.token_name == "cfg"

    async def test_write_allowed_path(self, tmp_path, write_token):
        f = tmp_path / "artifact.txt"
        proof = await write_token.consume(
            operation="write", path=str(f), content="hello"
        )
        assert f.read_text() == "hello"
        assert "bytes" in proof.result_summary

    async def test_path_traversal_rejected(self, tmp_path, read_token):
        evil = str(tmp_path / ".." / ".." / "etc" / "passwd")
        with pytest.raises(FileAccessError):
            await read_token.consume(operation="read", path=evil)

    async def test_mode_enforcement_read_only(self, tmp_path, read_token):
        with pytest.raises(TokenScopeError, match="not allowed"):
            await read_token.consume(
                operation="write", path=str(tmp_path / "x"), content="bad"
            )

    async def test_mode_enforcement_write_only(self, tmp_path, write_token):
        with pytest.raises(TokenScopeError, match="not allowed"):
            await write_token.consume(operation="read", path=str(tmp_path / "x"))

    async def test_readwrite_allows_both(self, tmp_path, rw_token):
        f = tmp_path / "data.txt"
        f.write_text("original")
        proof = await rw_token.consume(operation="read", path=str(f))
        assert "original" in proof.result_summary

    async def test_reuse_raises(self, tmp_path, read_token):
        f = tmp_path / "test.txt"
        f.write_text("test")
        await read_token.consume(operation="read", path=str(f))
        with pytest.raises(TokenReusedError):
            await read_token.consume(operation="read", path=str(f))

    def test_to_tool_definition(self, read_token):
        tool_def = read_token.to_tool_definition()
        assert tool_def["name"] == "file_cfg"
        assert "ONE USE ONLY" in tool_def["description"]
        assert "read" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert schema["properties"]["operation"]["enum"] == ["read"]
        assert schema["properties"]["operation"]["type"] == "string"
        assert "path" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert schema["required"] == ["operation", "path"]
        # read-only should NOT have content property
        assert "content" not in schema["properties"]

    def test_to_tool_definition_readwrite_has_content(self, rw_token):
        tool_def = rw_token.to_tool_definition()
        schema = tool_def["input_schema"]
        assert "content" in schema["properties"]
        assert schema["properties"]["content"]["type"] == "string"
        assert schema["properties"]["operation"]["enum"] == [
            "read",
            "write",
        ]
        assert schema["required"] == ["operation", "path"]

    def test_to_tool_definition_write_only(self, write_token):
        tool_def = write_token.to_tool_definition()
        schema = tool_def["input_schema"]
        assert schema["properties"]["operation"]["enum"] == ["write"]
        assert "content" in schema["properties"]


class TestMultiUseFileToken:
    async def test_multiple_reads(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content")
        token = MultiUseFileToken(
            "reader", allowed_paths=[str(tmp_path)], mode="read", max_uses=3
        )
        await token.consume(operation="read", path=str(f))
        assert token.uses_remaining == 2
        await token.consume(operation="read", path=str(f))
        assert token.uses_remaining == 1
        await token.consume(operation="read", path=str(f))
        assert token.consumed

    async def test_exhausted_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content")
        token = MultiUseFileToken(
            "reader", allowed_paths=[str(tmp_path)], mode="read", max_uses=1
        )
        await token.consume(operation="read", path=str(f))
        with pytest.raises(TokenReusedError, match="exhausted"):
            await token.consume(operation="read", path=str(f))


class TestSecretToken:
    async def test_header_injection(self):
        inner = HttpToken("api", url="https://api.example.com", methods=["POST"])
        token = SecretToken(
            "authed-api",
            secret_value="Bearer sk-secret-123",
            injection=SecretInjection(kind="header", key="Authorization"),
            inner_token=inner,
        )
        mock_result = {"status": 200, "body": "ok"}
        with patch("linear_agentics.tokens.http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_result
            proof = await token.consume(method="POST", body={"action": "deploy"})

            # Verify the header was injected
            mock_req.assert_called_once_with(
                "https://api.example.com",
                "POST",
                ["POST"],
                body={"action": "deploy"},
                headers={"Authorization": "Bearer sk-secret-123"},
            )

        # Verify secret is NOT in proof
        assert "sk-secret-123" not in str(proof.args)
        assert "sk-secret-123" not in proof.result_summary
        assert "[secret injected" in proof.result_summary

    async def test_env_injection(self):
        inner = ShellToken("migrate", allowed=["python manage.py"])
        token = SecretToken(
            "authed-migrate",
            secret_value="postgres://secret@host/db",
            injection=SecretInjection(kind="env", key="DATABASE_URL"),
            inner_token=inner,
        )
        with patch("linear_agentics.tokens.shell_exec_with_env", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "Migrations applied"
            proof = await token.consume(command="python manage.py migrate")

            mock_exec.assert_called_once_with(
                "python manage.py migrate",
                ["python manage.py"],
                {"DATABASE_URL": "postgres://secret@host/db"},
            )

        assert "postgres://secret@host/db" not in str(proof.args)
        assert "postgres://secret@host/db" not in proof.result_summary
        assert "[secret injected" in proof.result_summary

    async def test_reuse_raises(self):
        inner = HttpToken("api", url="https://example.com", methods=["GET"])
        token = SecretToken(
            "test",
            secret_value="secret",
            injection=SecretInjection(kind="header", key="X-Key"),
            inner_token=inner,
        )
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume(method="GET")

    def test_to_tool_definition_mirrors_inner(self):
        inner = HttpToken("api", url="https://example.com", methods=["GET"])
        token = SecretToken(
            "authed",
            secret_value="secret",
            injection=SecretInjection(kind="header", key="Authorization"),
            inner_token=inner,
        )
        tool_def = token.to_tool_definition()
        inner_def = inner.to_tool_definition()
        assert tool_def["name"] == "secret_authed"
        assert tool_def["input_schema"] == inner_def["input_schema"]
        assert "ONE USE ONLY" in tool_def["description"]
        assert "header" in tool_def["description"]
        assert "Authorization" in tool_def["description"]
        token._consumed = True
        inner._consumed = True

    def test_secret_not_in_tool_definition(self):
        inner = ShellToken("cmd", allowed=["echo"])
        token = SecretToken(
            "s",
            secret_value="super-secret-value",
            injection=SecretInjection(kind="env", key="KEY"),
            inner_token=inner,
        )
        tool_def = token.to_tool_definition()
        assert "super-secret-value" not in str(tool_def)
        token._consumed = True
        inner._consumed = True


class TestDatabaseToken:
    async def test_allowed_query(self):
        token = DatabaseToken(
            "users-db",
            dsn="postgres://localhost/test",
            allowed_patterns=["SELECT"],
        )
        mock_rows = [{"id": 1, "name": "alice"}]
        with patch("linear_agentics.tokens.db_query", new_callable=AsyncMock) as mock_db:
            mock_db.return_value = mock_rows
            proof = await token.consume(
                query="SELECT * FROM users WHERE id = $1", params=[1]
            )
            mock_db.assert_called_once_with(
                "postgres://localhost/test",
                "SELECT * FROM users WHERE id = $1",
                [1],
                ["SELECT"],
            )
        assert proof.token_name == "users-db"
        assert "alice" in proof.result_summary

    async def test_reuse_raises(self):
        token = DatabaseToken("db", dsn="postgres://x", allowed_patterns=["SELECT"])
        token._consumed = True
        with pytest.raises(TokenReusedError):
            await token.consume(query="SELECT 1")

    def test_to_tool_definition(self):
        token = DatabaseToken(
            "db", dsn="postgres://x", allowed_patterns=["SELECT", "INSERT"]
        )
        tool_def = token.to_tool_definition()
        assert tool_def["name"] == "db_db"
        assert "ONE USE ONLY" in tool_def["description"]
        assert "SELECT" in tool_def["description"]
        assert "INSERT" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "params" in schema["properties"]
        assert schema["required"] == ["query"]
        # DSN must not appear in tool definition
        assert "postgres://x" not in str(tool_def)
        token._consumed = True

    def test_dsn_not_in_tool_definition(self):
        token = DatabaseToken(
            "db", dsn="postgres://user:pass@host/db", allowed_patterns=["SELECT"]
        )
        tool_def = token.to_tool_definition()
        assert "postgres://" not in str(tool_def)
        token._consumed = True


class TestMultiUseDatabaseToken:
    async def test_multiple_queries(self):
        token = MultiUseDatabaseToken(
            "reader",
            dsn="postgres://localhost/test",
            allowed_patterns=["SELECT"],
            max_uses=3,
        )
        with patch("linear_agentics.tokens.db_query", new_callable=AsyncMock) as mock_db:
            mock_db.return_value = [{"count": 42}]
            await token.consume(query="SELECT count(*) FROM users")
            assert token.uses_remaining == 2
            await token.consume(query="SELECT count(*) FROM orders")
            assert token.uses_remaining == 1
            await token.consume(query="SELECT count(*) FROM items")
            assert token.consumed

    async def test_exhausted_raises(self):
        token = MultiUseDatabaseToken(
            "reader",
            dsn="postgres://localhost/test",
            allowed_patterns=["SELECT"],
            max_uses=1,
        )
        with patch("linear_agentics.tokens.db_query", new_callable=AsyncMock) as mock_db:
            mock_db.return_value = []
            await token.consume(query="SELECT 1")
            with pytest.raises(TokenReusedError, match="exhausted"):
                await token.consume(query="SELECT 2")


class TestWaitToken:
    @pytest.fixture
    def wait_token(self):
        return WaitToken("pause", max_seconds=60)

    async def test_consume_sleeps(self, wait_token):
        with patch("linear_agentics.tokens.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            proof = await wait_token.consume(seconds=5, reason="waiting for deploy")
            mock_sleep.assert_called_once_with(5)
        assert "Waited 5s" in proof.result_summary
        assert "waiting for deploy" in proof.result_summary
        assert proof.args == {"seconds": 5, "reason": "waiting for deploy"}

    async def test_max_seconds_exceeded(self, wait_token):
        with pytest.raises(ValueError, match="exceeds maximum"):
            await wait_token.consume(seconds=120)

    async def test_negative_seconds_rejected(self, wait_token):
        with pytest.raises(ValueError, match="non-negative"):
            await wait_token.consume(seconds=-1)

    async def test_reuse_raises(self, wait_token):
        with patch("linear_agentics.tokens.asyncio.sleep", new_callable=AsyncMock):
            await wait_token.consume(seconds=1)
        with pytest.raises(TokenReusedError):
            await wait_token.consume(seconds=1)

    def test_to_tool_definition(self, wait_token):
        tool_def = wait_token.to_tool_definition()
        assert tool_def["name"] == "wait_pause"
        assert "ONE USE ONLY" in tool_def["description"]
        assert "max 60s" in tool_def["description"]
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "seconds" in schema["properties"]
        assert schema["properties"]["seconds"]["type"] == "integer"
        assert "reason" in schema["properties"]
        assert schema["required"] == ["seconds"]
