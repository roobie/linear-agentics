"""Tests for audit trail."""

import json

import pytest

from linear_agentics.audit import AuditTrail, ErrorRecord, Proof


class TestProof:
    def test_immutable(self):
        p = Proof(
            token_name="test",
            scope="shell:echo",
            args={"command": "echo hi"},
            timestamp="2025-01-01T00:00:00Z",
            result_summary="hi",
            duration_ms=10.0,
        )
        with pytest.raises(AttributeError):
            p.token_name = "changed"

    def test_to_dict(self):
        p = Proof(
            token_name="test",
            scope="shell:echo",
            args={},
            timestamp="2025-01-01T00:00:00Z",
            result_summary="ok",
            duration_ms=5.0,
        )
        d = p.to_dict()
        assert d["token_name"] == "test"
        assert d["duration_ms"] == 5.0


class TestAuditTrail:
    def test_record_proof(self):
        trail = AuditTrail()
        proof = Proof("t1", "shell:echo", {}, "2025-01-01T00:00:00Z", "ok", 1.0)
        trail.record_proof(proof)
        assert len(trail.proofs) == 1
        assert trail.proofs[0] is proof

    def test_record_error(self):
        trail = AuditTrail()
        trail.record_error("t1", RuntimeError("boom"))
        assert len(trail.errors) == 1
        assert trail.errors[0].error_type == "RuntimeError"
        assert trail.errors[0].message == "boom"

    def test_to_json(self):
        trail = AuditTrail()
        proof = Proof("t1", "scope", {"k": "v"}, "2025-01-01T00:00:00Z", "ok", 1.0)
        trail.record_proof(proof)
        trail.record_error("t2", ValueError("bad"))

        data = json.loads(trail.to_json())
        assert len(data["proofs"]) == 1
        assert len(data["errors"]) == 1
        assert data["proofs"][0]["token_name"] == "t1"

    def test_to_table(self):
        trail = AuditTrail()
        proof = Proof("my-token", "shell:echo", {}, "2025-01-01T00:00:00Z", "hello world", 42.0)
        trail.record_proof(proof)
        trail.record_error("bad-token", RuntimeError("failed"))

        table = trail.to_table()
        assert "my-token" in table
        assert "42ms" in table
        assert "Errors:" in table
        assert "failed" in table

    def test_empty_trail(self):
        trail = AuditTrail()
        data = json.loads(trail.to_json())
        assert data == {"proofs": [], "approvals": [], "errors": [], "negotiations": []}
