"""Tests for low-level action primitives (path validation, query validation)."""

import os

import pytest

from linear_agentics.actions import (
    FileAccessError,
    QueryNotAllowedError,
    _validate_file_path,
    _validate_query,
    file_read,
    file_write,
)


class TestValidateFilePath:
    def test_allows_exact_file(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value")
        result = _validate_file_path(str(f), [str(f)])
        assert result == str(f)

    def test_allows_child_of_directory(self, tmp_path):
        f = tmp_path / "sub" / "data.txt"
        f.parent.mkdir()
        f.write_text("hello")
        result = _validate_file_path(str(f), [str(tmp_path)])
        assert result == os.path.realpath(str(f))

    def test_rejects_traversal(self, tmp_path):
        evil_path = str(tmp_path / ".." / ".." / "etc" / "passwd")
        with pytest.raises(FileAccessError, match="not within"):
            _validate_file_path(evil_path, [str(tmp_path)])

    def test_rejects_symlink_escape(self, tmp_path):
        # Create a symlink inside allowed dir that points outside
        target = tmp_path / "allowed"
        target.mkdir()
        link = target / "escape"
        link.symlink_to("/tmp")
        evil_path = str(link / "something")
        with pytest.raises(FileAccessError, match="not within"):
            _validate_file_path(evil_path, [str(target)])

    def test_rejects_unrelated_path(self, tmp_path):
        with pytest.raises(FileAccessError):
            _validate_file_path("/etc/passwd", [str(tmp_path)])


class TestFileReadWrite:
    async def test_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        content = await file_read(str(f), [str(tmp_path)])
        assert content == "hello world"

    async def test_write_file(self, tmp_path):
        f = tmp_path / "output.txt"
        result = await file_write(str(f), "written", [str(tmp_path)])
        assert "bytes" in result
        assert f.read_text() == "written"

    async def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "deep" / "file.txt"
        await file_write(str(f), "nested", [str(tmp_path)])
        assert f.read_text() == "nested"

    async def test_read_rejects_traversal(self, tmp_path):
        with pytest.raises(FileAccessError):
            await file_read(
                str(tmp_path / ".." / ".." / "etc" / "passwd"), [str(tmp_path)]
            )


class TestValidateQuery:
    def test_allows_matching_prefix(self):
        _validate_query("SELECT * FROM users WHERE id = $1", ["SELECT"])

    def test_case_insensitive(self):
        _validate_query("select * from users", ["SELECT"])

    def test_normalises_whitespace(self):
        _validate_query("SELECT  *   FROM  users", ["SELECT * FROM users"])

    def test_rejects_non_matching_prefix(self):
        with pytest.raises(QueryNotAllowedError):
            _validate_query("DELETE FROM users", ["SELECT"])

    def test_rejects_semicolon(self):
        with pytest.raises(QueryNotAllowedError, match=";"):
            _validate_query("SELECT 1; DROP TABLE users", ["SELECT"])

    def test_rejects_line_comment(self):
        with pytest.raises(QueryNotAllowedError, match="comment"):
            _validate_query("SELECT 1 -- comment", ["SELECT"])

    def test_rejects_block_comment(self):
        with pytest.raises(QueryNotAllowedError, match="comment"):
            _validate_query("SELECT /* evil */ 1", ["SELECT"])

    def test_multiple_allowed_patterns(self):
        _validate_query("INSERT INTO logs VALUES ($1)", ["SELECT", "INSERT INTO logs"])
