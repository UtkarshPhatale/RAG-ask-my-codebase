"""
generate_answer must not need an API key unless it actually calls Claude.

These tests make no network calls. They exist so a CI run without
ANTHROPIC_API_KEY can still import the app and exercise everything that
does not generate an answer.
"""
import pytest

import app.generation as generation


def test_empty_chunks_need_no_client(monkeypatch):
    monkeypatch.setattr(generation, "_client", None)
    monkeypatch.setattr(generation, "ANTHROPIC_API_KEY", None)
    answer = generation.generate_answer("anything", [])
    assert "couldn't find" in answer.lower()
    assert generation._client is None  # no client was built


def test_missing_key_fails_clearly_when_a_call_is_needed(monkeypatch):
    monkeypatch.setattr(generation, "_client", None)
    monkeypatch.setattr(generation, "ANTHROPIC_API_KEY", None)
    chunks = [{"required_scope": "contractor", "content": "x"}]
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generation.generate_answer("anything", chunks)