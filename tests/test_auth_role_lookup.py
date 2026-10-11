"""
The role lookup in get_current_user must tell three cases apart:
  - lookup raised an error      -> 503 (logged; access still denied)
  - lookup returned zero rows   -> 403 (user genuinely has no role)
  - lookup returned a role      -> authenticated user

Before this distinction, any lookup error was reported as 403 "no assigned role",
so a transient failure looked like a permission problem for a valid user.
A stub client is used so these tests are fast and deterministic; the JWT
verification in front of the lookup is real.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.auth as auth


class _FakeQuery:
    def __init__(self, behavior):
        self._behavior = behavior

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        if isinstance(self._behavior, Exception):
            raise self._behavior
        return SimpleNamespace(data=self._behavior)


class _FakeClient:
    def __init__(self, behavior):
        self._behavior = behavior
        self.postgrest = SimpleNamespace(auth=lambda token: None)

    def table(self, name):
        return _FakeQuery(self._behavior)


def _call(monkeypatch, token, behavior):
    monkeypatch.setattr(auth, "create_client", lambda url, key: _FakeClient(behavior))
    return auth.get_current_user(authorization=f"Bearer {token}")


def test_lookup_error_is_503_not_403(monkeypatch, contractor_token):
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, contractor_token, RuntimeError("connection reset"))
    assert exc.value.status_code == 503


def test_zero_rows_is_403(monkeypatch, contractor_token):
    with pytest.raises(HTTPException) as exc:
        _call(monkeypatch, contractor_token, [])
    assert exc.value.status_code == 403


def test_found_role_returns_authenticated_user(monkeypatch, contractor_token):
    user = _call(monkeypatch, contractor_token, [{"role": "contractor"}])
    assert user.role == "contractor"
