"""
HTTP-level tests for the /ask endpoint.

Everything in test_rbac_adversarial.py tests retrieve_chunks() directly,
which proves RLS enforces access correctly at the database layer. It does
NOT prove that app/main.py actually wires auth -> role lookup ->
retrieve_chunks() -> generation together correctly as a real HTTP request.
This file closes that gap: it hits the FastAPI app in-process (no server
needed) with real tokens and confirms the full request path behaves as
expected, including how it fails for bad or missing auth.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_with_valid_contractor_token_returns_200(contractor_token):
    response = client.post(
        "/ask",
        json={"question": "What does this codebase do?"},
        headers={"Authorization": f"Bearer {contractor_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "contractor"
    assert "answer" in body
    assert "sources" in body


def test_ask_with_valid_senior_token_returns_200(senior_token):
    response = client.post(
        "/ask",
        json={"question": "What does this codebase do?"},
        headers={"Authorization": f"Bearer {senior_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "senior_engineer"
    assert "answer" in body
    assert "sources" in body


def test_ask_with_malformed_bearer_token_returns_401():
    response = client.post(
        "/ask",
        json={"question": "anything"},
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401


def test_ask_with_missing_authorization_header_returns_422():
    # NOTE: this is 422, not 401. get_current_user declares
    # `authorization: str = Header(...)` with no default, so FastAPI's own
    # request validation rejects a request with no Authorization header at
    # all before get_current_user's body ever runs -- the function's own
    # `if not authorization.startswith("Bearer ")` check can only ever
    # fire when the header is present but malformed. If you later want a
    # missing header to come back as 401 instead, the fix is
    # `authorization: str | None = Header(None)` plus an explicit None
    # check inside the function -- that's a real design choice to make
    # deliberately, not a bug this test should silently paper over.
    response = client.post(
        "/ask",
        json={"question": "anything"},
    )
    assert response.status_code == 422


def test_ask_with_malformed_header_not_bearer_prefix_returns_401(contractor_token):
    response = client.post(
        "/ask",
        json={"question": "anything"},
        headers={"Authorization": contractor_token},  # missing "Bearer " prefix
    )
    assert response.status_code == 401