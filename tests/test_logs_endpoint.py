"""
HTTP-level tests for GET /logs. The handler has no role check; these tests prove that
the visibility rules of query_logs (migration 008) hold through the endpoint:
contractors see only their own rows, seniors see everyone's, and unauthenticated or
malformed requests are rejected. Marker rows are removed with the secret key
(users cannot delete log rows by design).
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from supabase import create_client

from app.config import SUPABASE_SECRET_KEY, SUPABASE_URL
from app.main import app

client = TestClient(app)
MARK = "__logs_endpoint_test__"


@pytest.fixture(scope="module", autouse=True)
def cleanup_marker_rows():
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()
    yield
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()


def _insert(db_client, role: str, text: str) -> str:
    uid = db_client.auth.get_user().user.id
    db_client.table("query_logs").insert(
        {
            "user_id": uid,
            "role": role,
            "question": f"{MARK} {text}",
            "retrieved": [{"chunk_id": "c1", "repo": "r", "path": "p.py", "scope": "contractor", "similarity": 0.5}],
            "retrieved_count": 1,
            "senior_only_count": 0,
            "retrieval_ms": 1,
            "generation_ms": 2,
            "total_ms": 3,
        }
    ).execute()
    return uid


def _get(token: str, path: str = "/logs?limit=200"):
    return client.get(path, headers={"Authorization": f"Bearer {token}"})


def test_contractor_sees_only_own_rows_and_senior_sees_all(
    contractor_client, senior_client, contractor_token, senior_token
):
    tag = uuid.uuid4().hex
    c_uid = _insert(contractor_client, "contractor", f"{tag} contractor")
    s_uid = _insert(senior_client, "senior_engineer", f"{tag} senior")

    def visible_users(token: str) -> set[str]:
        res = _get(token)
        assert res.status_code == 200
        return {r["user_id"] for r in res.json() if tag in r["question"]}

    assert visible_users(contractor_token) == {c_uid}        # only its own row
    assert visible_users(senior_token) == {c_uid, s_uid}     # sees everyone's


def test_entries_have_the_documented_shape(senior_client, senior_token):
    tag = uuid.uuid4().hex
    _insert(senior_client, "senior_engineer", f"{tag} shape")
    row = next(r for r in _get(senior_token).json() if tag in r["question"])
    assert set(row) == {
        "id", "created_at", "user_id", "role", "question", "k", "retrieved",
        "retrieved_count", "senior_only_count", "retrieval_ms", "generation_ms", "total_ms",
    }
    assert set(row["retrieved"][0]) == {"chunk_id", "repo", "path", "scope", "similarity"}


def test_entries_are_newest_first(senior_client, senior_token):
    tag = uuid.uuid4().hex
    _insert(senior_client, "senior_engineer", f"{tag} first")
    _insert(senior_client, "senior_engineer", f"{tag} second")
    questions = [r["question"] for r in _get(senior_token).json() if tag in r["question"]]
    assert questions == [f"{MARK} {tag} second", f"{MARK} {tag} first"]


def test_limit_is_validated_and_applied(senior_token):
    assert _get(senior_token, "/logs?limit=0").status_code == 422
    assert _get(senior_token, "/logs?limit=201").status_code == 422
    assert len(_get(senior_token, "/logs?limit=3").json()) <= 3


def test_logs_requires_a_valid_token():
    assert client.get("/logs").status_code == 422  # missing header, same as /ask
    bad = client.get("/logs", headers={"Authorization": "Bearer not-a-real-jwt"})
    assert bad.status_code == 401
