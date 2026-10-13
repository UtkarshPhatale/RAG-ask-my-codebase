"""
End-to-end checks that /ask records a query_logs row, and that logging can never break
an answer. Marker rows are removed with the secret key afterwards (users cannot delete
log rows by design); everything under test goes through the users' own clients.
"""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from supabase import create_client

from app.config import SUPABASE_SECRET_KEY, SUPABASE_URL
from app.main import app
from app.query_log import write_query_log

client = TestClient(app)
MARK = "__log_writer_test__"
QUESTION = "How does the agent decide which domains and routes it may visit?"


@pytest.fixture(scope="module", autouse=True)
def cleanup_marker_rows():
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()
    yield
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()


def _ask(token: str) -> tuple[str, dict]:
    question = f"{MARK} {uuid.uuid4().hex} {QUESTION}"
    res = client.post(
        "/ask",
        json={"question": question},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    return question, res.json()


def _the_row(db_client, question: str) -> dict:
    rows = db_client.table("query_logs").select("*").eq("question", question).execute().data
    assert len(rows) == 1, f"expected exactly one log row, found {len(rows)}"
    return rows[0]


def test_ask_writes_a_row_with_the_callers_identity(contractor_token, contractor_client):
    question, body = _ask(contractor_token)
    row = _the_row(contractor_client, question)

    assert row["user_id"] == contractor_client.auth.get_user().user.id
    assert row["role"] == "contractor"
    assert row["k"] == 5
    assert row["retrieved_count"] == len(body["citations"]) == len(row["retrieved"])
    assert row["senior_only_count"] == 0
    assert {c["chunk_id"] for c in body["citations"]} == {r["chunk_id"] for r in row["retrieved"]}
    assert set(row["retrieved"][0]) == {"chunk_id", "repo", "path", "scope", "similarity"}
    assert row["retrieval_ms"] >= 0 and row["generation_ms"] >= 0
    assert row["total_ms"] >= row["generation_ms"]


def test_senior_count_matches_what_was_actually_returned(senior_token, senior_client):
    # Compared against the response itself, not a fixed number, so this does not
    # depend on retrieval ranking (which is measured by eval/, not tested here).
    question, body = _ask(senior_token)
    row = _the_row(senior_client, question)
    assert row["role"] == "senior_engineer"
    assert row["senior_only_count"] == sum(
        1 for c in body["citations"] if c["required_scope"] == "senior_engineer"
    )


def test_a_logging_failure_never_propagates():
    class BrokenClient:
        def table(self, name):
            raise RuntimeError("database unavailable")

    citations = [SimpleNamespace(chunk_id="c", repo="r", path="p", required_scope="contractor", similarity=0.5)]
    write_query_log(
        BrokenClient(),
        user_id="u", role="contractor", question="q", k=5,
        citations=citations, retrieval_ms=1, generation_ms=1, total_ms=2,
    )  # must simply return
