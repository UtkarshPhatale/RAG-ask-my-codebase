"""
RLS tests for the query_logs table (migration 008), run as real users.

Proves the access rules the observability feature depends on:
  - a user can insert rows about themselves only, and cannot claim another role
  - a user with no role cannot insert
  - contractors read only their own rows; seniors read all
  - log rows cannot be modified or deleted by users
  - anon has no access

Users cannot delete log rows (by design), so marker rows are cleaned up with the
secret key. That is the only use of the secret key here: every read and write
under test goes through the user's own client, so RLS applies.
"""
import uuid

import pytest
from postgrest.exceptions import APIError
from supabase import create_client

from app.config import SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_URL

MARK = "__rls_test__"


def _uid(client) -> str:
    return client.auth.get_user().user.id


def _row(user_id: str, role: str, question: str) -> dict:
    return {
        "user_id": user_id,
        "role": role,
        "question": f"{MARK} {question}",
        "retrieved": [],
        "retrieved_count": 0,
        "senior_only_count": 0,
    }


@pytest.fixture(scope="module", autouse=True)
def cleanup_marker_rows():
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()
    yield
    admin.table("query_logs").delete().like("question", f"{MARK}%").execute()


def test_user_can_insert_own_row(contractor_client):
    res = contractor_client.table("query_logs").insert(
        _row(_uid(contractor_client), "contractor", "own row")
    ).execute()
    assert len(res.data) == 1


def test_user_cannot_insert_row_for_another_user(contractor_client, senior_client):
    with pytest.raises(APIError) as exc:
        contractor_client.table("query_logs").insert(
            _row(_uid(senior_client), "contractor", "spoofed user")
        ).execute()
    assert exc.value.code == "42501"  # row-level security violation


def test_user_cannot_claim_a_role_they_do_not_have(contractor_client):
    with pytest.raises(APIError) as exc:
        contractor_client.table("query_logs").insert(
            _row(_uid(contractor_client), "senior_engineer", "spoofed role")
        ).execute()
    assert exc.value.code == "42501"


def test_user_with_no_role_cannot_insert(norole_client):
    with pytest.raises(APIError) as exc:
        norole_client.table("query_logs").insert(
            _row(_uid(norole_client), "contractor", "no role")
        ).execute()
    assert exc.value.code == "42501"


def test_contractor_reads_only_own_rows_senior_reads_all(contractor_client, senior_client):
    c_uid, s_uid = _uid(contractor_client), _uid(senior_client)
    tag = uuid.uuid4().hex
    contractor_client.table("query_logs").insert(_row(c_uid, "contractor", f"vis-{tag} contractor")).execute()
    senior_client.table("query_logs").insert(_row(s_uid, "senior_engineer", f"vis-{tag} senior")).execute()

    def visible_to(client):
        rows = (
            client.table("query_logs").select("user_id,question")
            .like("question", f"%vis-{tag}%").execute().data
        )
        return {r["user_id"] for r in rows}

    assert visible_to(contractor_client) == {c_uid}          # only its own row
    assert visible_to(senior_client) == {c_uid, s_uid}       # sees everyone's


def test_users_cannot_modify_or_delete_log_rows(contractor_client):
    q = f"{MARK} immutable-{uuid.uuid4().hex}"
    contractor_client.table("query_logs").insert(
        {**_row(_uid(contractor_client), "contractor", "x"), "question": q}
    ).execute()

    for attempt in (
        lambda: contractor_client.table("query_logs").update({"question": "tampered"}).eq("question", q).execute(),
        lambda: contractor_client.table("query_logs").delete().eq("question", q).execute(),
    ):
        try:
            attempt()
        except APIError:
            pass  # a permission error is acceptable; a silent no-op is too

    rows = contractor_client.table("query_logs").select("question").eq("question", q).execute().data
    assert len(rows) == 1 and rows[0]["question"] == q  # neither changed nor deleted


def test_anon_has_no_access():
    anon = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)  # no user session
    try:
        rows = anon.table("query_logs").select("id").limit(5).execute().data
    except APIError:
        return  # permission denied is the expected outcome
    assert rows == []
