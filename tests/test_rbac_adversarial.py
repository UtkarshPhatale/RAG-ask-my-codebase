import pytest

from app.retrieval import retrieve_chunks


def test_contractor_direct_query_never_returns_senior_content(contractor_client):
    chunks = retrieve_chunks(
        contractor_client,
        "irreversible actions require explicit confirmation, replay mode never auto-confirms",
        match_count=20,
    )
    assert all(c["required_scope"] != "senior_engineer" for c in chunks), (
        "Contractor retrieval returned a senior_engineer-scoped chunk -- RLS leak"
    )


def test_contractor_indirect_query_never_returns_senior_content(contractor_client):
    chunks = retrieve_chunks(
        contractor_client,
        "what safety mechanisms prevent dangerous actions from running automatically",
        match_count=20,
    )
    assert all(c["required_scope"] != "senior_engineer" for c in chunks), (
        "Contractor retrieval returned a senior_engineer-scoped chunk via indirect phrasing -- RLS leak"
    )


def test_senior_engineer_can_retrieve_senior_content(senior_client):
    chunks = retrieve_chunks(
        senior_client,
        "irreversible actions require explicit confirmation, replay mode never auto-confirms",
        match_count=20,
    )
    assert any(c["required_scope"] == "senior_engineer" for c in chunks), (
        "senior_engineer retrieval returned no senior-scoped chunks at all -- "
        "either RLS is over-restricting, or this query needs adjusting"
    )


def test_contractor_result_count_never_exceeds_contractor_scope(contractor_client):
    chunks = retrieve_chunks(contractor_client, "how does this codebase work", match_count=50)
    non_contractor = [c for c in chunks if c["required_scope"] != "contractor"]
    assert non_contractor == [], (
        f"Found {len(non_contractor)} non-contractor-scoped chunks in contractor's results"
    )


def test_contractor_blocked_from_different_repo_senior_content(contractor_client):
    """
    Cross-repo check: the earlier tests only proved enforcement against
    interface-ai-computer-use-project's senior-only content. This confirms
    the same boundary holds for a completely different repo, language, and
    kind of sensitive content (a billing-bypass admin function in a
    TypeScript kanban app, vs. a Python safety policy) -- ruling out that
    RLS enforcement is somehow coincidentally specific to one repo's data.
    """
    chunks = retrieve_chunks(
        contractor_client,
        "fictional admin-only operation that force-extends a team's trial period, bypassing normal billing",
        match_count=20,
    )
    assert all(c["required_scope"] != "senior_engineer" for c in chunks), (
        "Contractor retrieval returned senior-scoped billingOverride.ts content -- RLS leak"
    )


def test_senior_engineer_can_retrieve_billing_override_content(senior_client):
    chunks = retrieve_chunks(
        senior_client,
        "fictional admin-only operation that force-extends a team's trial period, bypassing normal billing",
        match_count=20,
    )
    assert any(c["required_scope"] == "senior_engineer" for c in chunks), (
        "senior_engineer could not retrieve billingOverride.ts content at all -- "
        "either RLS is over-restricting, or this query needs adjusting"
    )


def test_match_chunks_rejects_malformed_match_count(contractor_client):
    """
    Adversarial input-handling check on the .rpc() call itself. match_count
    is declared as `int` in the match_chunks() Postgres function signature
    (migration 006), so Postgres's own type system rejects anything that
    isn't a valid integer -- there is no raw SQL string-building anywhere
    in this call path for an attacker to inject into. This test doesn't
    expect to find a real vulnerability; it demonstrates that the input
    boundary is actually enforced by the database driver/type system,
    not just assumed to be safe.
    """
    with pytest.raises(Exception):
        retrieve_chunks(
            contractor_client,
            "how does this codebase work",
            match_count="5; DROP TABLE chunks;--",  # type: ignore
        )


def test_empty_retrieval_handled_gracefully(contractor_client):
    """
    A query with essentially no semantic match to anything in the corpus
    should return a small/irrelevant result set (pgvector always returns
    *some* nearest neighbors, even if distant) or, at match_count=0,
    nothing at all -- and either way, retrieve_chunks must not raise.
    """
    chunks = retrieve_chunks(contractor_client, "asdkjhaskjdh qwoeiqwoe nonsense query", match_count=0)
    assert chunks == []


def test_generation_handles_empty_chunk_list():
    """
    generate_answer must degrade gracefully with zero chunks, not crash or
    silently hallucinate an answer with no grounding. This is a real code
    path (a very obscure or off-topic question could plausibly retrieve
    nothing relevant), not a hypothetical.
    """
    from app.generation import generate_answer

    result = generate_answer("What is the meaning of life?", [])
    assert "couldn't find" in result.lower() or "don't" in result.lower()


def test_user_with_no_assigned_role_is_rejected(norole_client):
    """
    A real, valid Supabase session for a user who has never been given a
    role in user_roles must fail CLOSED (403), not silently default to
    contractor-level or any other access. This confirms app/auth.py's
    explicit role_row.data check (not just relying on RLS) actually does
    something -- a user who authenticates successfully but was never
    assigned a role should get no access at all, not implicit access.
    """
    from fastapi import HTTPException
    from app.auth import get_current_user

    # Extract the access token the same way a real Authorization header would.
    session = norole_client.auth.get_session()
    token = session.access_token

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=f"Bearer {token}")

    assert exc_info.value.status_code == 403