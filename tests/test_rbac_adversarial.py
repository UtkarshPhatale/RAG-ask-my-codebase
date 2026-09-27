from app.retrieval import retrieve_chunks


# Known senior-only content, confirmed to exist in the live corpus
# (guardrails/policy.py). Used as ground truth across these tests --
# if this specific string ever appears in a contractor's results, RLS
# has failed, full stop, regardless of how the query was phrased.
KNOWN_SENIOR_ONLY_FRAGMENT = "replay mode NEVER auto-confirms"


def test_contractor_direct_query_never_returns_senior_content(contractor_client):
    """
    Directly asking about senior-only material, by name, must return
    zero senior-scoped chunks for a contractor. This is the most basic
    adversarial case: a contractor who knows exactly what they're looking
    for and asks for it directly.
    """
    chunks = retrieve_chunks(
        contractor_client,
        "irreversible actions require explicit confirmation, replay mode never auto-confirms",
        match_count=20,  # wide net -- we want to catch a leak even if it ranks low
    )
    assert all(c["required_scope"] != "senior_engineer" for c in chunks), (
        "Contractor retrieval returned a senior_engineer-scoped chunk -- RLS leak"
    )


def test_contractor_indirect_query_never_returns_senior_content(contractor_client):
    """
    Asking about the SAME underlying topic without naming the restricted
    file or using its exact wording -- e.g. asking generally about safety
    mechanisms rather than naming guardrails/policy.py directly. A leak
    here would be worse than the direct case: it means scope enforcement
    can be bypassed just by phrasing a question differently, which is
    exactly the kind of gap an app-layer filter (matching on file paths
    or keywords) would miss but RLS should not.
    """
    chunks = retrieve_chunks(
        contractor_client,
        "what safety mechanisms prevent dangerous actions from running automatically",
        match_count=20,
    )
    assert all(c["required_scope"] != "senior_engineer" for c in chunks), (
        "Contractor retrieval returned a senior_engineer-scoped chunk via indirect phrasing -- RLS leak"
    )


def test_senior_engineer_can_retrieve_senior_content(senior_client):
    """
    The inverse check, equally important: confirms the RLS policy isn't
    simply blocking senior_engineer content for everyone (which would
    pass the contractor tests above for the wrong reason -- broken
    retrieval, not correct enforcement).
    """
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
    """
    Sanity check on the aggregate: every single chunk contractor ever
    gets back, across a broad generic query, must be contractor-scoped.
    Not a targeted adversarial probe like the tests above -- a general
    correctness check that nothing slips through by accident.
    """
    chunks = retrieve_chunks(contractor_client, "how does this codebase work", match_count=50)
    non_contractor = [c for c in chunks if c["required_scope"] != "contractor"]
    assert non_contractor == [], (
        f"Found {len(non_contractor)} non-contractor-scoped chunks in contractor's results"
    )