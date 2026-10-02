# ADR 0003: Test `retrieve_chunks()` directly; add HTTP-level tests as a complement, not a replacement

## Status
Accepted.

## Context

The adversarial RBAC test suite (`tests/test_rbac_adversarial.py`) calls
`retrieve_chunks()` directly, using real authenticated Supabase clients
(session-scoped fixtures for `contractor`, `senior_engineer`, and a
no-role user). It does not go through the FastAPI `/ask` endpoint or
inspect the LLM's final generated answer.

Later, a real gap was found in a coverage review: nothing verified that
`app/main.py` actually wires authentication, role lookup, retrieval, and
generation together correctly as a live HTTP request. That gap was closed
by adding `tests/test_ask_endpoint.py`, a separate 5-test file using
FastAPI's `TestClient` against the real app, in-process.

This ADR records why the suite is split this way rather than written as
one HTTP-level suite from the start.

## Decision

Two layers of tests exist, each proving a different claim:

1. **`test_rbac_adversarial.py`** (15 tests) proves the *security
   guarantee*: a restricted chunk is never retrieved by a role that
   shouldn't see it, regardless of query wording, at the database/retrieval
   layer — independent of whatever the application or LLM does with the
   result afterward.
2. **`test_ask_endpoint.py`** (5 tests) proves the *application wiring*:
   that `/ask` actually calls the auth dependency, passes the correctly
   user-scoped client through to retrieval, and returns sane responses and
   status codes for valid, malformed, and missing credentials.

## Why this matters

A test suite that only checked the final LLM-generated answer for
restricted content (e.g. "assert the response text doesn't mention
`policy.py`") would be a much weaker guarantee than it looks. It would
pass even in a world where the restricted chunk was retrieved
successfully, placed directly into the model's context, and the model
simply chose, on that occasion, not to repeat it back. LLM output is not
a reliable access-control boundary — it can be prompted around, and
"the model didn't say it this time" is not the same claim as "the model
never had it." Testing `retrieve_chunks()` directly proves the stronger,
actually-meaningful claim: the restricted chunk was never retrieved in
the first place, so there was nothing for the model to leak regardless of
how it's prompted.

Conversely, a suite that *only* tested `retrieve_chunks()` directly, and
never touched the HTTP layer, would have a real blind spot: it says
nothing about whether `app/main.py` actually calls any of this correctly.
A bug in `main.py` — passing an unscoped client, swallowing an exception,
misrouting the dependency — would go completely undetected by the
retrieval-level suite alone, because that suite never exercises `main.py`
at all. This blind spot was real, not hypothetical: it was found during a
Week 3 coverage review and closed the same day.

## Consequence worth stating explicitly

The two suites are intentionally testing different things and neither is
redundant with the other. `test_ask_endpoint.py` does not re-verify the
RBAC boundary itself in detail (it doesn't need 15 adversarial phrasing
variations) — it only needs to confirm the wiring is correct, so its
assertions are comparatively shallow (status code, role field, presence of
expected keys), while `test_rbac_adversarial.py` does the deep adversarial
work. Combining them into one file would blur this distinction and make it
harder to see which test is proving which claim.

A genuinely interesting side effect of writing the HTTP-level tests: they
surfaced that a missing `Authorization` header returns `422` (FastAPI's
own request validation, since `get_current_user`'s `authorization`
parameter has no default), not `401`, which the retrieval-level tests
could never have caught since they never go through the HTTP layer at all.
Documented in `test_ask_endpoint.py` and left as a deliberate open
question (see `docs/adr/` — not yet its own ADR, since no decision has
been made either way).

## Alternative considered and rejected

Writing one HTTP-level suite from the very start of Phase 1, instead of
retrieval-level tests, was the initial instinct many RAG tutorials follow
(test the endpoint, since that's "what users actually call"). Rejected
because it would have made every test both a retrieval test and an
authentication test at once — any failure would require first ruling out
whether the bug was in retrieval, in auth, or in request parsing, before
even getting to the actual RBAC question. Separating "does the database
enforce this boundary" from "does the app call the database correctly"
into two files makes failures much faster to diagnose.