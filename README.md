# RAG-ask-my-codebase

A "codebase Q&A" RAG assistant with role-based access control enforced at the
**database layer**, not the application layer or the UI.

## The problem

Engineers waste real time answering "how does X work here" questions across
unfamiliar codebases. An internal Q&A tool that retrieves from your own repos
solves that — but the moment a codebase has anything sensitive in it
(safety-critical logic, admin/billing internals, credentials-adjacent code),
naive RAG becomes a liability: anyone who can query the tool can retrieve
anything in the index, regardless of whether they should be able to see it.

Most RBAC implementations bolt access control onto the application layer —
an `if user.role == "admin"` check somewhere before returning results. That
works until someone adds a new code path that forgets the check. This project
asks a narrower, more defensible question: **can access control be enforced
at the database itself**, so no application code path can accidentally bypass
it?

## The approach

Two roles: `contractor` (restricted) and `senior_engineer` (full access).
Every chunk of code and documentation in the corpus is tagged with a
`required_scope`. Postgres **Row-Level Security (RLS)** — not a `WHERE`
clause written by hand in Python — decides which rows a given authenticated
user's query can see, at query time, in the database itself.

The retrieval function that powers `/ask` (`match_chunks`, a Postgres
function) is declared `SECURITY INVOKER`, not the `SECURITY DEFINER` default
shown in most Supabase examples. This one keyword is the entire security
model: `SECURITY DEFINER` runs a function with its *owner's* privileges,
which — for a function created by an admin-equivalent role — would silently
bypass RLS. `SECURITY INVOKER` runs it as the *calling user*, so the RLS
policy applies exactly as if that user ran a plain `SELECT` themselves.

Nowhere in the Python application code is there a manual scope filter. There
is no `if chunk.scope == "senior_engineer" and user.role != "senior_engineer": skip`.
The chunks that reach the FastAPI layer have already been filtered by
Postgres before they exist in memory — the database has already decided,
not the application.

## Architecture

```
Client (with Supabase JWT)
    |
    v
FastAPI /ask endpoint
    |
    v
JWKS-based JWT verification (ES256) --> role lookup in user_roles
    |
    v
Supabase client scoped to THIS user's session (publishable key + their JWT)
    |
    v
match_chunks() Postgres RPC (SECURITY INVOKER)
    |
    v
RLS policy on `chunks` table filters rows -- enforced by Postgres, not Python
    |
    v
Only permitted chunks ever reach the app
    |
    v
Claude API generates an answer from ONLY those chunks
```

**Why the generation step has no scope awareness at all:** an LLM is not a
reliable access-control boundary — it can be prompted, confused, or simply
wrong about what it should or shouldn't say. The generation function receives
only chunks that have *already* passed through RLS. It doesn't know what a
"contractor" is. It cannot leak what it never received.

## What's been verified, and how

The adversarial test suite (`tests/test_rbac_adversarial.py`) tests
`retrieve_chunks()` directly — not the HTTP endpoint, and not the final LLM
answer. This distinction matters: a test that only checks "did the model's
answer mention the restricted file" would pass even if the restricted chunk
was sitting in the model's context the whole time and the model just chose
not to repeat it. That's a much weaker guarantee than "the restricted chunk
was never retrieved in the first place." This suite proves the latter.

Covered:
- A contractor directly querying for known senior-only content, by close
  paraphrase of its exact wording — asserts zero senior-scoped chunks return.
- The same underlying topic asked indirectly, without naming the restricted
  file — a phrasing-based bypass attempt.
- The inverse: `senior_engineer` genuinely can retrieve that content (so the
  contractor tests aren't just passing because retrieval is broken for
  everyone).
- Cross-repo coverage: the same boundary holds against a second, unrelated
  repo's senior-only content (a different language, a different kind of
  sensitive material), not just one file that happens to work.
- A malformed/injection-style input to the retrieval RPC call, confirming
  Postgres's typed function signature rejects it at the type layer — there is
  no raw SQL string-building in this call path.
- A user authenticated via Supabase Auth but with no row in `user_roles` at
  all is rejected outright (403), rather than silently falling through to
  either role's access.
- A prompt-injection attempt embedded in the natural-language question itself
  does not expand retrieval beyond the caller's own RLS-permitted scope — the
  injection has no path to the database layer where enforcement happens.

All tests pass against the live corpus (~155 documents, ~2,716 chunks across
4 real repositories, 10 files marked senior-only).

## What this project does and doesn't prove

There's an important distinction between two different claims, worth stating
explicitly rather than letting them blur together.

**RLS enforcement is wording-independent.** A contractor cannot retrieve a
senior-scoped chunk no matter how a question is phrased — the block happens
at the database row level, checked against `required_scope` on every
candidate row, after retrieval has already picked its candidates. It doesn't
matter whether the query is vague, exact, or guesses the right file name;
the enforcement doesn't look at wording at all.

**Retrieval quality is a separate, wording-dependent property.** Whether a
natural-language question successfully surfaces the *right* chunk in the
first place depends on how well the embedding model (`all-MiniLM-L6-v2`, a
small, free, local model) captures semantic similarity between a question
and a chunk's content. During testing, a real example came up: a fictional
lab-notes chunk about GPU allocation credentials didn't rank in the top 50
results for any of three different natural-language paraphrases of its own
content — despite the chunk being correctly embedded and ranking itself #1
against its own exact embedding. That's a retrieval-precision limitation
shared by every RAG system built on a small embedding model, not a defect
in this project's access control.

The adversarial test suite's senior-access tests were adjusted accordingly:
rather than relying on a paraphrase closing an embedding-similarity gap that
has nothing to do with security, tests that need to confirm "can this role
retrieve this known chunk" use the chunk's own exact embedding directly,
isolating the actual claim under test (RLS permits/denies correctly) from
an unrelated one (does the embedding model rank paraphrases well).

**A related, more fundamental limitation**, worth stating outright: this
project's scope classifications (`docs/access_design.md`) were all made by
hand, by someone who already knows the corpus intimately. The system
enforces a given classification correctly and provably — it does not
determine what the classification should be. Applying this to an unfamiliar
codebase (a client's repo, an open-source project) is a genuinely harder,
unsolved problem, deliberately deferred to Phase 4 of this project's roadmap
rather than claimed as solved here.

## Stack, and why

| Choice | Reasoning |
|---|---|
| **FastAPI** | Matches typed, schema-driven backend patterns (Pydantic request/response contracts). |
| **Supabase (Postgres + pgvector + Supabase Auth)** | Chosen over Qdrant/FAISS specifically because Postgres RLS enforces access *in the database*, provably — not via an app-layer filter sitting near the vector search. |
| **all-MiniLM-L6-v2 embeddings (384-dim, local)** | Small, free, fast — appropriate for a corpus this size; precision tradeoffs are known and documented, not hidden. |
| **Supabase Auth, not hand-rolled JWT** | Deliberate: this project's differentiator is learning how Postgres RLS reads `auth.uid()` natively, not re-proving JWT implementation skills. |
| **JWKS/ES256 JWT verification** | Supabase's current signing scheme is asymmetric (ES256), not the HS256 shared-secret scheme most tutorials assume — verification fetches Supabase's public keys rather than trusting a static secret. |
| **claude-haiku-4-5 for generation** | Appropriate for a small, already-relevant 5-chunk context; not the same as a generation-quality decision that requires a bigger model. |

## Project status

**Phase 0 (corpus + access design + schema):** complete.
**Phase 1 (retrieval + RBAC enforcement):** complete through the adversarial
test suite above. No demo UI yet — this is API-only.
**Phase 2 (eval set, frontend, observability):** not started.
**Phase 3 (CI/CD, Docker/deployment, hybrid search):** not started.

This README will be updated as later phases land — it does not claim a live
demo, deployed instance, or frontend that don't exist yet.

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own Supabase project
credentials and Anthropic API key. Then:

```bash
uvicorn app.main:app --reload
```

Run the adversarial test suite:

```bash
python -m pytest tests/ -v
```

## Security note on the corpus

Several "sensitive" files in the ingested corpus (e.g.
`nextplay_kanban/src/config/secrets_template.py`,
`brain-tumor-segmentation/internal/lab_notes.md`) are **deliberately fictional
fixtures** created for RBAC testing — clearly marked as such inside the files
themselves. No real credentials or genuinely private information exist in
this repository. Full classification reasoning for every file's scope is in
`docs/access_design.md`.

