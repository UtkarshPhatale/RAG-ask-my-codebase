# ADR 0004: Separate "RLS enforcement is correct" from "retrieval found the right chunk" as two different claims

## Status
Accepted. This decision shapes how several tests are written, not just how
the project is described.

## Context

While writing `test_senior_engineer_can_retrieve_lab_notes_content`, a
specific fictional chunk (`internal/lab_notes.md`, about GPU allocation
credentials) failed to appear in the top 50 retrieval results for three
different natural-language paraphrases of its own content — even though
the chunk was confirmed, via direct SQL, to be correctly embedded and to
rank itself #1 (similarity = 1.0) against its own exact embedding.

This raised a legitimate concern (raised by the project owner, not
discovered independently): if a test needs to already know a chunk's exact
wording to retrieve it, is the test actually proving anything? Doesn't a
real contractor — who, by definition, doesn't know what's in a file they
can't see — never get close enough in phrasing to trigger a leak in the
first place, making the "indirect query" tests look more rigorous than
they are?

## Decision

Two claims are explicitly separated, in both the test suite and the
README:

1. **RLS enforcement is wording-independent.** Whether a chunk is returned
   to a given role is decided entirely by `required_scope` evaluated
   against the querying role, at the database level, after candidate rows
   are selected. It does not matter whether the query wording is close,
   exact, vague, or guesses a filename outright — the enforcement
   mechanism never inspects the query's wording at all, only the chunk's
   scope and the caller's role.
2. **Retrieval quality — whether a natural-language question surfaces the
   *right* candidate chunk at all — is a separate, wording-dependent
   property**, bounded by how well `all-MiniLM-L6-v2` (a small, local,
   free embedding model) captures semantic similarity between a question
   and a chunk's content. This is a property of the embedding model, not
   of the access-control system, and every RAG system built on a
   small embedding model shares it.

Tests whose purpose is to confirm claim 1 (can a given role retrieve a
*known* chunk — i.e., "does RLS correctly permit this," not "can a
contractor word a question well enough to find it") were rewritten to pull
the chunk's own exact embedding directly (via an authenticated client's
own read access to it) and pass that embedding straight to
`match_chunks()`, rather than relying on a natural-language paraphrase to
close a similarity gap that has nothing to do with security.

## Why this matters

Conflating these two claims would make the test suite either misleading or
untrustworthy, depending on which way it failed:

- If a "can senior retrieve X" test is written as a paraphrase query and
  passes, it's genuinely ambiguous whether it passed because RLS correctly
  permitted it, or because the paraphrase happened to score well enough —
  the test doesn't isolate which thing it's actually checking.
- If the same test fails (as it did, three times, before the fix), the
  honest conclusion is "the embedding model didn't rank this paraphrase
  highly enough" — a retrieval-quality finding — not "RLS is broken." Three
  real failed test runs were spent confirming this distinction empirically
  before concluding it was a retrieval-quality limitation, not a security
  bug: direct SQL confirmed the chunk's embedding was present, correct, and
  ranked itself #1 against its own vector, with the other 14 nearest
  results all being unrelated, correctly-contractor-scoped chunks — meaning
  this specific chunk's content sits in a part of embedding space that
  these particular paraphrases simply don't land near.

Isolating the embedding-quality variable out of the RLS-correctness tests
means a future regression in `match_chunks()` or the RLS policy will show
up as a clear, unambiguous test failure — not get lost in "well, maybe the
paraphrase was just bad this time."

## Consequence worth stating explicitly (the harder, unresolved point)

The project owner's underlying concern — a contractor doesn't know a
file's exact wording either, so isn't the "indirect query" testing
approach itself somewhat artificial? — surfaces a real and more
fundamental limitation that this ADR does not resolve: **every scope
classification in `docs/access_design.md` was made by hand, by someone who
already knows the corpus intimately.** This project proves that a *given*
classification is enforced correctly and provably. It does not — and
currently cannot — determine what the classification *should be* for a
codebase nobody has read yet. That's a materially harder, unsolved
problem, deliberately deferred to Phase 4 rather than claimed as solved by
anything built so far. The RLS-vs-retrieval-quality distinction in this
ADR is necessary but not sufficient context for that larger question.

## Alternative considered and rejected

Leaving the paraphrase-based tests as-is and treating the three failures
as flaky tests to retry until they happened to pass was rejected outright
— that would have hidden a real, explainable limitation behind
test-flakiness noise, which is worse than documenting the limitation
honestly.
## Addendum (2026-10-06): the embedding-model attribution above was unverified

The Phase 2 retrieval eval found that an IVFFlat index (`probes = 1`, built before
data was loaded) was truncating results to 2-4 candidates for many real questions
(see ADR 0005). The lab-notes example recorded in this ADR, where the chunk missed
the top 50 for three paraphrases, was attributed to the embedding model's limits.
That cause was never isolated, and the index is a likelier explanation. After the
index was removed, q015 (an indirect question about the same chunk, avoiding its
distinctive terms) retrieves `lab_notes.md` within the top 5 for senior_engineer.
The three original paraphrases were not re-run, so this ADR's example stands as
unverified. What remains valid: RLS enforcement is wording-independent, and
retrieval quality is a separate property that should be measured, which `eval/`
now does.
