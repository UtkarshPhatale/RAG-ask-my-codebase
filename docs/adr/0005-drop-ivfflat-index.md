# ADR 0005: Drop the IVFFlat index on chunks.embedding

Status: Accepted
Date: 2026-10-06

## Context

Migration 003 created an IVFFlat index (`lists = 100`) on `chunks.embedding` at
table creation, before any chunks were ingested. IVFFlat clusters vectors around
centroids learned from the data present when the index is built, so these
centroids were trained on an empty table. At query time the default
`ivfflat.probes = 1` searches only the single nearest cluster, and RLS then
filters that small candidate set.

The Phase 2 retrieval eval (28 hand-labeled questions, both roles, k=5) exposed it:

- `match_chunks` asked for 5, 50 and 200 rows returned only 2 and 4 rows for two
  real questions, regardless of the requested count.
- "What is the PTO accrual policy and how much is the on-call stipend?" had a best
  similarity of 0.044 although `company_handbook.txt` contains that content almost
  verbatim.

It went unnoticed because the Phase 1 tests that need a known chunk look it up with
that chunk's own exact embedding, which always falls in the chunk's own cluster.
Natural-language questions did not.

## Decision

Drop the index (`db/migrations/007_drop_ivfflat_index.sql`) and use exact
nearest-neighbor search. At ~2.7k chunks of 384 dimensions a sequential scan has
100% recall. Measured after the change: median 83 ms, p95 140 ms per
`match_chunks` call (n=56), including the network round trip from a laptop to
Supabase. There is no "before" latency to compare, since the index is gone.

## Evidence (same questions, labels, model, chunking and k=5; only the index changed)

| Role (questions scored) | Metric | Before | After |
|---|---|---|---|
| contractor (15) | hit@5 | 0.67 | 0.93 |
| | phrase@5 | 0.13 | 0.73 |
| | recall@5 | 0.63 | 0.87 |
| | MRR | 0.42 | 0.77 |
| senior_engineer (25) | hit@5 | 0.44 | 0.92 |
| | phrase@5 | 0.08 | 0.72 |
| | recall@5 | 0.42 | 0.86 |
| | MRR | 0.25 | 0.78 |

Raw per-question results: `eval/results/baseline_ivfflat_k5.json` (before) and
`eval/results/after_exact_k5.json` (after).

## Alternatives considered

- **Rebuild IVFFlat after loading data**, with `lists` near sqrt(rows) and higher
  `probes`: adds tuning knobs that only approximate what an exact scan already does
  at this size, and needs rebuilding after every re-ingestion.
- **HNSW**: needs no training step, so it avoids the empty-table failure, and is the
  better choice if the corpus grows. But it is approximate, and RLS filters after the
  index scan, so a restrictive policy can return fewer rows than requested. Not
  worth that risk at 2.7k chunks.
- **Keep as is**: rejected; it silently returned 2-4 mostly irrelevant chunks.

## Consequences

- Security is unaffected. RLS filters after any index scan, so the index could only
  remove rows, never expose them. The 20 security tests pass before and after.
- Earlier zero-leak results are weaker evidence than they looked: before the fix a
  contractor query returned 2-4 mostly irrelevant chunks. After the fix, the eval's
  leak check finds 0 leaks in 56 retrievals, including 10 contractor questions that
  ask directly about senior-only files (senior retrieves the expected file for 9 of
  those 10).
- Retrieval quality is now a property of the embedding model and chunking only, and
  is measurable. Remaining misses at k=5: q003 (both roles) and q016 (senior). The
  gap between hit@5 (0.92) and phrase@5 (0.72) suggests the right file is often
  retrieved but not the chunk containing the answer.
- With 28 questions, one question is worth 4-7 points of a role's hit rate.
  Differences smaller than about two questions are noise, which matters when judging
  later changes such as hybrid search.
- If the corpus grows by orders of magnitude, create an index after loading data and
  re-run the eval before trusting it.
