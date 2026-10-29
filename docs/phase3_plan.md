# Phase 3 plan: engineering maturity

Written 2026-10-28. This records the agreed order and the rules for judging
retrieval experiments, before any experiment runs.

## Order

1. CI: GitHub Actions running the tests, the RLS adversarial suite, and the
   eval (as a main-only gate). Local Supabase stack in Docker, no second cloud
   project. Generation is stubbed by default; real Claude calls run only on main
   with a spend-limited key (`live_llm` marker).
2. Retrieval experiments, each judged against the Phase 2 baseline below.
3. Docker and live deployment on a free tier, alternating with the experiments.

## Baseline to beat (k = 5, frozen v1 eval, 28 questions)

| role | hit@5 | phrase@5 | precision@5 | recall@5 | MRR |
|---|---|---|---|---|---|
| contractor (n=15) | 0.93 | 0.73 | 0.45 | 0.87 | 0.77 |
| senior (n=25) | 0.92 | 0.72 | 0.39 | 0.86 | 0.78 |

Leak check: 0 leaks in 56 retrievals. Reproduced against the hosted project on
2026-10-27.

## Rules for the experiments

- Predictions are written down before each run (as for the k-sweep).
- The v1 labels stay frozen. New questions go in a separate v2 file, because 28
  questions cannot detect a change smaller than about two questions (each is
  worth 4-7 points).
- A technique is kept only if it beats the baseline by more than that noise
  floor on the v2 set. Otherwise it is documented as a negative result.
- Any new retrieval path must keep the RLS guarantee. Keyword search inside
  Postgres (full-text search on `chunks`) inherits RLS the way `match_chunks()`
  does. A separate in-memory index in Python would not, and would amount to
  app-layer filtering. That decision gets its own ADR before any code.

## Candidate experiments, in order

1. Prepend the file path to each chunk before embedding (needs re-ingestion).
   Targets the finding that 500-character chunks fragment docstrings and that
   docs restating the code outrank the source.
2. Hybrid BM25 + vector search. Targets questions whose answers sit at rank 20+
   (q005, q010), which reranking alone cannot recover: phrase@5 is 0.72 and
   phrase@20 is 0.92.
3. A reranker, only if hybrid search justifies it.

## Not established

- Whether any of these beats the baseline. Nothing has been run.
- Whether extra, unlabeled chunks in the top 5 hurt answer quality (no
  answer-quality eval exists).
- Whether the local-stack eval reproduces the hosted numbers (to be measured
  before the eval is used as a CI gate).