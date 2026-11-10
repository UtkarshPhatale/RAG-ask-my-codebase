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

## CI job inventory (2026-11-01)

What each test file needs, read from the files themselves. `tests/conftest.py` has a
session-wide autouse fixture that connects to the database with the secret key, so any
run that keeps conftest needs a database; `--noconftest` skips it.

| File | Needs | Planned CI job |
|---|---|---|
| test_generation_lazy.py | nothing (monkeypatch only) | `unit` (running since 2026-10-31) |
| test_embedding_concurrency.py | torch, sentence-transformers, model download; no database; run with `--noconftest` | separate job with model cache |
| test_auth_role_lookup.py | real login token from Supabase Auth (stubs only the role lookup) | local stack |
| test_query_log_writer.py | database (module fixture connects; 1 of 3 tests is database-free) | local stack |
| test_ask_endpoint.py | database; currently makes real Claude calls | local stack, generation stubbed |
| test_rbac_adversarial.py | database and ingested corpus | local stack |
| test_roles_lockdown.py | database | local stack |
| test_logs_endpoint.py, test_query_logs_rls.py | not re-read for this note; expected to need the database | local stack |

Consequence: the database-free job covers little, and most of the value arrives with
the local-stack job. The unit workflow has run green on a GitHub runner (runs #1 to #4, 13-15 s each, as of 2026-11-05).

## Hosted project pause (2026-11-05)

Supabase paused the free-tier project RAG-ask-my-codebase on 2026-11-04 after 7 days
without activity (last hosted run: 2026-10-28). It can be unpaused from the dashboard
within 90 days. Consequences:

- CI is unaffected: it uses a local stack (ADR 0008) and never touches the hosted project.
- Anything run against `.env` (check.sh, the hosted test runs) fails until it is unpaused.
- The demo recording needs a live backend: unpause the hosted project or point the
  frontend at the local stack.
- Deployment checklist: a free-tier project that pauses after a week will break a public
  live URL. Options to evaluate: a keep-alive request, or a paid plan. Not decided.

## CI status (2026-11-09)

- `unit` ran green four times (#1 to #4, 13-15 s each).
- Runs #5 to #8 and `local-stack` #1 failed within 2-4 s. The job annotation read "The job
  was not started because your account is locked due to a billing issue", and the
  Licensing page showed the account disabled for a failed charge. The repo is public and
  the plan is GitHub Free, with $0 billable Actions usage, so this was an account billing
  state, not a workflow bug.
- After the payment method was updated on 2026-11-09, a re-run of `unit` #8 went green.
- Still not established: `local-stack` has never run, so Supabase startup on a runner,
  its time, and the JWKS algorithm on Linux are unmeasured.
- 2026-11-09: `local-stack` #2 ran green on a GitHub runner (start-stack 1m 48s). Not yet
  recorded: the JWKS algorithm on Linux, migrations, users and ingestion on the runner.
  