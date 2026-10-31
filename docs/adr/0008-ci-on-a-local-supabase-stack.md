# ADR 0008: CI runs against a local Supabase stack, not a second cloud project

Status: Accepted (decision); implementation pending
Date: 2026-10-30

## Context

The tests that matter most here (the RLS adversarial suite, the `query_logs` policies,
the `/ask` and `/logs` endpoints) need real Postgres, Supabase Auth and PostgREST,
because the guarantee depends on `auth.uid()`, real JWTs and the caller's Postgres role.
A plain Postgres service container cannot provide that. The obvious alternative is a
second hosted Supabase project for CI, but that means a new cloud project, secrets in
GitHub, and a shared remote database that tests mutate.

Two more constraints: every `/ask` makes a paid Claude call, and the secret key bypasses
RLS and is used by test cleanup, so CI must never hold the hosted project's key.

## Decision

1. **Database:** CI starts the Supabase CLI's local stack in Docker (`supabase start`)
   on the runner, applies migrations 001-004 and 006-010 with `psql`, skips 005 (a seed
   file; the fake users come from `scripts/setup_local_test_users.py`), and ingests
   `corpus_repos/` (tracked in git, 1.7 MB) with `scripts/ingest_repo.py`. A second hosted
   project (Option B) is the fallback only if the local stack fails on a runner for a
   reason we cannot fix.
2. **Secrets:** the keys come from the throwaway local stack. The hosted project's keys
   are never given to CI.
3. **Generation:** stubbed by default. Tests that call the real API carry the
   `live_llm` marker and run only on pushes to `main`, with a spend-limited API key.
   The Anthropic client is created lazily, so a run without a key works.
4. **Eval:** `scripts.eval_retrieval` runs as a `main`-only gate. It already exits
   non-zero on any leak. A hit@5 floor will be set below the baseline once a runner
   reproduces the baseline numbers; the value is not chosen yet.
5. **Corpus validation:** `scripts.validate_eval` fails, rather than skips, when a local
   corpus file is missing, so a fresh checkout cannot pass with the phrase check disabled.

## Evidence

On a Mac (not a runner):

- The local stack publishes an ES256 key at the JWKS path `app/auth.py` uses, and a
  local token verifies with the same python-jose settings. No auth code change needed.
- Migrations 001-004 and 006-008 applied with no errors; 009 and 010 were applied and
  checked afterwards.
- After ingestion the local database matches hosted: 155 documents, 2,716 chunks, 10
  senior-only documents.
- The 15 adversarial tests pass on the local stack, and the retrieval eval reproduces
  the hosted baseline exactly (contractor hit@5 0.93, senior 0.92, same precision,
  recall and MRR) with 0 leaks in 56 retrievals.
- Building CI's database this way found real problems: the `roles` table had RLS off and
  broad grants on the local stack (009), the other three tables kept unused privileges
  (010), and one test hard-coded a chunk UUID that changes on re-ingest.

## Not established / limits

- Nothing has run on a GitHub Actions runner. Linux, image pull time and total runtime
  are unmeasured.
- Library versions on a runner may differ from the Mac and change chunking or
  embeddings slightly. The eval-as-gate decision depends on the runner reproducing the
  baseline; if it does not, the floor needs a margin or the gate needs rethinking.
- The local stack can differ from hosted in ways not yet seen. On hosted, `roles` already
  had RLS on although the migration never enabled it; the migrations are the source of
  truth for CI, hosted is not.
- A 28-question eval cannot detect changes under about two questions, so the gate catches
  leaks and large regressions, not small ones.
- Observed locally: the stack's container names come from the project folder name, and
  the CLI cannot read keys while Docker is stopped. CI should start the stack and the app
  in the same job, stack first, because `app/auth.py` caches the JWKS for the process
  lifetime.
- The test users' credentials are hard-coded in `tests/conftest.py` and duplicated in the
  setup script. That is acceptable only because the stack is throwaway and the script
  refuses non-local URLs.
- The `live_llm` marker exists, but no test is tagged with it yet; the `/ask` tests
  still make real calls until generation is stubbed.