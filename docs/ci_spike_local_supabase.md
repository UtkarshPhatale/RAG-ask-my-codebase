# Spike: can a local Supabase stack stand in for the hosted project in CI?

Date: 2026-10-20 (auth check), 2026-10-21 (migrations), 2026-10-23 (roles
follow-up). Status: auth and schema checks passed locally; RLS behaviour through
PostgREST on the local stack and a Linux runner are open.

## Question

CI needs Postgres + Supabase Auth + PostgREST to run the RLS tests. Running the
Supabase CLI's local stack in Docker would avoid a second cloud project and any
database secrets in CI, but only if it behaves like the hosted project where
the app depends on it.

## What was done

- Installed the Supabase CLI (2.121.0) and ran `supabase init` and `supabase start`
  in a scratch folder outside this repo (Docker Desktop 27.3.1, macOS).
- Fetched `/auth/v1/.well-known/jwks.json` from the local stack.
- Signed up a throwaway user, signed in with the local publishable key
  (`sb_publishable_...`), and verified the returned access token with python-jose
  using the same settings as `_verify_jwt` in app/auth.py: look up the key by
  `kid`, `algorithms=["ES256"]`, `audience="authenticated"`.
- Applied migrations 001-004 and 006-008, in order, to the local database with
  `psql -v ON_ERROR_STOP=1`. Migration 005 (seed fixtures) was deliberately skipped.

## What was established

- The local stack publishes an ES256 (P-256) key at the same JWKS path that
  app/auth.py uses. A token from it carried an ES256 header with a matching `kid`
  and verified (audience `authenticated`, role `authenticated`). No change to the
  verification code is needed, and CI does not have to accept extra algorithms.
- Migrations 001-004 and 006-008 apply with no errors on the local stack, so
  pgvector, `match_chunks()` and the `auth.uid()` policies all create fine.
  The only output was the expected ivfflat "little data" NOTICE from 003, whose
  index 007 drops.
- After the migrations, row-level security is on for chunks, documents,
  user_roles and query_logs, and off for roles (see below).
- The `weak_password` field in the sign-in response was `None`; nothing to act on.

## Finding: `roles` was writable on the local stack; hosted differed

- Local stack: `roles` had RLS off (migration 001 never enables it) and full
  grants for anon and authenticated. An anonymous insert returned HTTP 201.
- Hosted project: `roles` already had RLS on and zero policies, so writes were
  denied by default, despite the same broad grants. This was read from
  `pg_tables` and `pg_policies`; no write was attempted against hosted.
- So the migration files and the hosted database had drifted: hosted had RLS on
  for a table whose migration never enables it. The cause is unknown.
- Migration 009 (RLS on, privileges revoked) fixes the files. On hosted, only
  the REVOKE changed anything. After it, the grants query shows no `roles` rows,
  and `test_roles_lockdown.py` plus `test_rbac_adversarial.py` pass (17 passed)
  against hosted. The authenticated-user lockdown test has passed on hosted but
  has not yet run on the local stack.
- Lesson: the grants query alone does not show RLS state, so I wrongly assumed
  hosted matched local. Check `pg_tables.rowsecurity` and `pg_policies` too.

## Update 2026-10-28: local stack results

- Fake users (contractor, senior_engineer, one with no role) are created on the
  local stack by `scripts/setup_local_test_users.py`, which refuses to run
  unless the URL is local.
- The corpus was ingested into the local stack: 155 documents, 2,716 chunks, 10
  senior-only documents, the same totals as hosted.
- `tests/test_rbac_adversarial.py` passes 15/15 against the local stack, and the
  retrieval eval reproduces the hosted baseline exactly (contractor hit@5 0.93,
  senior 0.92, same precision, recall and MRR) with 0 leaks in 56 retrievals.
- Migration 010 (SELECT-only grants on chunks, documents, user_roles) is applied
  on both local and hosted. On hosted, after the revoke, the full `check.sh`
  passes (46 tests, question set OK, eval unchanged).

## What is NOT established

- The auth check used a copy of the verification logic, not app/auth.py itself
  (the adversarial suite does exercise app/auth.py against the local stack).
- Not tried on a GitHub Actions runner (Linux); only on a Mac.
- Whether the local signing key and `kid` stay the same across restarts:
  app/auth.py caches the JWKS for the process lifetime, so CI should start the
  stack before the app.
- Whether the local stack's results match hosted on a different machine or
  library versions.

## Next

Wire this into GitHub Actions (ADR 0008 first), starting with the tests that
need no database.