# Spike: can a local Supabase stack stand in for the hosted project in CI?

Date: 2026-10-20 (auth check), 2026-10-21 (migrations). Status: auth and schema
checks passed locally; RLS behaviour through PostgREST and a Linux runner are open.

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
  user_roles and query_logs.
- The `weak_password` field in the sign-in response was `None`; nothing to act on.

## What is NOT established

- The auth check used a copy of the verification logic, not app/auth.py itself.
- RLS enforcement through the local PostgREST with real user JWTs is untested.
  Applying the migrations only shows the policies exist, not that they behave.
- Not tried on a GitHub Actions runner (Linux); only on a Mac.
- Whether the local signing key and `kid` stay the same across restarts:
  app/auth.py caches the JWKS for the process lifetime, so CI should start the
  stack before the app.

## Open observation: `roles` has row-level security disabled

`pg_tables` shows `rowsecurity = f` for `roles`. Migration 001 creates the table
and never enables RLS, so this follows from the migrations and is expected to be
the same on the hosted project (not checked there). Whether `anon` or
`authenticated` can write to it through PostgREST depends on Supabase's default
grants and has NOT been tested. The likely impact is small (policies compare
against literal role names and foreign keys block deleting roles in use), but it
is a least-privilege gap and should be tested and, if confirmed, closed by a new
migration. Not changed here.

## Next

Create the fake test users and their user_roles rows on the local stack, ingest
the corpus, and run the RLS suite against it. If that fails for a reason the
local stack cannot fix, fall back to a second hosted project (Option B) and
record that in ADR 0008.