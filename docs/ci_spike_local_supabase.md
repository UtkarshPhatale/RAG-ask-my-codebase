# Spike: can a local Supabase stack stand in for the hosted project in CI?

Date: 2026-10-20. Status: first question answered; the rest is open.

## Question

CI needs Postgres + Supabase Auth + PostgREST to run the RLS tests. Running the
Supabase CLI's local stack in Docker would avoid a second cloud project and any
database secrets in CI, but only if it behaves like the hosted project where
app/auth.py depends on it.

## What was done

- Installed the Supabase CLI (2.121.0) and ran `supabase init` and `supabase start`
  in a scratch folder outside this repo (Docker Desktop 27.3.1, macOS).
- Fetched `/auth/v1/.well-known/jwks.json` from the local stack.
- Signed up a throwaway user, signed in with the local publishable key
  (`sb_publishable_...`), and verified the returned access token with python-jose
  using the same settings as `_verify_jwt` in app/auth.py: look up the key by
  `kid`, `algorithms=["ES256"]`, `audience="authenticated"`.

## What was established

- The local stack publishes an ES256 (P-256) key at the same JWKS path that
  app/auth.py uses.
- A token from the local stack carried an ES256 header with a `kid` matching that
  key, and verified successfully (audience `authenticated`, role `authenticated`).
- So no change to the verification code is needed for this path, and CI does not
  have to accept additional algorithms.

## What is NOT established

- The check used a copy of the verification logic, not app/auth.py itself.
- Migrations 001-008 have not been run on the local stack (005 is a seed file
  that must not run in CI). pgvector availability and the search_path behaviour
  that match_chunks() relies on are untested locally.
- RLS enforcement through the local PostgREST with a user JWT is untested.
- Not tried on a GitHub Actions runner (Linux); only on a Mac.
- The sign-in response included a `weak_password` field whose contents were not
  inspected.
- Whether the local signing key and `kid` stay the same across restarts: app/auth.py
  caches the JWKS for the process lifetime, so CI should start the stack before the app.

## Next

Apply the migrations to the local stack and run the RLS suite against it.
If that fails for a reason the local stack cannot fix, fall back to a second
hosted project (Option B), and record that in ADR 0008.