# ADR 0001: `match_chunks()` is `SECURITY INVOKER`, not `SECURITY DEFINER`

## Status
Accepted. This is the single load-bearing decision in the project.

## Context

Retrieval is implemented as a Postgres function, `match_chunks()`, called
via Supabase's RPC mechanism rather than a raw `SELECT` from the client.
Postgres functions default to `SECURITY DEFINER` in most tutorials and in
Supabase's own example code for pgvector similarity search — including,
notably, the official Supabase docs' own vector-search function examples,
which is precisely why this needed a deliberate decision rather than
accepting the default.

A `SECURITY DEFINER` function runs with the privileges of the user who
*created* it — typically a superuser-equivalent role when created through
the Supabase dashboard or migrations. A `SECURITY INVOKER` function runs
with the privileges of the user who *calls* it.

## Decision

`match_chunks()` is declared `SECURITY INVOKER`.

## Why this matters

Row-Level Security policies are evaluated against the role executing the
query — not the role that owns the function being called. If `match_chunks()`
were `SECURITY DEFINER` (the Supabase-example default), it would execute as
its owner — a role RLS policies typically don't restrict — and RLS would be
silently bypassed. Every row in `chunks` would come back to every caller,
contractor and senior engineer alike, and the function would still *look*
correct in a quick test: it would return results, no errors, nothing
visibly broken. The leak would only show up under the specific adversarial
condition this project exists to test for — which is exactly the kind of
bug that survives code review and ships to production.

`SECURITY INVOKER` means the exact same `SELECT ... WHERE required_scope ...`
RLS policy that governs a plain `SELECT * FROM chunks` also governs every
row this function considers, because as far as Postgres is concerned, the
calling user is the one running the query — there is no privilege
escalation happening through the function boundary at all.

## Consequence worth stating explicitly

Because enforcement lives here and not in Python, there is no corresponding
code path in `app/retrieval.py` or anywhere else in the application that
filters by `required_scope`. That absence is intentional, not an oversight
— `retrieve_chunks()` calls `.rpc("match_chunks", ...)` on a Supabase client
scoped to the calling user's own session (via `postgrest.auth(token)`,
publishable key) and trusts the database completely. If someone reviewing
this code expects to find a manual scope check in the Python layer and
doesn't, that's the system working as designed, not a gap.

## Alternative considered and rejected

Application-layer filtering (`if chunk.scope == "senior_engineer" and
user.role != "senior_engineer": skip`) was the implicit default this
project was built to avoid. It requires every new code path that touches
`chunks` to remember to add the check — one new feature, one forgotten
`if`, one leak. Pushing enforcement into the database means there is no
code path that can forget, because there is no code path involved in the
decision at all.