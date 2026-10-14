# ADR 0007: Query logs live in a Postgres table protected by RLS

Status: Accepted
Date: 2026-10-12

## Context

Each /ask call was appended to a local `logs/queries.jsonl`. Two problems: the file does
not survive redeploys on hosts with ephemeral disks (the planned free-tier deployment),
and exposing it to a viewer would need an `if role == "senior_engineer"` check in
application code, which is the pattern this project argues against.

## Decision

Store one row per /ask in `query_logs` (migration 008) and let Postgres decide who can
read it, exactly as with `chunks`:

- A user may insert a row only about themselves, and the `role` column must match their
  real role in `user_roles` (otherwise a contractor could claim to be senior).
- A user may read their own rows; a senior_engineer may read all rows.
- Nobody can update or delete rows. `anon` has no access.
- The row is written by a background task after the response, using the caller's own
  user-scoped client, so these policies apply to the writer too. Any error in the writer
  is caught and logged; it never reaches the user.
- Each retrieved chunk is recorded as repo/path/scope/similarity, without chunk text.

## Evidence

Seven tests run the real policies as real users (spoofed user, spoofed role, user with
no role, read isolation, immutability, anon). Three more check the writer: a row carries
the caller's identity and counts matching the response, the senior-only count matches
what was actually returned, and a failing writer cannot break an answer. A live check
showed contractor rows with 0 senior-only chunks and senior rows with 2 for the same
question, matching what the UI displayed.

## Not covered / limits

- Chunks that RLS blocked cannot be logged: they never reach the application. Counting
  them would need a privileged out-of-band comparison, which is a different tool.
- The row is written after the response; if the process dies in between, one row is lost.
- Questions are stored in plaintext. Fine for a demo; a real product needs a retention
  and privacy decision.
- Each /ask adds one database write (off the response path).
- Users cannot delete log rows by design, so the tests clean up their own marker rows
  with the secret key. That is the only use of the secret key in these tests.

## Read side (added with GET /logs)

`GET /logs` returns the rows the caller's own client can read, newest first, with a
validated `limit` (1-200). The handler contains no role check: contractors receive their
own rows and seniors receive everyone's because RLS decides, the same way it does for
chunks. Tests call the endpoint with real tokens and check both visibility sets, the
response shape, ordering, limit validation and authentication.
