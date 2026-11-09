# Running the local Supabase stack

Everything here was learned the hard way during Phase 3. The local stack is a throwaway
copy of the database for tests; the hosted project is never needed for it.

## Rules

- **Docker Desktop must be running.** With Docker stopped, `supabase status` cannot read
  the keys and the env helper prints `local stack keys not found`.
- **Container names come from the folder name** (`supabase_db_<folder>`). The stack runs
  from `~/supabase-spike`. Do not rename or move that folder while the stack is up.
- **Do not stop and restart from that folder without checking `supabase/config.toml`.**
  It is missing there; run `supabase init` first, and note the stack would start with
  defaults.
- **Always source the env file inside parentheses and chain with `&&`**, so a failed
  `source` stops the command instead of silently falling back to `.env` (the hosted
  project). This happened once and ran tests against hosted by mistake:

```bash
  ( source ~/.supabase_local_env.sh && python -c "from app.config import SUPABASE_URL; print(SUPABASE_URL)" && python -m pytest tests/test_rbac_adversarial.py -q )
```

  The first line printed must be `http://127.0.0.1:54321`.
- **Never run `scripts/ingest_repo.py` without checking the URL.** It has no duplicate
  check, so a run against the wrong database inserts every chunk a second time.
- `scripts/setup_local_test_users.py` refuses to run unless the URL is local.

## Build the database from scratch

Migration 005 is a seed file and is skipped on purpose.

```bash
for f in db/migrations/001_*.sql db/migrations/002_*.sql db/migrations/003_*.sql db/migrations/004_*.sql db/migrations/006_*.sql db/migrations/007_*.sql db/migrations/008_*.sql db/migrations/009_*.sql db/migrations/010_*.sql; do echo "== $f"; docker exec -i supabase_db_supabase-spike psql -U postgres -v ON_ERROR_STOP=1 -q < "$f" || break; done
```

Then create the fake users and ingest the four repos (see the URL rule above).
Expected totals: 155 documents, 2,716 chunks, 10 senior-only documents.

## Hosted project

Supabase pauses the free-tier project after 7 days of inactivity (paused 2026-11-04).
Unpause it from the dashboard before running anything against `.env`.