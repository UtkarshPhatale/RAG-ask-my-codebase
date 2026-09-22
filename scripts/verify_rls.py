"""
RLS verification script: proves RLS enforcement at the database layer,
with no application code involved beyond this script itself.

How it works:
  1. Logs in as the contractor test user via Supabase Auth -> gets a JWT
     scoped to that user's identity.
  2. Uses that JWT (not the service_role key) to query `chunks`, paging
     through the full result set (Supabase's REST API caps a single
     request at 1000 rows, so anything past that silently gets cut off
     if you don't paginate -- this matters now that the corpus is 2,795
     chunks, well past that cap).
  3. Repeats for the senior_engineer test user.
  4. Prints both result sets' full counts and confirms the contractor's
     results never include a senior_engineer-scoped chunk, even though
     both users are running the exact same query (`select * from chunks`).

This is the actual proof: Postgres itself is filtering rows based on the
querying user's identity, via the RLS policies in
db/migrations/004_rls_policies.sql -- not something the application chose
to hide.

Setup:
  pip install supabase python-dotenv
  Fill in .env with SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY (NOT the
  secret key -- this script deliberately uses the public/anon-equivalent
  key, the same one your real frontend/backend will use, so RLS applies).

Usage:
  python scripts/verify_rls.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_PUBLISHABLE_KEY = os.environ["SUPABASE_PUBLISHABLE_KEY"]

CONTRACTOR_EMAIL = "contractor-test@example.com"
CONTRACTOR_PASSWORD = "TestPass123!"

SENIOR_EMAIL = "senior-test@example.com"
SENIOR_PASSWORD = "TestPass456!"

PAGE_SIZE = 1000  # Supabase REST API's default max rows per request


def query_all_chunks_as(email: str, password: str) -> list[dict]:
    """Log in as a specific user and fetch every row in `chunks` visible
    to THEIR session (their JWT), paginating past the 1000-row API cap
    so the count is never silently truncated."""
    client: Client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)

    auth_response = client.auth.sign_in_with_password(
        {"email": email, "password": password}
    )
    print(f"Logged in as {email} (user_id={auth_response.user.id})")

    all_rows: list[dict] = []
    offset = 0
    while True:
        result = (
            client.table("chunks")
            .select("*, documents(path, repo)")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        batch = result.data
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return all_rows


def main():
    print("=" * 70)
    print("Querying ALL chunks AS the contractor test user (paginated)")
    print("=" * 70)
    contractor_chunks = query_all_chunks_as(CONTRACTOR_EMAIL, CONTRACTOR_PASSWORD)
    print(f"  -> contractor sees {len(contractor_chunks)} chunk(s) total\n")

    print("=" * 70)
    print("Querying ALL chunks AS the senior_engineer test user (paginated)")
    print("=" * 70)
    senior_chunks = query_all_chunks_as(SENIOR_EMAIL, SENIOR_PASSWORD)
    print(f"  -> senior_engineer sees {len(senior_chunks)} chunk(s) total\n")

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    leaked = [c for c in contractor_chunks if c["required_scope"] == "senior_engineer"]
    if leaked:
        print(f"FAIL: contractor's result set contains {len(leaked)} "
              f"senior_engineer-scoped chunk(s). RLS policy is not "
              f"enforcing correctly -- do not proceed until this is fixed.")
        for c in leaked[:5]:
            doc = c.get("documents") or {}
            print(f"    leaked: {doc.get('repo')}/{doc.get('path')}")
    else:
        senior_only_count = sum(
            1 for c in senior_chunks if c["required_scope"] == "senior_engineer"
        )
        print(f"PASS: contractor's result set (n={len(contractor_chunks)}) "
              f"contains zero senior_engineer-scoped chunks, out of "
              f"{senior_only_count} such chunks that exist in the corpus "
              f"and ARE visible to the senior_engineer session. Confirmed "
              f"by directly querying the database as each real user, not "
              f"by trusting application logic to filter results after "
              f"the fact.")


if __name__ == "__main__":
    main()