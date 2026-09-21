"""
Day 4 verification script: proves RLS enforcement at the database layer,
with no application code involved beyond this script itself.

How it works:
  1. Logs in as the contractor test user via Supabase Auth -> gets a JWT
     scoped to that user's identity.
  2. Uses that JWT (not the service_role key) to query `chunks`.
  3. Repeats for the senior_engineer test user.
  4. Prints both result sets so you can see the row counts and content
     differ -- specifically, that the contractor's results never include
     the senior_engineer-scoped chunk, even though both users are running
     the exact same query (`select * from chunks`).

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


def query_chunks_as(email: str, password: str) -> list[dict]:
    """Log in as a specific user and query chunks using THEIR session
    (their JWT), so RLS policies apply exactly as they would for a real
    authenticated request."""
    client: Client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)

    auth_response = client.auth.sign_in_with_password(
        {"email": email, "password": password}
    )
    print(f"Logged in as {email} (user_id={auth_response.user.id})")

    # This query runs AS this user -- RLS policies filter it server-side.
    result = client.table("chunks").select("*, documents(path, repo)").execute()
    return result.data


def main():
    print("=" * 70)
    print("Querying chunks AS the contractor test user")
    print("=" * 70)
    contractor_chunks = query_chunks_as(CONTRACTOR_EMAIL, CONTRACTOR_PASSWORD)
    for c in contractor_chunks:
        print(f"  [{c['required_scope']}] {c['content'][:60]}...")
    print(f"  -> contractor sees {len(contractor_chunks)} chunk(s)\n")

    print("=" * 70)
    print("Querying chunks AS the senior_engineer test user")
    print("=" * 70)
    senior_chunks = query_chunks_as(SENIOR_EMAIL, SENIOR_PASSWORD)
    for c in senior_chunks:
        print(f"  [{c['required_scope']}] {c['content'][:60]}...")
    print(f"  -> senior_engineer sees {len(senior_chunks)} chunk(s)\n")

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    contractor_has_senior_chunk = any(
        c["required_scope"] == "senior_engineer" for c in contractor_chunks
    )
    if contractor_has_senior_chunk:
        print("FAIL: contractor's result set contains a senior_engineer-scoped "
              "chunk. RLS policy is not enforcing correctly -- do not proceed "
              "until this is fixed.")
    else:
        print("PASS: contractor's result set contains zero senior_engineer-"
              "scoped chunks, confirmed by directly querying the database "
              "as that user (not by trusting application logic to filter "
              "results after the fact).")


if __name__ == "__main__":
    main()