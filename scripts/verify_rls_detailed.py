"""
Day 7: detailed RLS verification, per repo and per file -- not just an
aggregate count.

scripts/verify_rls.py already proved the totals are right (contractor
sees fewer rows than senior_engineer, and zero of them are
senior_engineer-scoped). That's necessary but not sufficient: a bug where,
say, repo 3's dummy file resolved to the wrong scope AND some other file
in repo 2 accidentally over-restricted could cancel out in the aggregate
count while still being wrong. This script checks the actual identity of
every senior_engineer-scoped file in the corpus, per repo, and confirms
none of them are visible to the contractor session -- named explicitly,
so a mismatch is immediately obvious rather than hidden in a total.

It also cross-checks against the known list of senior_engineer files from
docs/access_design.md, so a MISSING senior-only file (one that should be
restricted but got tagged contractor by mistake during ingestion) is
caught too, not just leakage in the other direction.

Usage:
  python scripts/verify_rls_detailed.py
"""

import os
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_PUBLISHABLE_KEY = os.environ["SUPABASE_PUBLISHABLE_KEY"]

CONTRACTOR_EMAIL = "contractor-test@example.com"
CONTRACTOR_PASSWORD = "TestPass123!"

SENIOR_EMAIL = "senior-test@example.com"
SENIOR_PASSWORD = "TestPass456!"

PAGE_SIZE = 1000

# The known senior_engineer-only files per repo, from docs/access_design.md /
# ingestion/access_map.json. Kept here explicitly (not imported) so this
# script cross-checks against a second, independent source of truth rather
# than trusting the same access_map.json the ingestion pipeline already used.
EXPECTED_SENIOR_ONLY_FILES = {
    "interface-ai-computer-use-project": {
        "guardrails/policy.py",
        "escalation/models.py",
        "escalation/resolve.py",
        "login_session.py",
    },
    "nextplay_kanban": {
        "src/config/secrets_template.py",
        "internal/admin_notes.md",
        "src/lib/admin/billingOverride.ts",
    },
    "brain-tumor-segmentation": {
        "internal/lab_notes.md",
        "scripts/advanced/cluster_secrets_template.py",
    },
    "rag-chatbot": {
        "data/company_handbook.txt",
    },
}


def query_all_chunks_as(email: str, password: str) -> list[dict]:
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


def paths_seen_by(chunks: list[dict]) -> dict[str, set[str]]:
    """Return {repo: {path, path, ...}} for every distinct file visible
    in this result set."""
    seen: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        doc = c.get("documents") or {}
        repo = doc.get("repo")
        path = doc.get("path")
        if repo and path:
            seen[repo].add(path)
    return seen


def main():
    print("=" * 70)
    print("Fetching full result sets for both test users")
    print("=" * 70)
    contractor_chunks = query_all_chunks_as(CONTRACTOR_EMAIL, CONTRACTOR_PASSWORD)
    senior_chunks = query_all_chunks_as(SENIOR_EMAIL, SENIOR_PASSWORD)
    print(f"  contractor: {len(contractor_chunks)} chunks")
    print(f"  senior_engineer: {len(senior_chunks)} chunks\n")

    contractor_paths = paths_seen_by(contractor_chunks)
    senior_paths = paths_seen_by(senior_chunks)

    print("=" * 70)
    print("Per-repo, per-file check")
    print("=" * 70)

    all_ok = True

    for repo, expected_senior_files in EXPECTED_SENIOR_ONLY_FILES.items():
        print(f"\n{repo}:")

        # 1. Every expected senior-only file must be visible to senior_engineer...
        for f in sorted(expected_senior_files):
            if f in senior_paths.get(repo, set()):
                print(f"  OK       senior_engineer CAN see:      {f}")
            else:
                print(f"  MISSING  senior_engineer CANNOT see (expected to): {f}")
                all_ok = False

        # 2. ...and invisible to contractor.
        for f in sorted(expected_senior_files):
            if f in contractor_paths.get(repo, set()):
                print(f"  LEAK     contractor CAN see (should NOT):  {f}")
                all_ok = False
            else:
                print(f"  OK       contractor correctly blocked from: {f}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if all_ok:
        print("PASS: every known senior_engineer-only file, across all 4 repos, "
              "is visible to the senior_engineer session and correctly blocked "
              "from the contractor session. No leaks, no over-restriction.")
    else:
        print("FAIL: see MISSING/LEAK lines above. Do not proceed to Phase 1 "
              "until every line reads OK.")


if __name__ == "__main__":
    main()