"""
Validate eval/questions.jsonl.

Checks (errors fail the run, warnings don't):
  - every line is valid JSON with exactly the schema's fields
  - ids are unique and well-formed; categories are valid
  - 'unanswerable' <=> empty expected list
  - every question has author_verified == true
  - every expected (repo, path) exists in the LIVE documents table
  - every key_phrase appears verbatim in its expected file (local corpus_repos/)

Reads documents as senior-test via Supabase Auth + the publishable key, so RLS
applies and senior_engineer sees every row. No secret key is needed.

Usage: python -m scripts.validate_eval
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
CORPUS = ROOT / "corpus_repos"

CATEGORIES = {"direct", "indirect", "unanswerable"}
FIELDS = {"id", "category", "question", "expected", "key_phrases", "author_verified"}

# Test fixtures (same users the RLS tests use).
SENIOR_EMAIL = "senior-test@example.com"
SENIOR_PASSWORD = "TestPass456!"


def load_documents() -> dict[tuple[str, str], str]:
    """Return {(repo, path): required_scope} for every document."""
    load_dotenv(ROOT / ".env")
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_PUBLISHABLE_KEY"])
    auth = client.auth.sign_in_with_password(
        {"email": SENIOR_EMAIL, "password": SENIOR_PASSWORD}
    )
    client.postgrest.auth(auth.session.access_token)

    docs, start, page = {}, 0, 1000  # paginate: Supabase caps responses at 1000 rows
    while True:
        rows = (
            client.table("documents")
            .select("repo,path,required_scope")
            .range(start, start + page - 1)
            .execute()
            .data
        )
        for r in rows:
            docs[(r["repo"], r["path"])] = r["required_scope"]
        if len(rows) < page:
            return docs
        start += page


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    questions = []

    for n, line in enumerate(QUESTIONS.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e})")

    docs = load_documents()
    seen_ids, seen_questions = set(), set()
    scope_hits, repo_hits, covered = Counter(), Counter(), set()

    for q in questions:
        qid = q.get("id", "<no id>")
        if set(q) != FIELDS:
            errors.append(f"{qid}: fields differ from schema (diff: {set(q) ^ FIELDS})")
            continue
        if not re.fullmatch(r"q\d{3}", qid):
            errors.append(f"{qid}: id must look like q001")
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: bad category {q['category']!r}")
        if not q["question"].strip():
            errors.append(f"{qid}: empty question")
        if q["question"] in seen_questions:
            errors.append(f"{qid}: duplicate question text")
        seen_questions.add(q["question"])
        if q["author_verified"] is not True:
            errors.append(f"{qid}: author_verified is not true")
        if (q["category"] == "unanswerable") != (len(q["expected"]) == 0):
            errors.append(f"{qid}: 'unanswerable' must have an empty expected list, and only it")
        if q["category"] != "unanswerable" and not q["key_phrases"]:
            warnings.append(f"{qid}: no key_phrases (phrase-hit can't be measured)")

        texts = []
        for e in q["expected"]:
            key = (e.get("repo"), e.get("path"))
            if key not in docs:
                errors.append(f"{qid}: {key} not found in live documents table")
                continue
            scope_hits[docs[key]] += 1
            repo_hits[key[0]] += 1
            covered.add(key)
            local = CORPUS / key[0] / key[1]
            if local.exists():
                texts.append(local.read_text(errors="ignore"))
        if texts:
            for p in q["key_phrases"]:
                if not any(p in t for t in texts):
                    errors.append(f"{qid}: key_phrase not found in expected files: {p!r}")

    senior_docs = {k for k, s in docs.items() if s == "senior_engineer"}
    uncovered = sorted(senior_docs - covered)

    print(f"\n{len(questions)} questions")
    print("by category:", dict(Counter(q.get("category") for q in questions)))
    print("expected-file scope:", dict(scope_hits))
    print("expected-file repo:", dict(repo_hits))
    print(f"senior-only files covered: {len(senior_docs) - len(uncovered)}/{len(senior_docs)}")
    for k in uncovered:
        warnings.append(f"senior-only file has no question: {k}")

    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("\nFAILED" if errors else "\nOK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())