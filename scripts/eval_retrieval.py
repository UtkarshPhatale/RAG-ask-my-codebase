"""
Retrieval-quality eval (Phase 2). Measures whether retrieval is any good,
separately from whether access control is correct (see ADR 0004).

For every question in eval/questions.jsonl, retrieves as BOTH roles through the
real app code path (app.retrieval.retrieve_chunks on a user-scoped client, so
RLS applies) and reports:
  hit@k, phrase@k, precision@k, recall@k (file-level), MRR
plus a leak check: no chunk from a senior-only file (per access_map.json, NOT
per the DB's own labels) may ever reach the contractor.

Scoring rules:
  - A role is scored on a question only if it may see at least one expected file.
  - Questions whose expected files are all hidden from the role are not scored
    (their only job is the leak check).
  - 'unanswerable' questions are not scored; top-1 file and similarity are shown.

Usage:
  python -m scripts.eval_retrieval                 # k=5, same default as /ask
  python -m scripts.eval_retrieval --k 10
  python -m scripts.eval_retrieval --out eval/results/baseline_k5.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from supabase import Client, create_client

from app.config import SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL
from app.retrieval import retrieve_chunks
from ingestion.chunking import load_access_map, resolve_scope

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
ACCESS_MAP = ROOT / "ingestion" / "access_map.json"

ROLE_CREDS = {
    "contractor": ("contractor-test@example.com", "TestPass123!"),
    "senior_engineer": ("senior-test@example.com", "TestPass456!"),
}


def authed_client(email: str, password: str) -> Client:
    client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    result = client.auth.sign_in_with_password({"email": email, "password": password})
    client.postgrest.auth(result.session.access_token)
    return client


def load_doc_map(senior_client: Client) -> dict[str, tuple[str, str]]:
    """document_id -> (repo, path). Read as senior so RLS shows every row."""
    doc_map, start, page = {}, 0, 1000
    while True:
        rows = (
            senior_client.table("documents")
            .select("id,repo,path")
            .range(start, start + page - 1)
            .execute()
            .data
        )
        for r in rows:
            doc_map[r["id"]] = (r["repo"], r["path"])
        if len(rows) < page:
            return doc_map
        start += page


def can_see(role: str, scope: str) -> bool:
    return role == "senior_engineer" or scope == "contractor"


def score(q: dict, role: str, chunks: list[dict], doc_map: dict, access_map: dict) -> dict:
    files = [doc_map[c["document_id"]] for c in chunks]  # (repo, path) per rank
    row = {
        "id": q["id"],
        "role": role,
        "category": q["category"],
        "repo": q["expected"][0]["repo"] if q["expected"] else "(none)",
        "files": [f"{r}/{p}" for r, p in files],
        "sims": [round(c["similarity"], 4) for c in chunks],
        # Leak check uses access_map.json, independent of the DB's own labels.
        "leaks": [
            f"{r}/{p}"
            for r, p in files
            if not can_see(role, resolve_scope(r, p, access_map))
        ],
    }

    if q["category"] == "unanswerable":
        row["kind"] = "unanswerable"
        return row

    expected = {(e["repo"], e["path"]) for e in q["expected"]}
    visible = {f for f in expected if can_see(role, resolve_scope(f[0], f[1], access_map))}
    if not visible:
        row["kind"] = "restricted"
        return row

    rel = [f in visible for f in files]
    first = next((i for i, x in enumerate(rel) if x), None)
    row.update(
        kind="scored",
        expected_visible=sorted(f"{r}/{p}" for r, p in visible),
        hit=any(rel),
        precision=sum(rel) / len(rel),
        recall=len(visible & set(files)) / len(visible),
        mrr=0.0 if first is None else 1 / (first + 1),
    )
    if q["key_phrases"]:
        # Conservative: a phrase split across two chunks by the 500-char splitter
        # will not match, so a file-hit with no phrase-hit can be a boundary split.
        row["phrase_hit"] = any(p in c["content"] for c in chunks for p in q["key_phrases"])
    else:
        row["phrase_hit"] = None
    return row


def mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["kind"] == "scored"]
    phrases = [r["phrase_hit"] for r in scored if r["phrase_hit"] is not None]
    return {
        "n": len(scored),
        "hit": mean([r["hit"] for r in scored]),
        "phrase": mean(phrases),
        "precision": mean([r["precision"] for r in scored]),
        "recall": mean([r["recall"] for r in scored]),
        "mrr": mean([r["mrr"] for r in scored]),
    }


def print_table(title: str, groups: dict[str, list[dict]], k: int) -> None:
    print(f"\n{title}")
    print(f"{'group':36}{'n':>3}{f'hit@{k}':>8}{f'phrase@{k}':>10}{f'P@{k}':>7}{f'R@{k}':>7}{'MRR':>7}")
    for label, rows in groups.items():
        s = summarize(rows)
        if s["n"] == 0:
            continue
        print(
            f"{label:36}{s['n']:>3}{s['hit']:>8.2f}{s['phrase']:>10.2f}"
            f"{s['precision']:>7.2f}{s['recall']:>7.2f}{s['mrr']:>7.2f}"
        )


def group_by(rows: list[dict], key) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return dict(sorted(groups.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="chunks retrieved per question (default 5, as /ask)")
    ap.add_argument("--out", type=Path, default=None, help="write per-question results as JSON")
    args = ap.parse_args()
    k = args.k

    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    assert all(q["author_verified"] for q in questions), "unverified question in the set"
    access_map = load_access_map(ACCESS_MAP)

    clients = {role: authed_client(*creds) for role, creds in ROLE_CREDS.items()}
    doc_map = load_doc_map(clients["senior_engineer"])

    rows = []
    for q in questions:
        for role, client in clients.items():
            chunks = retrieve_chunks(client, q["question"], match_count=k)
            rows.append(score(q, role, chunks, doc_map, access_map))

    print(f"{len(questions)} questions x {len(clients)} roles, k={k}")

    # ---- metrics ----
    print_table("OVERALL (by role)", group_by(rows, lambda r: r["role"]), k)
    print_table("BY ROLE x CATEGORY", group_by(rows, lambda r: f"{r['role']} | {r['category']}"), k)
    print_table("BY ROLE x REPO", group_by(rows, lambda r: f"{r['role']} | {r['repo']}"), k)

    # ---- leak check ----
    leaks = [r for r in rows if r["leaks"]]
    restricted = [r for r in rows if r["kind"] == "restricted"]
    print(f"\nLEAK CHECK: {len(leaks)} leaking (question, role) pairs "
          f"(checked on all {len(rows)} retrievals; {len(restricted)} restricted rows not scored)")
    for r in leaks:
        print("  LEAK", r["id"], r["role"], r["leaks"])

    # ---- unanswerable ----
    print("\nUNANSWERABLE (top-1 result and similarity; not scored)")
    for r in rows:
        if r["kind"] == "unanswerable":
            print(f"  {r['id']} {r['role']:16} sim={r['sims'][0]:.3f}  {r['files'][0]}")
    for role in clients:
        scored = [r for r in rows if r["kind"] == "scored" and r["role"] == role]
        for label, sel in (("hits", [r for r in scored if r["hit"]]), ("misses", [r for r in scored if not r["hit"]])):
            if sel:
                print(f"  context: {role} median top-1 similarity on {label}: "
                      f"{statistics.median(r['sims'][0] for r in sel):.3f} (n={len(sel)})")

    # ---- misses and near-misses (raw material for failure analysis) ----
    print("\nMISSES (no expected file in top-k)")
    for r in rows:
        if r["kind"] == "scored" and not r["hit"]:
            print(f"  {r['id']} {r['role']:16} expected {r['expected_visible']}")
            for f, s in list(zip(r["files"], r["sims"]))[:3]:
                print(f"      got {s:.3f}  {f}")
    print("\nFILE-HIT BUT NO PHRASE-HIT")
    for r in rows:
        if r["kind"] == "scored" and r["hit"] and r["phrase_hit"] is False:
            print(f"  {r['id']} {r['role']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"meta": {"k": k, "n_questions": len(questions),
                      "run_at": datetime.now(timezone.utc).isoformat()},
             "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")

    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())