"""
Diagnose eval questions. For each one, show where the expected file's chunks rank
and whether any single chunk contains a key phrase at all.

Verdicts:
  OK        a phrase-bearing chunk is in the top --top (default 5)
  RANK      a phrase-bearing chunk exists but ranks below the top --top
            (a real retrieval-ranking weakness)
  SPLIT     no single chunk contains any key phrase, so the phrase spans a chunk
            boundary (a chunking/labeling artifact, not a ranking failure)
  MISS      no chunk of any expected file is within the top --k
  NOPHRASE  the question has no key_phrases (file-level check only)

Marks in the top-N listing: '*' = chunk is from an expected file,
'P' = chunk contains a key phrase.

Usage:
  python -m scripts.eval_inspect q003 q016 --role senior_engineer
  python -m scripts.eval_inspect q005 --role contractor --show 8
"""
from __future__ import annotations

import argparse
import json

from app.retrieval import retrieve_chunks
from ingestion.chunking import load_access_map, resolve_scope
from scripts.eval_retrieval import (
    ACCESS_MAP, QUESTIONS, ROLE_CREDS, authed_client, can_see, load_doc_map,
)


def one_line(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def inspect(q, role, k, top, show, clients, doc_map, path_to_id, access_map) -> None:
    print("=" * 78)
    print(f"{q['id']} [{role}] ({q['category']})  {q['question']}")

    expected = [(e["repo"], e["path"]) for e in q["expected"]]
    visible = [f for f in expected if can_see(role, resolve_scope(f[0], f[1], access_map))]
    if not visible:
        print("  (restricted for this role: expected files are not visible, nothing to diagnose)")
        return

    phrases = q["key_phrases"]
    chunks = retrieve_chunks(clients[role], q["question"], match_count=k)
    rank_of = {c["id"]: i + 1 for i, c in enumerate(chunks)}
    sim_of = {c["id"]: c["similarity"] for c in chunks}

    print(f"\n  top {show} retrieved  (* = expected file, P = has key phrase)")
    for i, c in enumerate(chunks[:show], start=1):
        repo, path = doc_map[c["document_id"]]
        star = "*" if (repo, path) in visible else " "
        has_p = "P" if any(p in c["content"] for p in phrases) else " "
        print(f"   {i:>2}{star}{has_p} {c['similarity']:.3f}  {repo}/{path}")
        print(f"           {one_line(c['content'])}")

    all_rows = []
    for f in visible:
        # Read the expected file's chunks as senior (RLS shows everything) so
        # we can see chunks that were NOT retrieved.
        rows = (
            clients["senior_engineer"].table("chunks").select("id,content")
            .eq("document_id", path_to_id[f]).execute().data
        )
        all_rows.extend(rows)
        print(f"\n  expected file {f[0]}/{f[1]}: {len(rows)} chunks")
        ranked = sorted(rows, key=lambda r: rank_of.get(r["id"], 10**9))
        for r in ranked[:12]:
            rk = rank_of.get(r["id"])
            has_p = "P" if any(p in r["content"] for p in phrases) else " "
            where = f"rank {rk:>3}  sim {sim_of[r['id']]:.3f}" if rk else f"rank >{k}"
            print(f"    {where}  {has_p}  {one_line(r['content'], 80)}")
        if len(ranked) > 12:
            print(f"    ... {len(ranked) - 12} more chunks")

    file_rank = min((rank_of[r["id"]] for r in all_rows if r["id"] in rank_of), default=None)
    phrase_rows = [r for r in all_rows if any(p in r["content"] for p in phrases)]
    phrase_ranks = [rank_of[r["id"]] for r in phrase_rows if r["id"] in rank_of]

    if file_rank is None:
        verdict = f"MISS: no chunk of the expected file(s) within top {k}"
    elif not phrases:
        verdict = f"NOPHRASE: best expected-file chunk at rank {file_rank}"
    elif not phrase_rows:
        verdict = (f"SPLIT: no single chunk contains any key phrase "
                   f"(best expected-file chunk at rank {file_rank})")
    elif phrase_ranks and min(phrase_ranks) <= top:
        verdict = f"OK: phrase-bearing chunk at rank {min(phrase_ranks)}"
    elif phrase_ranks:
        verdict = f"RANK: phrase-bearing chunk at rank {min(phrase_ranks)} (file's best chunk: rank {file_rank})"
    else:
        verdict = f"RANK: phrase-bearing chunk exists but is below rank {k} (file's best chunk: rank {file_rank})"
    print(f"\n  VERDICT: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", help="question ids, e.g. q003 q016")
    ap.add_argument("--role", choices=list(ROLE_CREDS), default="senior_engineer")
    ap.add_argument("--k", type=int, default=100, help="retrieval depth used to find ranks")
    ap.add_argument("--top", type=int, default=5, help="the k that counts as 'retrieved' for the verdict")
    ap.add_argument("--show", type=int, default=5, help="how many top chunks to print")
    args = ap.parse_args()

    questions = {}
    for line in QUESTIONS.read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            questions[q["id"]] = q
    unknown = [i for i in args.ids if i not in questions]
    if unknown:
        raise SystemExit(f"unknown question ids: {unknown}")

    access_map = load_access_map(ACCESS_MAP)
    clients = {"senior_engineer": authed_client(*ROLE_CREDS["senior_engineer"])}
    if args.role != "senior_engineer":
        clients[args.role] = authed_client(*ROLE_CREDS[args.role])
    doc_map = load_doc_map(clients["senior_engineer"])
    path_to_id = {v: k for k, v in doc_map.items()}

    for qid in args.ids:
        inspect(questions[qid], args.role, args.k, args.top, args.show,
                clients, doc_map, path_to_id, access_map)


if __name__ == "__main__":
    main()