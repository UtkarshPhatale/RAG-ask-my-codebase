"""
k-sweep and role comparison for the retrieval eval (Phase 2).

Retrieves each question ONCE at the largest k and scores slices of that result at
k = 3, 5, 10, 20. This is valid only because search is exact (ADR 0005): with an
approximate index, the top-5 of a query is not guaranteed to be a prefix of its
top-20.

Reports:
  1. metrics per role per k
  2. the same metrics restricted to questions BOTH roles are scored on, so the two
     roles can be compared fairly, plus how often senior-only chunks appear in the
     senior's top 5 on those questions
  3. every (question, role) not fully recovered at k=5, with the rank at which the
     expected file and a key phrase first appear within the top 20
  4. a leak check over the full top 20 for every retrieval (exit code 1 on a leak)

Usage:
  python -m scripts.eval_sweep
  python -m scripts.eval_sweep --out eval/results/sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.retrieval import retrieve_chunks
from ingestion.chunking import load_access_map, resolve_scope
from scripts.eval_retrieval import (
    ACCESS_MAP, QUESTIONS, ROLE_CREDS, authed_client, can_see, load_doc_map,
    score, summarize,
)

KS = (3, 5, 10, 20)


def first_ranks(q: dict, role: str, chunks: list[dict], doc_map: dict, access_map: dict):
    """Rank (1-based) of the first chunk from a visible expected file, and of the
    first chunk containing a key phrase. None if absent from the retrieved list."""
    expected = {(e["repo"], e["path"]) for e in q["expected"]}
    visible = {f for f in expected if can_see(role, resolve_scope(f[0], f[1], access_map))}
    file_rank = phrase_rank = None
    for i, c in enumerate(chunks, start=1):
        if file_rank is None and doc_map[c["document_id"]] in visible:
            file_rank = i
        if phrase_rank is None and any(p in c["content"] for p in q["key_phrases"]):
            phrase_rank = i
    return file_rank, phrase_rank


def print_sweep(title: str, rows_by_k: dict, roles, id_filter=None) -> None:
    print(f"\n{title}")
    print(f"{'role':17}{'k':>3}{'n':>4}{'hit':>7}{'phrase':>8}{'P':>7}{'R':>7}{'MRR':>7}")
    for role in roles:
        for k in KS:
            sel = [r for r in rows_by_k[k]
                   if r["role"] == role and (id_filter is None or r["id"] in id_filter)]
            s = summarize(sel)
            if s["n"]:
                print(f"{role:17}{k:>3}{s['n']:>4}{s['hit']:>7.2f}{s['phrase']:>8.2f}"
                      f"{s['precision']:>7.2f}{s['recall']:>7.2f}{s['mrr']:>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="write raw results as JSON")
    args = ap.parse_args()
    kmax = max(KS)

    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    assert all(q["author_verified"] for q in questions), "unverified question in the set"
    access_map = load_access_map(ACCESS_MAP)
    clients = {role: authed_client(*creds) for role, creds in ROLE_CREDS.items()}
    doc_map = load_doc_map(clients["senior_engineer"])

    rows_by_k = {k: [] for k in KS}
    ranks: dict[tuple[str, str], tuple] = {}
    senior_only_in_top5: dict[str, bool] = {}

    for q in questions:
        for role, client in clients.items():
            chunks = retrieve_chunks(client, q["question"], match_count=kmax)
            assert len(chunks) == kmax, (
                f"{q['id']} {role}: asked for {kmax} chunks, got {len(chunks)}. "
                "Exact search should always return the full count.")
            for k in KS:
                rows_by_k[k].append(score(q, role, chunks[:k], doc_map, access_map))
            if rows_by_k[5][-1]["kind"] == "scored":
                ranks[(q["id"], role)] = first_ranks(q, role, chunks, doc_map, access_map)
            if role == "senior_engineer":
                senior_only_in_top5[q["id"]] = any(
                    resolve_scope(*doc_map[c["document_id"]], access_map) == "senior_engineer"
                    for c in chunks[:5])

    roles = list(clients)
    print(f"{len(questions)} questions x {len(roles)} roles, retrieved once at k={kmax}")
    print_sweep("K-SWEEP (all scored questions per role)", rows_by_k, roles)

    scored = {role: {r["id"] for r in rows_by_k[5] if r["role"] == role and r["kind"] == "scored"}
              for role in roles}
    shared = scored["contractor"] & scored["senior_engineer"]
    print_sweep(f"SAME {len(shared)} QUESTIONS FOR BOTH ROLES (fair comparison)",
                rows_by_k, roles, id_filter=shared)
    n_cross = sum(senior_only_in_top5[i] for i in shared)
    print(f"\nOn those {len(shared)} questions, senior-only chunks appear in the senior's "
          f"top 5 for {n_cross} of them.")

    by_role5 = {(r["id"], r["role"]): r for r in rows_by_k[5]}
    diffs = [i for i in sorted(shared)
             if (by_role5[(i, "contractor")]["hit"], by_role5[(i, "contractor")]["phrase_hit"])
             != (by_role5[(i, "senior_engineer")]["hit"], by_role5[(i, "senior_engineer")]["phrase_hit"])]
    print(f"Shared questions where the two roles differ at k=5 (hit or phrase): {diffs or 'none'}")

    print("\nNOT FULLY RECOVERED AT k=5  (file rank / phrase rank; '-' = beyond top 20)")
    for (qid, role), (fr, pr) in sorted(ranks.items()):
        if fr is None or fr > 5 or pr is None or pr > 5:
            print(f"  {qid} {role:16} file {fr or '-':>3}   phrase {pr or '-':>3}")

    leaks = [r for r in rows_by_k[kmax] if r["leaks"]]
    n_retrievals = len(rows_by_k[kmax])
    print(f"\nLEAK CHECK over the top {kmax} of all {n_retrievals} retrievals: {len(leaks)} leaking")
    for r in leaks:
        print("  LEAK", r["id"], r["role"], r["leaks"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"ks": list(KS),
             "rows": {str(k): rows_by_k[k] for k in KS},
             "ranks": {f"{q}|{r}": v for (q, r), v in ranks.items()}}, indent=2))
        print(f"\nwrote {args.out}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())