# Retrieval eval baseline (Phase 2)

This measures retrieval quality only: not answer quality, and not access control.
Access control is proven separately by the adversarial suite and by the eval's
leak check (0 leaks across 56 retrievals).

## Setup

- 28 hand-labeled questions (20 direct, 5 indirect, 3 unanswerable), frozen 2026-10-05.
  Labels are file-level plus optional key phrases (see `eval/README.md`).
- Both roles, through `app.retrieval.retrieve_chunks` on user-scoped clients, so RLS applies.
- k = 5, the value `/ask` uses.
- `all-MiniLM-L6-v2` embeddings, 500-character chunks with 50 overlap, exact
  nearest-neighbor search (ADR 0005).
- Reproduce: `python -m scripts.eval_retrieval --out eval/results/after_exact_k5.json`
  Diagnose a question: `python -m scripts.eval_inspect q003 --role senior_engineer`

## Metrics

| Metric | Definition |
|---|---|
| hit@k | At least one of the top-k chunks comes from an expected file the role may see |
| phrase@k | At least one top-k chunk contains a key phrase (stricter; a lower bound, see below) |
| precision@k | Fraction of the k chunks that come from expected files |
| recall@k | Fraction of the visible expected files that appear in the top k |
| MRR | 1 / rank of the first chunk from an expected file, 0 if none |

A role is scored only on questions whose expected files it may see. Questions whose
expected files are all hidden from a role only feed the leak check.

## Results (k = 5)

Before the index fix (ADR 0005), hit@5 was 0.67 (contractor) and 0.44 (senior).
After:

| Role (questions scored) | hit@5 | phrase@5 | P@5 | R@5 | MRR |
|---|---|---|---|---|---|
| contractor (15) | 0.93 | 0.73 | 0.45 | 0.87 | 0.77 |
| senior_engineer (25) | 0.92 | 0.72 | 0.39 | 0.86 | 0.78 |

The two role rows are not directly comparable: the senior set includes the ten
senior-only questions that the contractor set excludes.

| Role | Category | n | hit@5 | phrase@5 | MRR |
|---|---|---|---|---|---|
| contractor | direct | 13 | 0.92 | 0.77 | 0.74 |
| contractor | indirect | 2 | 1.00 | 0.50 | 1.00 |
| senior_engineer | direct | 20 | 0.90 | 0.70 | 0.75 |
| senior_engineer | indirect | 5 | 1.00 | 0.80 | 0.90 |

| Repo | contractor n / hit / phrase | senior n / hit / phrase |
|---|---|---|
| brain-tumor-segmentation | 4 / 1.00 / 0.75 | 6 / 1.00 / 0.83 |
| interface-ai-computer-use-project | 5 / 0.80 / 0.60 | 9 / 0.78 / 0.44 |
| nextplay_kanban | 2 / 1.00 / 1.00 | 5 / 1.00 / 1.00 |
| rag-chatbot | 4 / 1.00 / 0.75 | 5 / 1.00 / 0.80 |

Per-repo groups hold 2-9 questions, so treat them as descriptive only.

Indirect questions scored no worse than direct ones. With only 5 indirect questions
that is not evidence that paraphrase handling is solved.

## Failure analysis

Diagnosed with `scripts/eval_inspect.py`. Rank is the position of the answer-bearing
chunk among all chunks (retrieved at depth 100).

| Question | Role | Failure | Answer-bearing rank | Cause |
|---|---|---|---|---|
| q003 | both | no expected-file chunk in top 5 (file-level miss) | 10 (file), 17 (phrase) | REPORT.md and dashboard/report.py share the vocabulary (replay, success, failure) and outrank engine.py |
| q016 | senior | file-level miss | 11 | models.py describes the same hand-back; labeled ambiguity recorded before the run |
| q005 | both | phrase miss | 48-50 | top 5 are generic overview chunks; the answer is deep in long files |
| q008 | both | phrase miss | 18 | top 4 are all pipeline.py incl. prompt-template chunks; metric is too strict here |
| q010 | senior | phrase miss | 22 | docstring split across 3 chunks; the confirmation rule is in the lowest-ranked one |
| q012 | senior | phrase miss | 6 | one position outside k = 5 |
| q023 | both | phrase miss | 8 | three positions outside k = 5 |

### Findings

1. **No failure was a chunk-boundary artifact.** In all seven cases the phrase-bearing
   chunk exists and was simply outranked.
2. **Module docstrings are fragmented by 500-character chunking.** The title chunk
   ranks high; the specific sentence ranks lower and has no file or module context
   in its text. For q010 the allowlist half of the answer reaches the top 5 but the
   "replay never auto-confirms" half does not, so generation would give a partial answer.
3. **Documentation echoes source code.** In q003, q010, q012, q016 and q023, REPORT.md
   chunks sit in the top 5. The repo with the most such docs is also the weakest.
   Returning the doc may be a good answer that file-level labels do not credit.
4. **phrase@5 is a strict lower bound.** It asks whether the specific chunk I picked
   was retrieved, not whether the answer could be assembled. True retrieval quality
   lies between phrase@5 and hit@5.
5. **The real misses are confidently wrong.** q003's top-1 similarity (0.69) is above
   the median on hits (0.56-0.60), so a similarity threshold cannot flag it. The three
   unanswerable questions had top-1 similarity 0.21-0.37, so a threshold would only
   help for those; with n = 3 that is an observation, not a result.

## Limitations

- 28 questions: each is worth 4-7 points of a role's hit rate, so differences under
  about two questions are noise.
- One author wrote the corpus, the questions and the labels, and the set has few
  paraphrased questions (5).
- File-level labels cannot credit an alternative file that also answers the question
  (see `eval/README.md`, post-baseline observations).
- One key phrase per chunk-of-interest; no answer-quality evaluation.

## Next (each judged against this baseline)

1. k-sweep (k = 3, 5, 10, 20). Prediction: k = 10 recovers q012 and q023 and none of
   the others.
2. Candidate fixes for fragmentation: a path/context header on each chunk before
   embedding, or larger chunks. Both require re-ingestion.
3. Hybrid (BM25 + vector) search and reranking, kept only if they improve this baseline.
   Detecting a real gain probably needs a larger v2 question set.

## k-sweep predictions (written 2026-10-07, before running the sweep)

From the answer-bearing chunk ranks found by `eval_inspect` (senior role):
hit misses at k=5 are q003 (first expected-file chunk at rank 10) and q016 (rank 11);
phrase misses are q012 (rank 6), q023 (8), q016 (11), q003 (17), q008 (18), q010 (22)
and q005 (48).

| k | senior hit | senior phrase | contractor hit | contractor phrase |
|---|---|---|---|---|
| 5 (measured) | 0.92 | 0.72 | 0.93 | 0.73 |
| 10 | 0.96 | 0.80 | 1.00 | 0.80 |
| 20 | 1.00 | 0.92 | 1.00 | 0.93 |

Still unrecovered at k=20: phrase for q005 and q010. Contractor ranks can shift by a
few positions, since contractor retrieval excludes senior-only chunks.

## k-sweep results (2026-10-07)

Run: `python -m scripts.eval_sweep --out eval/results/sweep_exact.json`. Each question
is retrieved once at k=20 and scored on prefixes (valid because search is exact, ADR 0005).

| Role (questions) | k | hit | phrase | P | R | MRR |
|---|---|---|---|---|---|---|
| senior_engineer (25) | 3 | 0.88 | 0.60 | 0.48 | 0.82 | 0.77 |
| | 5 | 0.92 | 0.72 | 0.39 | 0.86 | 0.78 |
| | 10 | 0.96 | 0.80 | 0.28 | 0.92 | 0.79 |
| | 20 | 1.00 | 0.92 | 0.19 | 1.00 | 0.79 |
| contractor (15) | 3 | 0.87 | 0.67 | 0.53 | 0.80 | 0.76 |
| | 5 | 0.93 | 0.73 | 0.45 | 0.87 | 0.77 |
| | 10 | 1.00 | 0.80 | 0.33 | 0.93 | 0.78 |
| | 20 | 1.00 | 0.93 | 0.25 | 1.00 | 0.78 |

### Predictions vs. measured

All eight predicted numbers (hit and phrase at k=10 and k=20, both roles) matched.
This is a consistency check, not independent confirmation: the predictions were
arithmetic on chunk ranks that `eval_inspect` had already measured on the same data. It
confirms that the sweep's slicing and scoring agree with the per-question diagnostics.

### Role comparison on the 15 questions both roles are scored on

Hit and phrase are identical at every k. MRR at k=5 is 0.74 (senior) vs 0.77
(contractor). Senior-only chunks appear in the senior's top 5 for 2 of the 15 questions
without changing hit or phrase. On this set, the ten senior-only files do not crowd out
relevant content for senior users; the MRR gap is under one question's worth and is
noise. Example of the mechanism: q003's answer-bearing chunk ranks 16 for the
contractor and 17 for the senior.

### Findings

1. The top 20 contains the key phrase for 23 of 25 senior questions and 14 of 15
   contractor questions. Never recovered at k=20: q005 (both roles) and q010 (senior).
2. Headroom: phrase@5 is 0.72 and phrase@20 is 0.92, so reordering the top 20 (e.g.
   reranking) can recover at most 0.20 (5 of 25 questions). q005 and q010 need better
   candidate generation (e.g. path headers, hybrid search), not reranking.
3. Raising k from 5 to 10 adds 1 hit-question and 2 phrase-questions for senior, 1 and 1
   for contractor, while precision falls from 0.39 to 0.28 and the context sent to the
   generator doubles. By the noise rule above (under about two questions), weak evidence.
4. File-level MRR barely moves (0.77-0.79) because the expected file is usually found at
   rank 1-3. It says nothing about the answer chunk; a phrase-level MRR would be more
   informative in a v2 eval.

### Not measured

Answer quality. Whether more retrieved context helps or hurts the generated answer is
unknown, so the sweep alone cannot justify changing the `/ask` default of k=5.


## Why precision@5 is low (ceiling analysis, 2026-10-08)

Precision@5 counts the fraction of the 5 retrieved chunks that come from a labeled
file. It is capped by file size (a 3-chunk file can score at most 3/5), and it
scores any other file's chunk as wrong even when it is useful context (e.g.
REPORT.md restating the code). The ceiling below assumes every chunk of a labeled
file counts as relevant, which is how the metric itself works, so it is generous.

| Role (questions) | mean P@5 | mean ceiling | P@5 / ceiling |
|---|---|---|---|
| contractor (15) | 0.45 | 0.93 | 0.49 |
| senior_engineer (25) | 0.39 | 0.88 | 0.45 |

Precision is not a metric to optimize for this pipeline: the generator needs the
answer present in its context, not a clean context. hit@5 and phrase@5 are the more
informative numbers. Whether the extra, unlabeled chunks hurt generated answers is
untested (no answer-quality eval).

## Decision: `/ask` keeps k = 5 (2026-10-08)

Raising k to 10 would add 1-2 recovered questions per role, which is at the noise
floor of a 28-question set, and would double the context sent to the generator
with no answer-quality eval to show it helps. A Phase 3 reranker experiment
(retrieve 20, rerank to 5) may make the question moot. Revisit with evidence.
