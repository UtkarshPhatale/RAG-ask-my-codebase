# Eval v2: plan and pre-registered predictions

Written 2026-11-24, before any v2 question exists and before any retrieval experiment has run.

## Why a second set

The v1 set has 28 questions, so one question is worth 4-7 points per role and differences
under about two questions are noise. Retrieval experiments (path-prefixed chunks, hybrid
BM25 + vector, a reranker) need a larger set that was not used to find the weaknesses they
target. v1 stays frozen and untouched.

## Rules

1. New file `eval/questions_v2.jsonl`, same schema as v1, ids `v001`, `v002`, ...
2. Target size about 48 questions: roughly 30 direct, 12 indirect, 6 unanswerable, spread
   over all four repos (more from brain-tumor-segmentation, which has the most files), and
   at least 12 expected files that are senior-only, so the restricted side is measured too.
3. Questions are written by reading the source files, never from retrieval output, and each
   is `author_verified` by the author. Draft help is fine; verification is not delegated.
4. `key_phrases` must appear verbatim in an expected file (checked by `validate_eval`,
   which needs a `--questions` option added first).
5. The v2 baseline is measured with the current system (exact search, 500-character chunks,
   MiniLM, k=5) before any experiment. No question is edited after that measurement,
   except to fix a proven labelling error, logged in this file.
6. Nothing is tuned on v2. Experiments run once each, with the result recorded whether
   or not it helps.

## Keep-or-drop rule for a technique

Kept only if all of these hold:
- v2 hit@5 improves by at least 6 points for the role it targets (about three questions at
  n=48; one question is about 2 points), measured against the v2 baseline;
- v1 hit@5 does not fall by more than one question per role (0.07 contractor, 0.04 senior);
- the leak check reports 0 leaks, and the CI gate (hit@5 >= 0.85 on v1) still passes;
- the RLS guarantee is unchanged: any new retrieval path runs inside Postgres under the
  caller's role (ADR needed before code).

Otherwise it is recorded as a negative result.

## Predictions (written before running anything; my guesses, not measurements)

- v2 baseline hit@5 will be lower than v1's 0.93 / 0.92, because v1 was the set the first
  failures were found on. Guess: 0.75-0.88.
- Path-prefixed chunks: small gain on questions about a file's purpose, little change on
  questions about a specific code line. Guess: +0 to +6 points.
- Hybrid BM25 + vector: the larger gain, on exact-identifier questions (function names,
  flags, config keys). Guess: +4 to +10 points, with a risk of lowering indirect questions.
- A reranker on top of hybrid: gain mostly in precision and MRR, not hit@5.
- If a prediction is wrong in either direction, that is recorded here as the finding.

## Not established

- Whether 48 questions suffices to separate a 6-point gain from noise; it only exceeds the
  noise of single questions, not sampling variation between question sets.
- Whether hit@5 tracks answer quality: no answer-quality eval exists.
