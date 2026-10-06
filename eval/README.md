# Retrieval Eval Set

Hand-labeled questions for measuring retrieval quality (Phase 2). This is
separate from the RLS adversarial tests: those prove *enforcement*, this
measures *retrieval quality* (see ADR 0004).

## Format: eval/questions.jsonl (one JSON object per line)

| Field | Meaning |
|---|---|
| `id` | Stable ID (`q001`...). Never reused or renumbered. |
| `category` | `direct` (uses the file's own vocabulary), `indirect` (paraphrase, no distinctive terms), or `unanswerable` (nothing in the corpus answers it). |
| `question` | The question exactly as it would be typed into /ask. |
| `expected` | List of `{repo, path}` files that contain the answer. Empty for `unanswerable`. |
| `key_phrases` | Optional. Exact strings copied from the expected file(s). A retrieved chunk containing one counts as a phrase-hit (stricter than a file-hit). |
| `author_verified` | `true` only after the author opened every expected file and confirmed the label. Unverified questions are rejected by the validator. |

## Design decisions

- **File-level labels, not chunk IDs.** Chunk IDs change on every re-ingest,
  so chunk-level labels would silently rot. Cost: a file-hit can be a
  wrong chunk of the right file, so `key_phrases` provides a stricter check
  and both numbers are reported.
- **Scope is NOT stored in the labels.** Whether a file is senior-only comes
  from `ingestion/access_map.json`, the existing single source of truth.
  Duplicating it here would let labels drift from the real access rules.
- **Scoring by role.** For a role that may see the expected file, we score
  retrieval quality. For a role that may not, we assert the file is absent
  (that is the RLS guarantee, expected to be 100%, and is not a quality score).
- **Labels are written before looking at any results**, and the set is frozen
  after Day 2 of Phase 2. Later fixes are limited to labeling mistakes and
  are documented. New questions go in a separate v2 file.
## Label corrections made during verification (before any retrieval was run)

Labels were first drafted from docs/access_design.md descriptions, then checked
against the actual files. Changes:
- q001: reworded. The only files explaining why unfiltered queries are safe
  (RLS) are senior-only, so no contractor-visible file answered the original.
- q005: expected narrowed to README.md and EXPERIMENT_LOG.md (the thesis
  summary only mentions intensity as an augmentation setting).
- q006: reworded to match what the (very short) verification report contains.
- q011: guardrails/policy.py added; its redact() also keeps secrets out of
  logs and artifacts.
- q015: reworded; the file gives no budget size or grantor, only what happens
  when the budget is exceeded.

## Set frozen (Phase 2, Day 2): 28 questions

Composition: 20 direct, 5 indirect, 3 unanswerable. All 10 senior-only files
have at least one question. Validated by `python -m scripts.validate_eval`,
which checks the schema, that every expected (repo, path) exists in the live
documents table, and that every key_phrase appears verbatim in its file.

From here on, existing questions change only to fix labeling mistakes, and each
fix is logged here. New questions go in a separate v2 file, never into this one.

## Label corrections and known ambiguities (recorded before any retrieval was run)

- q025: grep for loss-class definitions found two files, so both are expected:
  scripts/advanced/advanced_losses.py and scripts/train_segmentation.py. The
  key phrase comes only from advanced_losses.py, so a train_segmentation.py
  chunk can count as a file-hit but not a phrase-hit.
- q016: escalation/models.py also describes the resolve() hand-back in its
  docstring. Only resolve.py is labeled; a models.py hit is a near-miss.
- q018: internal/lab_notes.md also mentions the allocation credentials and may
  compete with cluster_secrets_template.py (both senior-only).
- attention_unet*.py has several near-duplicate variants, so no question
  targets it; a file-level label there would be ambiguous.
- q019-q021 (unanswerable): the only keyword hit in the repos was
  nextplay_kanban/package-lock.json, which is not an ingested file type.
