# Access Design — Role & Boundary Decisions

This document is the ground truth for the RBAC layer. Every adversarial test
in the project asserts against the boundaries defined here — if a test and
this doc ever disagree, this doc wins and the test gets fixed.

## Roles

Two roles to start (more can be added later without changing the mechanism):

| Role | Description |
|---|---|
| `contractor` | External/temporary engineer. Can ask about how the system works at a functional level — architecture, target apps, general flow — but should never retrieve implementation details of security-relevant subsystems. |
| `senior_engineer` | Full-time, trusted engineer. Full retrieval access across the entire corpus. |

## Classification principle

A chunk is scoped to `senior_engineer` only if it meets at least one of:
1. It defines or explains a **security/safety enforcement mechanism** (how the system decides what's allowed, how it protects secrets, how it hands off control) — knowing the exact logic could let someone deliberately work around it.
2. It documents a **known weakness or internal reasoning** that isn't meant for external-facing consumption (e.g. internal design tradeoffs, deferred risks).

Everything else — general architecture, how to run the project, target-app behavior, test files, dashboards/reporting, artifact schema, discovery/replay mechanics that don't touch enforcement — is `contractor`-visible. The bar is deliberately narrow: over-restricting defeats the point of the tool (a contractor who can't ask basic "how does this work" questions gets no value), so only genuinely sensitive material is walled off.

## Repo 1: Interface-ai-computer-use-project

| Path | Scope | Why |
|---|---|---|
| `guardrails/policy.py` | `senior_engineer` | Defines the entire safety boundary: domain/route allowlist, irreversible-action confirmation rules, secret-redaction patterns. Knowing this exactly is how you'd deliberately evade it. |
| `escalation/models.py` | `senior_engineer` | Human-takeover control-transfer protocol (`InterventionRequest`, status machine, control-handoff invariant). Sensitive because it describes exactly how/when automation cedes or reclaims control — a boundary condition worth protecting like the guardrails. |
| `escalation/resolve.py` | `senior_engineer` | Directly operates on the escalation store (`resolve()` call path) — same sensitivity as `models.py`. |
| `login_session.py` | `senior_engineer` | Explains and implements the specific mechanism for keeping credentials out of traces/artifacts. The *reasoning* here is exactly the kind of thing that shouldn't be casually browsable. |
| `escalation/console.py` | `contractor` | Operator-facing console UI for reviewing/acting on interventions — no enforcement logic itself, just display/interaction. Revisit if a later read shows it embeds policy decisions. |
| `agent/discovery.py` | `contractor` | Discovery/locator-fallback logic — core mechanics, not a security boundary. |
| `artifact/schema.py`, `artifact/builder.py` | `contractor` | Capability artifact data model — structural, not sensitive. |
| `replay/engine.py`, `replay/self_heal.py` | `contractor` | Replay mechanics, outcome classification, self-healing — core engineering, not access-control. |
| `evidence/logger.py` | `contractor` | Logging mechanics (structure, not content policy). |
| `dashboard/*.py` | `contractor` | Reporting/aggregation — read-only views over already-redacted data. |
| `target_app/`, `target_app_v2/` (all files) | `contractor` | Fake legacy demo apps — no real sensitivity. |
| `tests/*.py` | `contractor` | Test suite — useful for understanding expected behavior. |
| `README.md`, `FINDINGS.md`, `REPORT.md` | `contractor` | Public-facing project documentation. |
| `run_agent.py`, `run_replay.py` | `contractor` | Entry-point scripts, no embedded secrets/policy. |

**Dummy sensitive files to add** (Day 2): none needed yet for this repo — it already has 4 genuinely sensitive files with real architectural weight. Dummy files will supplement the other 3 repos, which are less naturally security-relevant, to ensure the corpus has sensitive material spread across all of them (avoids the RBAC test suite being dominated by one repo's files).

## Repo 2: nextplay_kanban

Audited: clean React/TypeScript frontend, no committed RLS SQL (policies live
in the Supabase dashboard, not version-controlled) and no embedded
credentials (`env.example` only has placeholders). Every real file here is
`contractor`-visible — genuinely nothing sensitive in the actual app code.

| Path | Scope | Why |
|---|---|---|
| `src/lib/supabase.ts`, `src/hooks/useBoard.ts` | `contractor` | Client setup + data-access hook layer. Relies entirely on Postgres RLS for scoping (no manual `user_id` filtering in queries) — this *is* the resume-worthy pattern, and it's not sensitive; understanding it doesn't help anyone bypass anything, since the enforcement isn't in this code at all. |
| `src/App.tsx`, `src/components/*.tsx`, `src/types/index.ts` | `contractor` | Plain UI components and type defs. |
| `env.example`, `package.json`, config files | `contractor` | No real secrets — placeholders only. |

**Dummy sensitive files added** (this repo needed all of them — nothing natively sensitive existed):
| Path | Scope | Why |
|---|---|---|
| `src/config/secrets_template.py` | `senior_engineer` | Fake service-role key, fake Stripe secret, fake admin override token. Represents "the file that documents which real secrets exist and where," which is exactly the class of thing that should never be retrievable by a contractor. |
| `internal/admin_notes.md` | `senior_engineer` | Fictional known-issue note (RLS/guest-session cleanup gap) + admin override procedure. Represents internal reasoning not meant for external-facing consumption. |
| `src/lib/admin/billingOverride.ts` | `senior_engineer` | Fictional admin-only module using a service-role client that bypasses RLS entirely. Represents the "payments/admin stub" pattern — a real enterprise app would have logic like this walled off. |

All three are clearly marked as dummy/fictional inside the files themselves (in case they're ever seen outside this project's context) and use obviously-fake values (`FAKE_` prefixes) — never real credentials.

## Repo 3: Brain Tumor Segmentation (Master's thesis)

Audited: a large research repo (~90 code/doc files once scoped down, see
ingestion note below). Public BraTS dataset IDs only — no real patient PHI
anywhere. No credentials or account info found in any script (including
SLURM submission scripts, which use only generic paths and job params).
Like kanban, this repo has nothing natively sensitive — dummy files were
added, but flavored for a research context (compute allocation, data-use
agreement) rather than a product context (secrets, billing), so the
boundary still feels authentic to what this repo actually is.

**Ingestion scoping decision:** this repo has ~350 files total, but the vast
majority are generated experiment artifacts — per-run PNGs, CSVs, and JSONs
under `results/`, `backups/`, `all_experiments/`, `individual_analysis/`,
`thesis_figures/`, plus bulk data references (`data/patient_list.txt`,
`data/splits/`). These add retrieval noise without adding "how does this
codebase work" value (a vector search for "how does the loss function work"
shouldn't compete against 50 near-duplicate `test_summary.json` files).
**Only `scripts/` (all code), top-level docs (`README.md`,
`COMPLETE_THESIS_SUMMARY.txt`, `EXPERIMENT_LOG.md`, `PROGRESS_TRACKER.md`),
`data/data_verification_report.txt`, and `graphs/*.py`** (the plotting
scripts, not their `.png` outputs) are ingested. This is a deliberate
signal-to-noise tradeoff, documented here so it reads as an intentional
engineering decision, not an oversight, if ever questioned.

| Path | Scope | Why |
|---|---|---|
| `scripts/**/*.py`, `scripts/**/*.sh` | `contractor` | Model architecture, training, evaluation, loss functions, postprocessing — the actual engineering substance. No security/safety logic of any kind; this is research code over public data. |
| `README.md`, `COMPLETE_THESIS_SUMMARY.txt`, `EXPERIMENT_LOG.md`, `PROGRESS_TRACKER.md` | `contractor` | Narrative documentation — genuinely useful for "why did you make this choice" questions. |
| `data/data_verification_report.txt` | `contractor` | Data-quality methodology, not bulk data itself. |
| `graphs/*.py` | `contractor` | Plotting/analysis scripts. |

**Dummy sensitive files added:**
| Path | Scope | Why |
|---|---|---|
| `internal/lab_notes.md` | `senior_engineer` | Fictional compute-allocation details + a fictional note about a deleted pilot dataset under a data-use agreement that restricts naming the collaborating institution. Represents the research-context equivalent of "internal reasoning not meant for external consumption." |
| `scripts/advanced/cluster_secrets_template.py` | `senior_engineer` | Fake SLURM account ID, fake SSH passphrase, fake data-portal token. Represents "the file documenting real credentials," same pattern as kanban's `secrets_template.py` but flavored for an academic-compute context. |

## Repo 4: RAG-chatbot

Audited: small, clean repo (5 data files, 2 source files, 1 test file). Worth
noting for your own awareness: `pipeline.py`'s comments (`PRE-BUILT (do not
modify)`, `YOUR WORK GOES HERE`, numbered `TODO`s) strongly suggest this was
built from a guided tutorial/exercise scaffold rather than designed fully
from scratch — `ask_question` (retrieval → prompt → generation) is the part
that's genuinely yours. Worth being ready to speak precisely to that
distinction if it comes up, since "I implemented X against a provided
scaffold" and "I designed X from nothing" are different claims.

The `data/` files are two coherent fictional demo corpora — a marketing
agency's public client-facing materials (`faq.txt`, `pricing.txt`,
`services.txt`) and "Acme Corp"'s product FAQ (`product_faq.txt`) — plus
`company_handbook.txt`, a fictional *internal* HR/policy document (remote
work, PTO, expense reimbursement, on-call pay, L&D budget). All of it is
clearly fictional/demo content, not real business data, but the handbook
file is naturally the right shape for a `senior_engineer`-only boundary
without needing an invented dummy file — it's already "internal HR content"
by design, unlike the other four files which are public-facing.

| Path | Scope | Why |
|---|---|---|
| `data/faq.txt`, `data/pricing.txt`, `data/services.txt` | `contractor` | Public-facing client materials for the marketing-agency persona — no reason to restrict. |
| `data/product_faq.txt` | `contractor` | Public product FAQ for the Acme Corp persona — same reasoning. |
| `data/company_handbook.txt` | `senior_engineer` | Internal HR/policy content (PTO accrual, on-call stipend amounts, expense limits). This is the one file in this repo's corpus that's naturally internal-only by its own content, not an invented boundary — a contractor asking the "customer-facing" chatbot shouldn't be able to retrieve internal comp/policy details. |
| `src/knowledge_base.py`, `src/pipeline.py` | `contractor` | Ingestion + retrieval/generation pipeline code — no security logic, nothing sensitive. |
| `tests/test_pipeline.py` | `contractor` | Test suite. |
| `README.md`, `requirements.txt` | `contractor` | Standard project docs/deps. |

**Dummy sensitive files:** none needed — `company_handbook.txt` already
provides a natural, un-invented senior-only boundary for this repo.

---

## Corpus summary (all 4 repos classified)

| Repo | Natural sensitive content? | Dummy files added |
|---|---|---|
| interface-ai-computer-use-project | Yes — 4 files (guardrails, escalation, login_session) | None needed |
| nextplay_kanban | No — fully clean frontend | 3 (secrets_template.py, admin_notes.md, billingOverride.ts) |
| brain-tumor-segmentation | No — public research data, no credentials | 2 (lab_notes.md, cluster_secrets_template.py) |
| RAG-chatbot | Yes — company_handbook.txt is naturally internal | None needed |

Design phase (Phase 0, Days 1-2 of the roadmap) is complete. Next: Supabase
schema + pgvector + RLS policy design (Day 3).

## Open question to revisit at Phase 4

If a contractor asks an *indirect* question that would only be answerable by combining a `contractor`-visible chunk with a `senior_engineer`-only chunk (e.g. "why does replay never auto-confirm irreversible actions?" — the fact is in `policy.py`, senior-only, but referenced indirectly in `replay/engine.py` comments), the retrieval layer should still only return contractor-visible chunks; the generation step may end up giving a vague or partial answer. This is expected and correct — documented explicitly in the README as the retrieval-guarantee vs. answer-completeness tradeoff.

## Phase 0 verification — RLS enforcement proven (Day 4)

Before any application code existed, the core RBAC claim of this project —
"a chunk's `required_scope` is enforced by Postgres itself, not by
application logic" — was verified directly against the database.

**Method:** two real Supabase Auth users were created (`contractor-test@example.com`
mapped to `contractor` in `user_roles`, `senior-test@example.com` mapped to
`senior_engineer`), plus two seeded chunks — one `contractor`-scoped
(`README.md`), one `senior_engineer`-scoped (`guardrails/policy.py`). A
verification script (`scripts/verify_rls.py`) logged in as each user via
Supabase Auth to obtain a real session JWT, then issued the identical query
(`select * from chunks`) using each session's own credentials — deliberately
using the publishable key, not the service_role key, so RLS policies were
actually in effect (the SQL Editor's default connection bypasses RLS
entirely and cannot be used for this test).

**Result:**
- Contractor session: 1 chunk returned (`README.md` only)
- Senior session: 2 chunks returned (`README.md` + `guardrails/policy.py`)
- The senior-only chunk never appeared in the contractor's result set

This confirms the retrieval-layer guarantee this project is built around:
restricted content is filtered by the database before it ever reaches
application code, not hidden after the fact by a prompt or a UI check. This
is the same class of proof the Phase 1 adversarial test suite will extend
to the full corpus, with embeddings and real similarity search in place of
the two hand-seeded rows used here.