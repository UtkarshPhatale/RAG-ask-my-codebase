# Internal Research Notes (DUMMY — RBAC test fixture)

> Added deliberately for the RAG-ask-my-codebase project. Fictional content,
> not a real research operations doc. senior_engineer scope only — see
> docs/access_design.md in RAG-ask-my-codebase.

## Compute allocation (fictional)

This project's SLURM jobs run under a shared departmental GPU allocation
(fictional account: `FAKE-alloc-cs-grad-2026-0091`). The allocation has a
fixed core-hour budget per semester; going over triggers automatic job
throttling for the whole lab group, not just this project. Contractors or
external collaborators should never be given the raw allocation credentials
or account ID directly — request compute time through the lab's shared
intake form instead.

## Known data-handling caveat (fictional)

The BraTS dataset used here is public and de-identified, so there's no PHI
concern with the patient IDs themselves. However, an earlier internal
experiment (not part of the published thesis results) briefly used a
small pilot set of real, non-public clinical scans obtained under a
signed institutional data-use agreement before switching entirely to
BraTS for the final work. Those pilot scans were deleted and never
committed to this repo, but the DUA terms restrict even *mentioning*
the collaborating institution's name in public materials. This note
exists only so future contributors don't accidentally reference it —
it's not otherwise discoverable in the codebase.
