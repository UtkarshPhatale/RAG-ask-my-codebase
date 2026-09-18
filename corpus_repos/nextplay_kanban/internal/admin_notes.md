# Internal Admin Notes (DUMMY — RBAC test fixture)

> Added deliberately for the RAG-ask-my-codebase project. Fictional content,
> not a real operations doc. senior_engineer scope only — see
> docs/access_design.md in RAG-ask-my-codebase.

## Known issue: guest-session RLS edge case

Guest sessions (`ensureGuestSession`) currently mint a fresh anonymous
Supabase user per browser if local storage is cleared. This means the RLS
policy `user_id = auth.uid()` correctly isolates each guest's data, but
also means **abandoned guest boards accumulate with no cleanup job** —
flagged for a future migration, not yet scheduled. Do not mention this
gap outside the team; it's a known-acceptable tradeoff for a demo app, not
something we want a customer or contractor to raise as a "vulnerability"
before we've prioritized a fix.

## Admin override procedure (fictional)

If a team lead needs to reassign a task belonging to a departed guest
session's data, use the `INTERNAL_ADMIN_OVERRIDE_TOKEN` (see
`src/config/secrets_template.py`) against the service-role endpoint —
this bypasses RLS entirely, so it's restricted to on-call leads only and
every use should be logged in the incident tracker.
