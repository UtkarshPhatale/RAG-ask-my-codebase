-- Day 4 seed data: two test documents/chunks used to prove RLS enforcement
-- at the database layer, before any application code exists.
--
-- Corresponding test users (created via Supabase Auth dashboard, not SQL,
-- since Auth users aren't created through plain inserts):
--   contractor-test@example.com  -> role: contractor
--   senior-test@example.com      -> role: senior_engineer
--
-- Role assignment (run after creating the two users above, with their
-- real UUIDs from Authentication -> Users):
--
--   insert into user_roles (user_id, role) values
--     ('<contractor-user-uuid>', 'contractor'),
--     ('<senior-user-uuid>', 'senior_engineer');

-- A contractor-visible document + chunk
with doc as (
  insert into documents (repo, path, required_scope)
  values ('interface-ai-computer-use-project', 'README.md', 'contractor')
  returning id
)
insert into chunks (document_id, content, required_scope)
select id, 'This project automates browser interactions using Playwright.', 'contractor'
from doc;

-- A senior-only document + chunk
with doc as (
  insert into documents (repo, path, required_scope)
  values ('interface-ai-computer-use-project', 'guardrails/policy.py', 'senior_engineer')
  returning id
)
insert into chunks (document_id, content, required_scope)
select id, 'Irreversible actions require explicit confirm=True; replay mode never auto-confirms.', 'senior_engineer'
from doc;