-- 009: lock down the `roles` lookup table.
--
-- Migration 001 created `roles` without enabling row-level security, and
-- Supabase's default grants gave anon and authenticated full privileges on it.
-- Anyone holding the publishable key (it ships in the frontend) could therefore
-- insert rows through the REST API. Found by checking pg_tables.rowsecurity and
-- information_schema.role_table_grants during the CI spike, confirmed with an
-- anonymous insert on the local stack (HTTP 201).
--
-- Impact was a least-privilege gap, not a data leak: the RLS policies compare
-- against the literal names 'contractor' and 'senior_engineer', and nothing in
-- the app reads this table through the API (grep of app/, tests/, scripts/,
-- frontend/src found no reference).
--
-- The fix is deny-by-default: RLS on with no policies, and no table privileges
-- for the API roles. Foreign-key checks (user_roles, documents, chunks,
-- query_logs) run as the table owner, and ingestion uses the secret key, so
-- neither is affected.
--
-- Rollback:
--   alter table roles disable row level security;
--   grant all on roles to anon, authenticated;

alter table roles enable row level security;
revoke all on roles from anon, authenticated;