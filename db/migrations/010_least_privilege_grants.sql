-- 010: least-privilege grants on chunks, documents and user_roles.
--
-- Supabase's default grants give anon and authenticated full privileges
-- (including TRUNCATE, which RLS never checks) on every public table. RLS is on
-- for these three tables with select-only policies, so API writes were already
-- denied; this removes the unused privileges as defense in depth.
-- Found while closing the `roles` gap (009), via information_schema.role_table_grants.
--
-- After this:
--   anon           no access to these tables
--   authenticated  SELECT only (match_chunks() is SECURITY INVOKER and the RLS
--                  policies read user_roles as the caller, so SELECT is required)
-- Ingestion and the test-user setup use the secret key (service_role), which has
-- its own privileges and is unaffected.
--
-- Rollback:
--   grant all on chunks, documents, user_roles to anon, authenticated;

revoke all on chunks, documents, user_roles from anon, authenticated;
grant select on chunks, documents, user_roles to authenticated;