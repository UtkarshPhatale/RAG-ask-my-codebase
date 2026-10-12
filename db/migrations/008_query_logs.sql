-- 008: query_logs -- one row per /ask call, access enforced by RLS.
--
-- Same principle as `chunks`: the database decides who can read what, not app code.
--   * a user can write rows only about themselves, and the `role` column must match
--     their real role in user_roles (otherwise a contractor could claim to be senior)
--   * a user can read their own rows; a senior_engineer can read all rows
--   * nobody can update or delete rows (an audit trail should not be editable)
--   * anon has no access
-- Replaces the local logs/queries.jsonl, which does not survive redeploys on hosts
-- with ephemeral disks.

create table query_logs (
  id                uuid primary key default gen_random_uuid(),
  created_at        timestamptz not null default now(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  role              text not null references roles(name),
  question          text not null,
  k                 int  not null default 5,
  -- per retrieved chunk: {chunk_id, repo, path, scope, similarity}
  retrieved         jsonb not null default '[]'::jsonb,
  retrieved_count   int  not null default 0,
  senior_only_count int  not null default 0,
  retrieval_ms      int,
  generation_ms     int,
  total_ms          int
);

create index query_logs_created_at_idx on query_logs (created_at desc);
create index query_logs_user_id_idx on query_logs (user_id);

alter table query_logs enable row level security;

-- Least privilege on top of RLS: even without policies, these roles could not
-- update or delete.
revoke all on query_logs from anon, authenticated;
grant select, insert on query_logs to authenticated;

create policy query_logs_insert_own on query_logs
  for insert to authenticated
  with check (
    user_id = auth.uid()
    and role = (select ur.role from user_roles ur where ur.user_id = auth.uid())
  );

create policy query_logs_select on query_logs
  for select to authenticated
  using (
    user_id = auth.uid()
    or exists (
      select 1 from user_roles ur
      where ur.user_id = auth.uid() and ur.role = 'senior_engineer'
    )
  );
