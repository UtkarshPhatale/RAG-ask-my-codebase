alter table chunks enable row level security;
alter table documents enable row level security;
alter table user_roles enable row level security;

-- The core enforcement policy: a chunk is visible only if the caller's
-- role can see that chunk's required_scope.
-- senior_engineer sees everything; contractor sees only contractor-scoped chunks.
create policy "role_scoped_chunk_access"
on chunks
for select
using (
  required_scope = 'contractor'
  or (
    required_scope = 'senior_engineer'
    and exists (
      select 1 from user_roles
      where user_roles.user_id = auth.uid()
      and user_roles.role = 'senior_engineer'
    )
  )
);

-- Same logic for documents (so metadata queries respect scope too)
create policy "role_scoped_document_access"
on documents
for select
using (
  required_scope = 'contractor'
  or (
    required_scope = 'senior_engineer'
    and exists (
      select 1 from user_roles
      where user_roles.user_id = auth.uid()
      and user_roles.role = 'senior_engineer'
    )
  )
);

-- Users can only see their own role mapping
create policy "own_role_only"
on user_roles
for select
using (auth.uid() = user_id);