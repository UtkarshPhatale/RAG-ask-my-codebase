-- Similarity search function, callable via Supabase's .rpc().
--
-- SECURITY INVOKER (not the Supabase-doc-example default of SECURITY DEFINER)
-- is the critical choice here: it makes this function run with the CALLING
-- user's own privileges, so the existing RLS policy on `chunks`
-- (role_scoped_chunk_access) is enforced exactly as if the caller ran a
-- plain SELECT themselves. A SECURITY DEFINER version would run as the
-- function's owner and silently bypass RLS -- the exact bug this project
-- exists to prove doesn't happen.
create or replace function match_chunks(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  required_scope text,
  similarity float
)
language sql
security invoker
stable
as $$
  select
    chunks.id,
    chunks.document_id,
    chunks.content,
    chunks.required_scope,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  order by chunks.embedding <=> query_embedding
  limit match_count;
$$;