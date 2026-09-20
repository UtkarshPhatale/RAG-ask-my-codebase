create table documents (
  id uuid primary key default gen_random_uuid(),
  repo text not null,               -- e.g. 'interface-ai-computer-use-project'
  path text not null,               -- e.g. 'guardrails/policy.py'
  required_scope text not null references roles(name),
  created_at timestamptz default now()
);

create table chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  content text not null,
  embedding vector(384),            -- 384 dims for all-MiniLM-L6-v2
  required_scope text not null references roles(name),  -- denormalized from documents for fast RLS checks
  created_at timestamptz default now()
);

-- Index for fast similarity search
create index on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);