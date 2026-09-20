-- Maps a Supabase-authenticated user to exactly one role
create table user_roles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  role text not null references roles(name)
);