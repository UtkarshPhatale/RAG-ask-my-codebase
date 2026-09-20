-- Enable pgvector (idempotent if already on)
create extension if not exists vector;

-- Roles: a table, not an enum, so adding a third role later is a data
-- change, not a schema migration.
create table roles (
  name text primary key
);

insert into roles (name) values ('contractor'), ('senior_engineer');