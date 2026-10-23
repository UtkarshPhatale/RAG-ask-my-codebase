"""
The `roles` lookup table must not be writable through the public API.

Migration 001 created `roles` and never enabled RLS, and Supabase's default
grants gave `anon` and `authenticated` full privileges, so anyone holding the
publishable key (which ships in the frontend) could insert rows. Migration 009
closes that. These tests probe the real API as each caller.

Each probe inserts a uniquely named role and ALWAYS deletes it afterwards with
the secret key, so a failing run does not leave junk rows behind.
"""
import uuid

from postgrest.exceptions import APIError
from supabase import create_client

from app.config import SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_URL

PERMISSION_DENIED = "42501"  # Postgres: insufficient_privilege / RLS violation


def _insert_probe(client, admin):
    """Return None if the insert succeeded, else the Postgres error code."""
    name = f"probe_{uuid.uuid4().hex[:8]}"
    try:
        client.table("roles").insert({"name": name}).execute()
        return None
    except APIError as e:
        return e.code
    finally:
        admin.table("roles").delete().eq("name", name).execute()


def test_anon_cannot_write_roles():
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    anon = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    result = _insert_probe(anon, admin)
    assert result == PERMISSION_DENIED, f"anon insert into roles was not rejected (got {result!r})"


def test_authenticated_cannot_write_roles(contractor_client):
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    result = _insert_probe(contractor_client, admin)
    assert result == PERMISSION_DENIED, f"contractor insert into roles was not rejected (got {result!r})"