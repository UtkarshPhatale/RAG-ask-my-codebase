"""
Create the fake test users (and their roles) on a LOCAL Supabase stack.

Mirrors the credentials in tests/conftest.py. Used for local runs and CI, where
the database is a throwaway `supabase start` stack, never the hosted project.

Refuses to run unless SUPABASE_URL is a local address, so it cannot create
test users with known passwords in a real project by accident.

Idempotent: users that already exist are reused, role rows are upserted.

Usage (with the local stack running and env pointing at it):
    python -m scripts.setup_local_test_users
"""
import sys
from urllib.parse import urlparse

from supabase import create_client

from app.config import SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_URL

# Keep in sync with tests/conftest.py (None = deliberately no role).
USERS = [
    ("contractor-test@example.com", "TestPass123!", "contractor"),
    ("senior-test@example.com", "TestPass456!", "senior_engineer"),
    ("norole-test@example.com", "TestPass789!", None),
]


def _require_local() -> None:
    host = urlparse(SUPABASE_URL).hostname
    if host not in ("127.0.0.1", "localhost"):
        sys.exit(f"REFUSING to run: SUPABASE_URL host is {host!r}, not a local stack.")


def _get_user_id(email: str, password: str) -> str:
    client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    try:
        return client.auth.sign_in_with_password(
            {"email": email, "password": password}
        ).user.id
    except Exception:
        client.auth.sign_up({"email": email, "password": password})
        return client.auth.sign_in_with_password(
            {"email": email, "password": password}
        ).user.id


def main() -> int:
    _require_local()
    admin = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    for email, password, role in USERS:
        user_id = _get_user_id(email, password)
        if role is not None:
            admin.table("user_roles").upsert({"user_id": user_id, "role": role}).execute()
        print(f"ok  {email:32} role={role}")
    return 0


if __name__ == "__main__":
    sys.exit(main())