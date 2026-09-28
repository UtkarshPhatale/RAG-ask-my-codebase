import pytest
from supabase import create_client, Client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

CONTRACTOR_CREDS = ("contractor-test@example.com", "TestPass123!")
SENIOR_CREDS = ("senior-test@example.com", "TestPass456!")


def _authed_client(email: str, password: str) -> Client:
    client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    result = client.auth.sign_in_with_password({"email": email, "password": password})
    client.postgrest.auth(result.session.access_token)
    return client


@pytest.fixture(scope="session")
def contractor_client() -> Client:
    """Session-scoped: logs in ONCE for the whole test run, not once per
    test function. A JWT is valid for the session's duration regardless of
    how many tests reuse it, and this is what cut runtime from ~4.5 minutes
    down to a few seconds -- most of the original time was repeated
    network round-trips to Supabase Auth, not the actual RLS checks."""
    return _authed_client(*CONTRACTOR_CREDS)


@pytest.fixture(scope="session")
def senior_client() -> Client:
    return _authed_client(*SENIOR_CREDS)