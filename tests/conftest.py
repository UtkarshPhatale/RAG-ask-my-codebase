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


@pytest.fixture
def contractor_client() -> Client:
    """A Supabase client authenticated as the contractor test user.
    Used exactly like a real contractor's client would be in production --
    same publishable key, same JWT-based session, so RLS applies identically."""
    return _authed_client(*CONTRACTOR_CREDS)


@pytest.fixture
def senior_client() -> Client:
    """Same, but for the senior_engineer test user."""
    return _authed_client(*SENIOR_CREDS)