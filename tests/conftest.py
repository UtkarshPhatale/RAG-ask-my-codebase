import pytest
from supabase import create_client, Client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

CONTRACTOR_CREDS = ("contractor-test@example.com", "TestPass123!")
SENIOR_CREDS = ("senior-test@example.com", "TestPass456!")
NOROLE_CREDS = ("norole-test@example.com", "TestPass789!")



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


@pytest.fixture(scope="session")
def norole_client() -> Client:
    """A Supabase client authenticated as a real user who has NEVER been
    assigned a role in user_roles -- e.g. a fresh signup before role
    assignment happens. Used to confirm get_current_user fails closed
    (403) rather than defaulting to some implicit access level."""
    return _authed_client(*NOROLE_CREDS)


def _get_token(email: str, password: str) -> str:
    """Return the raw JWT access token for a real Supabase Auth user.

    Separate from _authed_client: the HTTP-level tests need the literal
    bearer string to put in an Authorization header, not a scoped client.
    """
    client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    result = client.auth.sign_in_with_password({"email": email, "password": password})
    return result.session.access_token


@pytest.fixture(scope="session")
def contractor_token() -> str:
    return _get_token(*CONTRACTOR_CREDS)


@pytest.fixture(scope="session")
def senior_token() -> str:
    return _get_token(*SENIOR_CREDS)