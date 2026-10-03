"""
Get a real Supabase Auth access token (JWT) for a test user, for manual
curl/Postman testing of the live API.

For automated tests, use the contractor_token / senior_token fixtures in
tests/conftest.py instead — this script is for a human pasting a token into
a curl command by hand, not for anything the test suite calls.

Usage:
    python -m scripts.get_test_token <email> <password>

Example:
    python -m scripts.get_test_token contractor-test@example.com TestPass123!
    # then:
    curl -H "Authorization: Bearer <token>" http://localhost:8000/me
"""

import sys

from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

if len(sys.argv) != 3:
    print("Usage: python -m scripts.get_test_token <email> <password>")
    sys.exit(1)

email, password = sys.argv[1], sys.argv[2]

client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
result = client.auth.sign_in_with_password({"email": email, "password": password})

print("\nAccess token:\n")
print(result.session.access_token)