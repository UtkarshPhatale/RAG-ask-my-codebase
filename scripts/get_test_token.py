import sys
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

if len(sys.argv) != 3:
    print("Usage: python scripts/get_test_token.py <email> <password>")
    sys.exit(1)

email, password = sys.argv[1], sys.argv[2]

client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
result = client.auth.sign_in_with_password({"email": email, "password": password})

print("\nAccess token:\n")
print(result.session.access_token)