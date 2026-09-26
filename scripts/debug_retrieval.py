import sys
from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY
from app.retrieval import retrieve_chunks

if len(sys.argv) != 3:
    print("Usage: python -m scripts.debug_retrieval <email> <password>")
    sys.exit(1)

email, password = sys.argv[1], sys.argv[2]

client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
result = client.auth.sign_in_with_password({"email": email, "password": password})
client.postgrest.auth(result.session.access_token)

# Exact wording of the known senior-only chunk, to remove embedding-similarity
# ambiguity from the test entirely.
question = "logs it loudly -- replay mode NEVER auto-confirms -- an unconfirmed irreversible action is blocked"

chunks = retrieve_chunks(client, question)

print(f"\nRetrieved {len(chunks)} chunks:\n")
for c in chunks:
    print(f"[{c['required_scope']}] similarity={c['similarity']:.4f}")
    print(f"  {c['content'][:100]}...")
    print()