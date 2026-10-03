"""
Manually run retrieve_chunks() for a given user and question, and inspect
the raw results — including similarity scores and required_scope per
chunk — outside of pytest and outside the HTTP layer.

Use this when something about retrieval looks wrong and you need to isolate
*where*: it bypasses JWT-header plumbing and the FastAPI layer entirely,
calling retrieve_chunks() directly the same way the adversarial test suite
does. If a result looks wrong here, the bug is in retrieval/match_chunks();
if it only looks wrong through the API, the bug is in app/main.py's wiring
instead (see tests/test_ask_endpoint.py and ADR 0003 for why that
distinction matters).

Usage:
    python -m scripts.debug_retrieval <email> <password> "<question>"

Example:
    python -m scripts.debug_retrieval contractor-test@example.com TestPass123! \
        "how does the escalation system work?"
"""

import sys

from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY
from app.retrieval import retrieve_chunks

if len(sys.argv) != 4:
    print('Usage: python -m scripts.debug_retrieval <email> <password> "<question>"')
    sys.exit(1)

email, password, question = sys.argv[1], sys.argv[2], sys.argv[3]

client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
result = client.auth.sign_in_with_password({"email": email, "password": password})
client.postgrest.auth(result.session.access_token)

chunks = retrieve_chunks(client, question)

print(f"\nRetrieved {len(chunks)} chunks:\n")
for c in chunks:
    print(f"[{c['required_scope']}] similarity={c['similarity']:.4f}")
    print(f"  {c['content'][:100]}...")
    print()