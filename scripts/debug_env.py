"""
Quick sanity check: confirms .env is actually being found and loaded,
and prints exactly what SUPABASE_URL resolves to (without printing the
key values in full, just enough to spot a typo/truncation).

Usage:
    python scripts/debug_env.py
"""
import os
from dotenv import load_dotenv

loaded = load_dotenv()
print(f".env file found and loaded: {loaded}")
print()

url = os.environ.get("SUPABASE_URL")
pub_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY")

print(f"SUPABASE_URL = {url!r}")
print(f"SUPABASE_PUBLISHABLE_KEY = {(pub_key[:12] + '...') if pub_key else None!r}")

if not url:
    print("\nPROBLEM: SUPABASE_URL is not set at all -- .env isn't being "
          "read, or the variable name doesn't match exactly.")
elif not url.startswith("https://") or ".supabase.co" not in url:
    print(f"\nPROBLEM: SUPABASE_URL doesn't look like a valid Supabase URL: {url!r}")
else:
    print("\nSUPABASE_URL looks well-formed.")