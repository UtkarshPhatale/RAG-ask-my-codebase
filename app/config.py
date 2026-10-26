import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_PUBLISHABLE_KEY = os.environ["SUPABASE_PUBLISHABLE_KEY"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

# Optional at import time: only a real Claude call needs it, and
# app/generation.py raises a clear error at that point if it is missing.
# Tests that stub generation (or never reach it) can run without a key.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")