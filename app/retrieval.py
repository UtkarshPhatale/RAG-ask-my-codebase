from sentence_transformers import SentenceTransformer
from supabase import Client

# Loaded once at import time — same model used at ingestion. Loading per-request
# would reload ~90MB of weights on every /ask call.
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def retrieve_chunks(user_client: Client, question: str, match_count: int = 5) -> list[dict]:
    """
    Retrieves the top-k most similar chunks for `question`, as the
    AUTHENTICATED USER represented by user_client (publishable key + their
    JWT — see app/auth.py get_current_user).

    Calls the match_chunks() Postgres function via .rpc(), not a raw
    .select(). match_chunks is SECURITY INVOKER (migration 006), so this
    query runs under the CALLER's own Postgres role, and the existing RLS
    policy on `chunks` (role_scoped_chunk_access) filters rows before they
    ever reach this process. There is no manual scope filter here, and
    there must never be one added later -- that would just be app-layer
    filtering hiding behind a database function, which defeats the whole
    point of proving enforcement at the retrieval layer.
    """
    query_embedding = embed_query(question)

    result = user_client.rpc(
        "match_chunks",
        {"query_embedding": query_embedding, "match_count": match_count},
    ).execute()

    return result.data