"""
Writes one query_logs row per /ask call (table and policies: migration 008).

Runs as a FastAPI background task AFTER the response is sent, using the caller's own
user-scoped client, so the table's RLS applies: a user can only write rows about
themselves, with their real role. A logging failure must never break or delay an
answer, so every error here is caught and sent to the application logger.

Chunk text is deliberately not stored (rows stay small, and the text is already in the
chunks table); each retrieved chunk is recorded as repo/path/scope/similarity.
Chunks that RLS blocked cannot be logged: they never reach the application.
"""
import logging

from supabase import Client

logger = logging.getLogger(__name__)


def write_query_log(
    client: Client,
    *,
    user_id: str,
    role: str,
    question: str,
    k: int,
    citations: list,
    retrieval_ms: int,
    generation_ms: int,
    total_ms: int,
) -> None:
    try:
        retrieved = [
            {
                "chunk_id": c.chunk_id,
                "repo": c.repo,
                "path": c.path,
                "scope": c.required_scope,
                "similarity": c.similarity,
            }
            for c in citations
        ]
        client.table("query_logs").insert(
            {
                "user_id": user_id,
                "role": role,
                "question": question,
                "k": k,
                "retrieved": retrieved,
                "retrieved_count": len(retrieved),
                "senior_only_count": sum(1 for c in citations if c.required_scope == "senior_engineer"),
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            }
        ).execute()
    except Exception:
        logger.exception("failed to write query_log for user %s", user_id)
