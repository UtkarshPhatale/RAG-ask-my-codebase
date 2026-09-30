import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "queries.jsonl"


def log_query(
    user_id: str,
    role: str,
    question: str,
    retrieved_chunk_ids: list[str],
    retrieved_scopes: list[str],
) -> None:
    """
    Appends one structured JSON record per /ask call. Deliberately logs
    retrieved_scopes alongside retrieved_chunk_ids -- not just IDs -- so a
    later audit can immediately see "did this user's retrieval ever include
    a senior_engineer-scoped chunk" without needing to cross-reference the
    database separately. This is an observability aid, not a security
    control: RLS has already made the access decision by the time this
    function runs. Logging what was retrieved lets you catch a leak in
    hindsight; it does not prevent one.
    """
    LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "role": role,
        "question": question,
        "retrieved_chunk_count": len(retrieved_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_scopes": retrieved_scopes,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")