from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    """One retrieved chunk, with enough provenance for a UI to cite it."""
    chunk_id: str
    repo: str
    path: str
    required_scope: str
    similarity: float
    content: str


class AskResponse(BaseModel):
    question: str
    role: str
    answer: str | None = None
    # Chunk text only. Kept for backward compatibility; same order as `citations`.
    sources: list[str] = []
    # Same chunks as `sources`, with repo/path/scope/similarity, for citation display.
    citations: list[Citation] = []


class LoggedChunk(BaseModel):
    chunk_id: str
    repo: str
    path: str
    scope: str
    similarity: float


class LogEntry(BaseModel):
    """One row of query_logs, exactly as RLS lets the caller see it."""
    id: str
    created_at: str
    user_id: str
    role: str
    question: str
    k: int
    retrieved: list[LoggedChunk] = []
    retrieved_count: int
    senior_only_count: int
    retrieval_ms: int | None = None
    generation_ms: int | None = None
    total_ms: int | None = None
