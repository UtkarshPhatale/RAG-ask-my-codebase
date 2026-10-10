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
