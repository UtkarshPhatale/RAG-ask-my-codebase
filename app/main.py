import time

from fastapi import BackgroundTasks, FastAPI, Depends

from app.auth import get_current_user, AuthedUser
from app.schemas import AskRequest, AskResponse, Citation
from app.retrieval import DEFAULT_K, retrieve_chunks, lookup_documents
from app.generation import generate_answer
from app.query_log import write_query_log

app = FastAPI(title="RAG-ask-my-codebase")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def me(user: AuthedUser = Depends(get_current_user)):
    return {"user_id": user.user_id, "email": user.email, "role": user.role}


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    background: BackgroundTasks,
    user: AuthedUser = Depends(get_current_user),
):
    t0 = time.perf_counter()
    chunks = retrieve_chunks(user.client, request.question, match_count=DEFAULT_K)
    docs = lookup_documents(user.client, [c["document_id"] for c in chunks])
    t1 = time.perf_counter()
    answer = generate_answer(request.question, chunks)
    t2 = time.perf_counter()

    citations = [
        Citation(
            chunk_id=c["id"],
            repo=docs.get(c["document_id"], {}).get("repo", "unknown"),
            path=docs.get(c["document_id"], {}).get("path", "unknown"),
            required_scope=c["required_scope"],
            similarity=round(c["similarity"], 4),
            content=c["content"],
        )
        for c in chunks
    ]

    # Written after the response is sent, as the caller (RLS applies). Never raises.
    background.add_task(
        write_query_log,
        user.client,
        user_id=user.user_id,
        role=user.role,
        question=request.question,
        k=DEFAULT_K,
        citations=citations,
        retrieval_ms=round((t1 - t0) * 1000),
        generation_ms=round((t2 - t1) * 1000),
        total_ms=round((t2 - t0) * 1000),
    )

    return AskResponse(
        question=request.question,
        role=user.role,
        answer=answer,
        sources=[c["content"] for c in chunks],
        citations=citations,
    )
