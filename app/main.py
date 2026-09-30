from fastapi import FastAPI, Depends

from app.auth import get_current_user, AuthedUser
from app.schemas import AskRequest, AskResponse
from app.retrieval import retrieve_chunks
from app.generation import generate_answer
from app.logging_utils import log_query

app = FastAPI(title="RAG-ask-my-codebase")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def me(user: AuthedUser = Depends(get_current_user)):
    return {"user_id": user.user_id, "email": user.email, "role": user.role}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, user: AuthedUser = Depends(get_current_user)):
    chunks = retrieve_chunks(user.client, request.question)
    answer = generate_answer(request.question, chunks)

    log_query(
        user_id=user.user_id,
        role=user.role,
        question=request.question,
        retrieved_chunk_ids=[c["id"] for c in chunks],
        retrieved_scopes=[c["required_scope"] for c in chunks],
    )

    return AskResponse(
        question=request.question,
        role=user.role,
        answer=answer,
        sources=[c["content"] for c in chunks],
    )