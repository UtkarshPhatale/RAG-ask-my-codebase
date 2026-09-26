from fastapi import FastAPI, Depends

from app.auth import get_current_user, AuthedUser
from app.schemas import AskRequest, AskResponse
from app.retrieval import retrieve_chunks
from app.generation import generate_answer

app = FastAPI(title="RAG-ask-my-codebase")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def me(user: AuthedUser = Depends(get_current_user)):
    return {"user_id": user.user_id, "email": user.email, "role": user.role}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, user: AuthedUser = Depends(get_current_user)):
    """
    Full pipeline: authenticate -> retrieve (RLS-scoped) -> generate.

    generate_answer() receives ONLY chunks that already passed through RLS
    for this specific user -- it has no scope awareness and doesn't need any,
    since the security boundary was already enforced at the database layer
    before this line ever runs.
    """
    chunks = retrieve_chunks(user.client, request.question)
    answer = generate_answer(request.question, chunks)

    return AskResponse(
        question=request.question,
        role=user.role,
        answer=answer,
        sources=[c["content"] for c in chunks],
    )