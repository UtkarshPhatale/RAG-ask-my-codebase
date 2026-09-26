from fastapi import FastAPI, Depends

from app.auth import get_current_user, AuthedUser
from app.schemas import AskRequest, AskResponse
from app.retrieval import retrieve_chunks

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
    Day 3: real retrieval wired in. Chunks come back already scoped by RLS --
    user.client carries this specific user's session, so match_chunks() (a
    SECURITY INVOKER function) can only see rows Postgres decides this user
    is allowed to see. No role check happens in this function's own code.

    Generation (Day 4) still not wired -- answer stays None so today's
    behavior isn't overstated.
    """
    chunks = retrieve_chunks(user.client, request.question)

    return AskResponse(
        question=request.question,
        role=user.role,
        answer=None,
        sources=[c["content"] for c in chunks],
    )