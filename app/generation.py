from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a codebase Q&A assistant. Answer the user's question \
using ONLY the provided context chunks below. If the context doesn't contain \
enough information to answer, say so honestly rather than guessing or using \
outside knowledge. Cite which chunk(s) you used when relevant."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generates an answer from already-retrieved, already-scope-filtered chunks.

    Important: this function has no awareness of roles or scopes at all, by
    design. By the time chunks reach here, RLS has already decided what the
    user is allowed to see (app/retrieval.py). This function's only job is
    to answer from what it's given -- it must never be the place that decides
    what's safe to show, since an LLM cannot be relied on as an access-control
    boundary the way a database constraint can.
    """
    if not chunks:
        return "I couldn't find any relevant information in the codebase to answer this question."

    context = "\n\n---\n\n".join(
        f"[Source {i+1}, scope={c['required_scope']}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    message = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return message.content[0].text