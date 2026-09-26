from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    role: str
    answer: str | None = None   # None until Day 4 wires generation
    sources: list[str] = []     # empty until Day 3 wires retrieval