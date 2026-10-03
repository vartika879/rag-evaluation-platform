from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceCitation(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    sources: list[SourceCitation]