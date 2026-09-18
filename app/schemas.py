from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    document: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    document: str = Field(min_length=1)
    question: str = Field(min_length=1)


class Source(BaseModel):
    document: str
    page: int
    chunk_id: int
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)