from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    use_rag: bool = True
    editor_code: str = ""


class SourceChunk(BaseModel):
    source: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    code: str
    sources: list[SourceChunk]


class IndexResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    skipped_files: list[str] = []


class IndexStartResponse(BaseModel):
    job_id: str
    status: str


class IndexStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    result: IndexResponse | None = None
