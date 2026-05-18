from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus


class VocabTermSchemeResponse(BaseModel):
    rdf_types: str = Field(...)
    properties: list[str] = Field(default_factory=list)
    applicable_relationships: list[str] = Field(default_factory=list)
    count: int = Field(...)

class VocabSchemeInfoResponse(BaseModel):
    identifier: str
    source: str
    rdf_format: str
    num_triples: int
    description: str | None = None
    vocab_term_schemes: list[VocabTermSchemeResponse] = Field(default_factory=list)

class VocabEmbeddingUpdateResponse(BaseModel):
    pending_updates: int
    task_status: TaskStatus