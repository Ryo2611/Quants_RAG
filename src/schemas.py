from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Signal = Literal["positive", "neutral", "negative", "mixed", "unknown"]


class PageDocument(BaseModel):
    doc_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    quarter: str
    document_type: str
    page: int
    text: str


class ChunkDocument(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    quarter: str
    document_type: str
    page: int
    chunk_index: int
    text: str


class Evidence(BaseModel):
    doc_id: str
    page: int
    text: str


class RetrievedChunk(ChunkDocument):
    similarity_score: float | None = None
    lexical_score: float | None = None


class RagAnswer(BaseModel):
    answer: str
    signal: Signal = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
