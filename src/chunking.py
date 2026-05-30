from __future__ import annotations

import json
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_JSONL, PAGES_JSONL, ensure_data_dirs
from src.pdf_loader import load_pages_from_jsonl
from src.schemas import ChunkDocument, PageDocument


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split long text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - chunk_overlap
    return chunks


def create_chunks_from_pages(
    pages: list[PageDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ChunkDocument]:
    """Convert page-level documents into chunk-level documents."""
    chunks = []
    for page in pages:
        for index, text in enumerate(chunk_text(page.text, chunk_size, chunk_overlap), start=1):
            chunks.append(
                ChunkDocument(
                    chunk_id=f"{page.doc_id}_p{page.page}_c{index}",
                    doc_id=page.doc_id,
                    ticker=page.ticker,
                    company_name=page.company_name,
                    fiscal_year=page.fiscal_year,
                    quarter=page.quarter,
                    document_type=page.document_type,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks


def save_chunks_to_jsonl(chunks: list[ChunkDocument], output_path: str | Path) -> None:
    """Save chunks as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def load_chunks_from_jsonl(input_path: str | Path) -> list[ChunkDocument]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk JSONL not found: {path}")
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(ChunkDocument.model_validate_json(line))
    return chunks


def main() -> None:
    ensure_data_dirs()
    chunks = create_chunks_from_pages(load_pages_from_jsonl(PAGES_JSONL))
    save_chunks_to_jsonl(chunks, CHUNKS_JSONL)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
