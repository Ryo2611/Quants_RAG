from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

from src.chunking import load_chunks_from_jsonl
from src.config import (
    CHROMA_COLLECTION_NAME,
    CHUNKS_JSONL,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_DB_DIR,
    ensure_data_dirs,
)
from src.schemas import ChunkDocument

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_warned_embedding_fallback = False
_st_model_cache: Any | None = None


def _get_sentence_transformer(model_name: str) -> Any:
    global _st_model_cache
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    if _st_model_cache is None:
        _st_model_cache = SentenceTransformer(model_name)
    return _st_model_cache


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[A-Za-z0-9$%.-]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0 else [value / norm for value in vector]


class LocalHashEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [_hash_embedding(text, dimensions=self.dimensions) for text in input]


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.fallback = LocalHashEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        global _warned_embedding_fallback
        try:
            model = _get_sentence_transformer(self.model_name)
            return [embedding.tolist() for embedding in model.encode(list(input), show_progress_bar=False)]
        except Exception as exc:
            if not _warned_embedding_fallback:
                print(f"SentenceTransformer embeddings failed; using local hash embeddings. Details: {exc}")
                _warned_embedding_fallback = True
            return self.fallback(input)


def _metadata_from_chunk(chunk: ChunkDocument) -> dict[str, str | int]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "ticker": chunk.ticker,
        "company_name": chunk.company_name,
        "fiscal_year": chunk.fiscal_year,
        "quarter": chunk.quarter,
        "document_type": chunk.document_type,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
    }


def _get_chroma_client(persist_dir: str | Path = VECTOR_DB_DIR):
    return chromadb.PersistentClient(path=str(persist_dir))


def _build_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clean_filters = {
        key: value
        for key, value in filters.items()
        if value not in (None, "", "All") and key in {"ticker", "fiscal_year", "quarter", "document_type"}
    }
    if not clean_filters:
        return None
    if len(clean_filters) == 1:
        key, value = next(iter(clean_filters.items()))
        return {key: {"$eq": value}}
    return {"$and": [{key: {"$eq": value}} for key, value in clean_filters.items()]}


def _lexical_score(query: str, text: str) -> float:
    stopwords = {"a", "an", "and", "are", "as", "by", "did", "for", "from", "main", "of", "or", "the", "to", "were", "what", "with"}
    query_terms = {
        term
        for term in re.findall(r"[A-Za-z][A-Za-z0-9-]+", query.lower())
        if term not in stopwords and len(term) > 2
    }
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for term in query_terms if term in text_lower)
    finance_terms = {"revenue", "growth", "income", "operating", "risk", "uncertainty", "cloud", "azure", "margin"}
    boost = sum(1 for term in finance_terms if term in query_terms and term in text_lower)
    if {"risk", "risks", "uncertainties", "uncertainty"} & query_terms:
        phrases = ["risk factors", "could adversely affect", "may adversely affect", "uncertainty", "adverse economic", "security risks", "geopolitical", "competition"]
    elif {"income", "operating"} & query_terms:
        phrases = ["operating income increased", "operating income decreased", "operating income", "gross margin", "cost of revenue", "operating expenses"]
    else:
        phrases = ["revenue increased", "revenue growth", "driven by", "growth across", "cloud revenue growth", "azure"]
    phrase_boost = sum(1.0 for phrase in phrases if phrase in text_lower)
    boilerplate = ["certification pursuant", "sarbanes-oxley", "signatures", "exhibit", "controls and procedures", "unregistered sales"]
    penalty = sum(1.5 for term in boilerplate if term in text_lower)
    return (matches + boost) / max(len(query_terms), 1) + phrase_boost - penalty


def load_vector_store(persist_dir: str | Path = VECTOR_DB_DIR, require_embeddings: bool = True):
    client = _get_chroma_client(persist_dir)
    embedding_function = None
    if require_embeddings:
        if EMBEDDING_BACKEND == "sentence-transformers":
            embedding_function = SentenceTransformerEmbeddingFunction()
        elif EMBEDDING_BACKEND == "hash":
            embedding_function = LocalHashEmbeddingFunction()
        else:
            raise ValueError("EMBEDDING_BACKEND must be 'hash' or 'sentence-transformers'.")
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_store(chunks_path: str | Path = CHUNKS_JSONL, persist_dir: str | Path = VECTOR_DB_DIR, reset_collection: bool = True) -> None:
    ensure_data_dirs()
    chunks = load_chunks_from_jsonl(chunks_path)
    if not chunks:
        raise ValueError("No chunks found to index.")
    if reset_collection:
        client = _get_chroma_client(persist_dir)
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
    collection = load_vector_store(persist_dir)
    for start in tqdm(range(0, len(chunks), 64), desc="Indexing chunks"):
        batch = chunks[start : start + 64]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[_metadata_from_chunk(chunk) for chunk in batch],
        )


def search_similar_chunks(query: str, filters: dict | None = None, top_k: int = TOP_K, persist_dir: str | Path = VECTOR_DB_DIR) -> list[dict]:
    if not query.strip():
        raise ValueError("query must not be empty")
    collection = load_vector_store(persist_dir)
    where = _build_chroma_where(filters)
    result = collection.query(
        query_texts=[query],
        n_results=max(top_k, top_k * 20),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    chunks_by_id: dict[str, dict] = {}
    for document, metadata, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
        item = dict(metadata or {})
        item["text"] = document
        item["similarity_score"] = 1.0 - float(distance) if distance is not None else None
        item["lexical_score"] = _lexical_score(query, document)
        item["_rank_score"] = (item["similarity_score"] or 0.0) + item["lexical_score"]
        chunks_by_id[str(item.get("chunk_id"))] = item
    lexical_result = collection.get(where=where, include=["documents", "metadatas"])
    for document, metadata in zip(lexical_result.get("documents", []), lexical_result.get("metadatas", [])):
        item = dict(metadata or {})
        chunk_id = str(item.get("chunk_id"))
        lexical_score = _lexical_score(query, document)
        if lexical_score <= 0:
            continue
        if chunk_id in chunks_by_id:
            chunks_by_id[chunk_id]["lexical_score"] = max(chunks_by_id[chunk_id].get("lexical_score", 0.0), lexical_score)
            chunks_by_id[chunk_id]["_rank_score"] = (chunks_by_id[chunk_id].get("similarity_score") or 0.0) + lexical_score
        else:
            item["text"] = document
            item["similarity_score"] = None
            item["lexical_score"] = lexical_score
            item["_rank_score"] = lexical_score
            chunks_by_id[chunk_id] = item
    chunks = list(chunks_by_id.values())
    chunks.sort(key=lambda item: item.get("_rank_score", 0.0), reverse=True)
    for item in chunks:
        item.pop("_rank_score", None)
    return chunks[:top_k]


def get_available_metadata(persist_dir: str | Path = VECTOR_DB_DIR) -> dict[str, list]:
    collection = load_vector_store(persist_dir, require_embeddings=False)
    metadatas = collection.get(include=["metadatas"]).get("metadatas", [])
    fields = ["ticker", "fiscal_year", "quarter", "document_type"]
    return {field: sorted({metadata[field] for metadata in metadatas if metadata and field in metadata}) for field in fields}


def get_available_documents(persist_dir: str | Path = VECTOR_DB_DIR) -> list[dict]:
    collection = load_vector_store(persist_dir, require_embeddings=False)
    documents: dict[str, dict] = {}
    for metadata in collection.get(include=["metadatas"]).get("metadatas", []):
        if not metadata:
            continue
        doc_id = str(metadata["doc_id"])
        documents[doc_id] = {
            "doc_id": doc_id,
            "ticker": metadata["ticker"],
            "company_name": metadata.get("company_name", metadata["ticker"]),
            "fiscal_year": metadata["fiscal_year"],
            "quarter": metadata["quarter"],
            "document_type": metadata["document_type"],
        }
    return sorted(documents.values(), key=lambda item: (str(item["ticker"]), int(item["fiscal_year"]), str(item["quarter"]), str(item["document_type"])))


def main() -> None:
    build_vector_store(CHUNKS_JSONL, VECTOR_DB_DIR, reset_collection=True)
    print(f"Built Chroma collection '{CHROMA_COLLECTION_NAME}' in {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
