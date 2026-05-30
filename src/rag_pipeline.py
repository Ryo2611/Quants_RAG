from __future__ import annotations

import json
import re

import ollama
from pydantic import ValidationError

from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from src.schemas import Evidence, RagAnswer, RetrievedChunk
from src.vector_store import search_similar_chunks

SYSTEM_PROMPT = """You are a careful financial analyst assistant.
Answer the question using only the retrieved context.
If the context is insufficient, say "The retrieved evidence is insufficient."
Always cite document id and page number in evidence.
Do not invent facts or numerical values.
Evidence text must be a short exact excerpt copied from the retrieved context.
Return only JSON. Do not wrap it in markdown."""

QUESTION_SEARCH_EXPANSIONS = {
    "What were the main drivers of revenue growth?": "revenue increased driven by segment growth cloud Azure Microsoft 365 Productivity Business Processes Intelligent Cloud More Personal Computing",
    "Did operating income improve or deteriorate?": "operating income increased decreased margin expenses cost of revenue operating expenses",
    "What risks or uncertainties were mentioned?": "risk factors uncertainties competition security cybersecurity regulation demand macroeconomic",
    "Was the management tone positive, neutral, or negative?": "revenue increased operating income increased decreased growth margin cloud Azure positive neutral negative performance",
    "What factors could affect the stock price after earnings?": "revenue growth operating income margin risks cloud Azure guidance demand AI capital expenditures",
}


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = str(chunk.get("text", ""))
        if len(chunk_text) > 2200:
            chunk_text = chunk_text[:2200] + "..."
        context_blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"doc_id: {chunk.get('doc_id')}",
                    f"page: {chunk.get('page')}",
                    f"chunk_id: {chunk.get('chunk_id')}",
                    f"text: {chunk_text}",
                ]
            )
        )
    schema = {
        "answer": "string",
        "signal": "positive | neutral | negative | mixed | unknown",
        "confidence": "float between 0.0 and 1.0",
        "evidence": [{"doc_id": "string", "page": "integer", "text": "short supporting excerpt"}],
        "limitations": "string",
    }
    return (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{chr(10).join(context_blocks)}\n\n"
        "Instructions:\n"
        "- Answer in 2-4 concise sentences.\n"
        "- Use no more than 3 evidence items.\n"
        "- Each evidence.text value must be copied exactly from one context block.\n"
        "- If the answer requires facts outside the context, say the retrieved evidence is insufficient.\n\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def build_search_query(question: str) -> str:
    return f"{question} {QUESTION_SEARCH_EXPANSIONS.get(question, '')}".strip()


def _fallback_answer(message: str, retrieved_chunks: list[dict] | None = None) -> RagAnswer:
    return RagAnswer(
        answer="The retrieved evidence is insufficient.",
        signal="unknown",
        confidence=0.0,
        evidence=[],
        limitations=message,
        retrieved_chunks=[_to_retrieved_chunk(chunk) for chunk in (retrieved_chunks or [])],
    )


def _question_keywords(question: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", question.lower()))
    aliases = {
        "revenue": {"revenue", "growth", "cloud", "azure", "sales", "commercial", "services"},
        "income": {"income", "operating", "margin", "gross", "expense", "cost"},
        "risks": {"risk", "risks", "uncertainties", "competition", "security", "regulatory", "demand"},
        "tone": {"growth", "increase", "decrease", "improve", "deteriorate", "strong", "weak"},
        "stock": {"revenue", "income", "growth", "margin", "risk", "guidance", "demand"},
    }
    expanded = set(words)
    for key, values in aliases.items():
        if key in words or key.rstrip("s") in words:
            expanded.update(values)
    return expanded


def _split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", compact) if sentence.strip()]


def _best_excerpt(text: str, keywords: set[str], max_chars: int = 420) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text[:max_chars].strip()
    scored = []
    for sentence in sentences:
        lower = sentence.lower()
        scored.append((sum(1 for keyword in keywords if keyword in lower), sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    total = 0
    for score, sentence in scored:
        if score == 0 and selected:
            continue
        if total + len(sentence) > max_chars and selected:
            break
        selected.append(sentence)
        total += len(sentence) + 1
        if len(selected) >= 2:
            break
    excerpt = " ".join(selected) if selected else sentences[0]
    return excerpt[:max_chars].strip()


def _infer_signal(question: str, evidence_text: str) -> str:
    lower = f"{question} {evidence_text}".lower()
    positive_terms = ["growth", "increase", "improve", "improved", "higher", "strong", "expanded"]
    negative_terms = ["risk", "decrease", "decline", "deteriorate", "lower", "uncertainty", "weak"]
    positive = sum(term in lower for term in positive_terms)
    negative = sum(term in lower for term in negative_terms)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def _extractive_fallback_answer(reason: str, question: str, retrieved_chunks: list[dict]) -> RagAnswer:
    if not retrieved_chunks:
        return _fallback_answer(reason, retrieved_chunks)
    keywords = _question_keywords(question)
    evidence = _evidence_from_top_chunks(question, retrieved_chunks, limit=3)
    if not evidence:
        return _fallback_answer(reason, retrieved_chunks)
    evidence_text = " ".join(item.text for item in evidence)
    citations = ", ".join(f"{item.doc_id} p.{item.page}" for item in evidence)
    answer = (
        "Ollama LLM generation was unavailable, so this is an extractive fallback based only on "
        f"retrieved passages. Relevant evidence appears in {citations}. "
        f"Key retrieved excerpts mention: {_best_excerpt(evidence_text, keywords)}"
    )
    return RagAnswer(
        answer=answer,
        signal=_infer_signal(question, evidence_text),
        confidence=0.35,
        evidence=evidence,
        limitations=f"{reason} The answer is extractive, not a full LLM synthesis. Use it to inspect retrieved evidence.",
        retrieved_chunks=[_to_retrieved_chunk(chunk) for chunk in retrieved_chunks],
    )


def _to_retrieved_chunk(chunk: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(chunk.get("chunk_id", "")),
        doc_id=str(chunk.get("doc_id", "")),
        ticker=str(chunk.get("ticker", "")),
        company_name=str(chunk.get("company_name", "")),
        fiscal_year=int(chunk.get("fiscal_year", 0)),
        quarter=str(chunk.get("quarter", "")),
        document_type=str(chunk.get("document_type", "")),
        page=int(chunk.get("page", 0)),
        chunk_index=int(chunk.get("chunk_index", 0)),
        text=str(chunk.get("text", "")),
        similarity_score=chunk.get("similarity_score"),
        lexical_score=chunk.get("lexical_score"),
    )


def _extract_json_from_text(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _align_evidence_to_retrieved_chunks(evidence: list[Evidence], retrieved_chunks: list[dict]) -> list[Evidence]:
    aligned = []
    for item in evidence:
        item_text = " ".join(item.text.split())
        item_lower = item_text.lower()
        matching_chunk = None
        for chunk in retrieved_chunks:
            chunk_text = " ".join(str(chunk.get("text", "")).split())
            if item_lower and item_lower in chunk_text.lower():
                matching_chunk = chunk
                break
        if matching_chunk is None:
            for chunk in retrieved_chunks:
                if chunk.get("doc_id") == item.doc_id and int(chunk.get("page", 0)) == item.page:
                    matching_chunk = chunk
                    break
        if matching_chunk is not None:
            aligned.append(Evidence(doc_id=str(matching_chunk.get("doc_id", item.doc_id)), page=int(matching_chunk.get("page", item.page)), text=item.text))
    return aligned


def _evidence_from_top_chunks(question: str, retrieved_chunks: list[dict], limit: int = 2) -> list[Evidence]:
    keywords = _question_keywords(question)
    evidence = []
    for chunk in retrieved_chunks[:limit]:
        excerpt = _best_excerpt(str(chunk.get("text", "")), keywords)
        if excerpt:
            evidence.append(Evidence(doc_id=str(chunk.get("doc_id", "")), page=int(chunk.get("page", 0)), text=excerpt))
    return evidence


def generate_rag_answer(question: str, retrieved_chunks: list[dict]) -> RagAnswer:
    if not retrieved_chunks:
        return _fallback_answer("No chunks were retrieved.", retrieved_chunks)
    prompt = build_rag_prompt(question, retrieved_chunks)
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
    except Exception as exc:
        return _extractive_fallback_answer(f"Ollama LLM call failed: {exc}. Is Ollama running at {OLLAMA_BASE_URL}?", question, retrieved_chunks)
    content = response.message.content or ""
    payload = _extract_json_from_text(content)
    if payload is None:
        return _extractive_fallback_answer(f"Failed to parse LLM JSON response. Raw output: {content[:300]}", question, retrieved_chunks)
    try:
        evidence = [Evidence(**item) for item in payload.get("evidence", [])]
        aligned_evidence = _align_evidence_to_retrieved_chunks(evidence, retrieved_chunks)
        if not aligned_evidence and "insufficient" not in str(payload.get("answer", "")).lower():
            aligned_evidence = _evidence_from_top_chunks(question, retrieved_chunks)
        return RagAnswer(
            answer=str(payload.get("answer", "The retrieved evidence is insufficient.")),
            signal=payload.get("signal", "unknown"),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=aligned_evidence,
            limitations=str(payload.get("limitations", "The answer is based only on retrieved document chunks and may miss information not retrieved.")),
            retrieved_chunks=[_to_retrieved_chunk(chunk) for chunk in retrieved_chunks],
        )
    except (TypeError, ValueError, ValidationError) as exc:
        return _extractive_fallback_answer(f"Failed to parse LLM JSON response: {exc}", question, retrieved_chunks)


def answer_question(question: str, filters: dict | None = None) -> RagAnswer:
    retrieved_chunks = search_similar_chunks(build_search_query(question), filters=filters, top_k=TOP_K)
    return generate_rag_answer(question, retrieved_chunks)
