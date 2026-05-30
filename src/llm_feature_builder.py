from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import EVENTS_CSV, LLM_FEATURES_CSV, ensure_data_dirs
from src.event_loader import load_events
from src.rag_pipeline import answer_question

FIXED_QUESTIONS = {
    "revenue": "What were the main drivers of revenue growth?",
    "operating_income": "Did operating income improve or deteriorate?",
    "risk": "What risks or uncertainties were mentioned?",
    "management_tone": "Was the management tone positive, neutral, or negative?",
    "stock_reaction": "What factors could affect the stock price after earnings?",
}

SIGNAL_COLUMNS = {
    "revenue": "revenue_signal",
    "operating_income": "operating_income_signal",
    "risk": "risk_signal",
    "management_tone": "management_tone",
    "stock_reaction": "stock_reaction_signal",
}


def run_rag_for_event(event: dict[str, Any], question_key: str, question: str) -> dict:
    """Run existing RAG pipeline for a specific event and question."""
    filters = {
        "ticker": event["ticker"],
        "fiscal_year": int(event["fiscal_year"]),
        "quarter": event["quarter"],
        "document_type": event["document_type"],
    }
    answer = answer_question(question, filters=filters)
    return {
        f"{question_key}_answer": answer.answer,
        f"{question_key}_confidence": answer.confidence,
        SIGNAL_COLUMNS[question_key]: answer.signal,
        f"{question_key}_evidence": json.dumps([item.model_dump() for item in answer.evidence], ensure_ascii=False),
    }


def build_llm_features(events_df: pd.DataFrame, cache_path: str | Path = LLM_FEATURES_CSV, force: bool = False) -> pd.DataFrame:
    """Build cached LLM feature table for all events."""
    cache = Path(cache_path)
    existing = pd.read_csv(cache) if cache.exists() and not force else pd.DataFrame()
    existing_event_ids = set(existing["event_id"]) if not existing.empty and "event_id" in existing.columns else set()
    rows = []
    for _, event_row in events_df.iterrows():
        event = event_row.to_dict()
        if event["event_id"] in existing_event_ids:
            continue
        row = {
            "event_id": event["event_id"],
            "ticker": event["ticker"],
            "fiscal_year": int(event["fiscal_year"]),
            "quarter": event["quarter"],
            "document_type": event["document_type"],
            "doc_id": event["doc_id"],
        }
        for key, question in FIXED_QUESTIONS.items():
            row.update(run_rag_for_event(event, key, question))
        rows.append(row)
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty and not force else new_df
    if not combined.empty:
        combined = combined.drop_duplicates("event_id", keep="last").sort_values("event_id").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache, index=False)
    return combined


def main() -> None:
    ensure_data_dirs()
    features = build_llm_features(load_events(EVENTS_CSV), LLM_FEATURES_CSV, force=False)
    print(f"Saved {len(features)} LLM feature rows to {LLM_FEATURES_CSV}")


if __name__ == "__main__":
    main()
