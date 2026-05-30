from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
EVENTS_DIR = PROJECT_ROOT / "data" / "events"
PRICES_DIR = PROJECT_ROOT / "data" / "prices"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest"
MODELS_DIR = PROJECT_ROOT / "models"

PAGES_JSONL = PROCESSED_TEXT_DIR / "pages.jsonl"
CHUNKS_JSONL = PROCESSED_TEXT_DIR / "chunks.jsonl"
EVENTS_CSV = EVENTS_DIR / "earnings_events.csv"
TARGETS_CSV = FEATURES_DIR / "targets.csv"
LLM_FEATURES_CSV = FEATURES_DIR / "llm_features.csv"
PRICE_FEATURES_CSV = FEATURES_DIR / "price_features.csv"
MODELING_DATASET_CSV = FEATURES_DIR / "modeling_dataset.csv"
PREDICTIONS_CSV = PREDICTIONS_DIR / "predictions.csv"
BACKTEST_RESULTS_CSV = BACKTEST_DIR / "backtest_results.csv"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hash").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "earnings_chunks")

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500
TOP_K = 5

COMPANY_NAMES = {
    "MSFT": "Microsoft",
    "AAPL": "Apple",
}

QUESTIONS = [
    "What were the main drivers of revenue growth?",
    "Did operating income improve or deteriorate?",
    "What risks or uncertainties were mentioned?",
    "Was the management tone positive, neutral, or negative?",
    "What factors could affect the stock price after earnings?",
]


def ensure_data_dirs() -> None:
    """Create data directories used by the local pipeline."""
    for path in [
        RAW_PDF_DIR,
        PROCESSED_TEXT_DIR,
        METADATA_DIR,
        VECTOR_DB_DIR,
        EVENTS_DIR,
        PRICES_DIR,
        FEATURES_DIR,
        PREDICTIONS_DIR,
        BACKTEST_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
