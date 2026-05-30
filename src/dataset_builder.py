from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EVENTS_CSV, LLM_FEATURES_CSV, MODELING_DATASET_CSV, PRICE_FEATURES_CSV, TARGETS_CSV, ensure_data_dirs

SIGNAL_COLUMNS = ["revenue_signal", "operating_income_signal", "risk_signal", "management_tone", "stock_reaction_signal"]


def build_modeling_dataset(
    events_path: str | Path,
    targets_path: str | Path,
    llm_features_path: str | Path,
    price_features_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge all feature tables and create modeling dataset."""
    events = pd.read_csv(events_path, parse_dates=["earnings_date", "filing_date"])
    targets = pd.read_csv(targets_path, parse_dates=["earnings_date"])
    llm_features = pd.read_csv(llm_features_path)
    price_features = pd.read_csv(price_features_path, parse_dates=["earnings_date"])
    df = events.merge(targets.drop(columns=["ticker", "earnings_date"], errors="ignore"), on="event_id", how="left")
    df = df.merge(llm_features.drop(columns=["ticker", "fiscal_year", "quarter", "document_type", "doc_id"], errors="ignore"), on="event_id", how="left")
    df = df.merge(price_features.drop(columns=["ticker", "earnings_date"], errors="ignore"), on="event_id", how="left")
    for column in SIGNAL_COLUMNS:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown")
    for column in [column for column in df.columns if column.endswith("_confidence")]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for column in df.select_dtypes(include=["number"]).columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median() if not df[column].dropna().empty else 0.0)
    one_hot = pd.get_dummies(df[SIGNAL_COLUMNS], prefix=SIGNAL_COLUMNS, dummy_na=False, dtype=int)
    answer_cols = [column for column in df.columns if column.endswith("_answer") or column.endswith("_evidence")]
    df = pd.concat([df.drop(columns=SIGNAL_COLUMNS + answer_cols, errors="ignore"), one_hot], axis=1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    ensure_data_dirs()
    df = build_modeling_dataset(EVENTS_CSV, TARGETS_CSV, LLM_FEATURES_CSV, PRICE_FEATURES_CSV, MODELING_DATASET_CSV)
    print(f"Saved modeling dataset with {len(df)} rows to {MODELING_DATASET_CSV}")


if __name__ == "__main__":
    main()
