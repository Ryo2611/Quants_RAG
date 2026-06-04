from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EVENTS_CSV

REQUIRED_EVENT_COLUMNS = [
    "event_id",
    "ticker",
    "company_name",
    "fiscal_year",
    "quarter",
    "document_type",
    "doc_id",
    "earnings_date",
    "filing_date",
]


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize earnings event metadata."""
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")

    events = df.copy()
    events["event_id"] = events["event_id"].astype(str)
    duplicate_ids = events.loc[events["event_id"].duplicated(), "event_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate event_id values found: {duplicate_ids}")

    events["ticker"] = events["ticker"].astype(str).str.upper()
    events["document_type"] = events["document_type"].astype(str).str.upper()
    events["quarter"] = events["quarter"].astype(str).str.upper()
    events["earnings_date"] = pd.to_datetime(events["earnings_date"], errors="coerce")
    events["filing_date"] = pd.to_datetime(events["filing_date"], errors="coerce")
    if events[["earnings_date", "filing_date"]].isna().any().any():
        raise ValueError("earnings_date and filing_date must use valid YYYY-MM-DD dates")

    events["fiscal_year"] = events["fiscal_year"].astype(int)
    expected_doc_ids = (
        events["ticker"]
        + "_"
        + events["fiscal_year"].astype(str)
        + "_"
        + events["quarter"]
        + "_"
        + events["document_type"]
    )
    mismatched = events.loc[events["doc_id"].astype(str) != expected_doc_ids, "event_id"].tolist()
    if mismatched:
        raise ValueError(f"doc_id does not match ticker/year/quarter/type for events: {mismatched}")

    return events.sort_values("earnings_date").reset_index(drop=True)


def load_events(path: str | Path = EVENTS_CSV) -> pd.DataFrame:
    """Load earnings event metadata and validate required columns."""
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(f"Earnings events file not found: {event_path}")
    df = pd.read_csv(event_path)
    return validate_events(df)


def main() -> None:
    events = load_events(EVENTS_CSV)
    print(f"Loaded {len(events)} events from {EVENTS_CSV}")


if __name__ == "__main__":
    main()
