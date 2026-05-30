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


def load_events(path: str | Path = EVENTS_CSV) -> pd.DataFrame:
    """Load earnings event metadata and validate required columns."""
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(f"Earnings events file not found: {event_path}")
    df = pd.read_csv(event_path)
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")
    df = df.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    return df.sort_values("earnings_date").reset_index(drop=True)


def main() -> None:
    events = load_events(EVENTS_CSV)
    print(f"Loaded {len(events)} events from {EVENTS_CSV}")


if __name__ == "__main__":
    main()
