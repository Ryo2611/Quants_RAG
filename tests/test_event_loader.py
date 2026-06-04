import pandas as pd
import pytest

from src.event_loader import validate_events


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "MSFT_2025_Q2",
                "ticker": "MSFT",
                "company_name": "Microsoft",
                "fiscal_year": 2025,
                "quarter": "Q2",
                "document_type": "10Q",
                "doc_id": "MSFT_2025_Q2_10Q",
                "earnings_date": "2025-01-29",
                "filing_date": "2025-01-29",
            }
        ]
    )


def test_validate_events_normalizes_dates_and_case() -> None:
    events = _valid_events()
    events.loc[0, "ticker"] = "msft"
    validated = validate_events(events)

    assert validated.loc[0, "ticker"] == "MSFT"
    assert pd.api.types.is_datetime64_any_dtype(validated["earnings_date"])


def test_validate_events_rejects_doc_id_mismatch() -> None:
    events = _valid_events()
    events.loc[0, "doc_id"] = "MSFT_2025_Q1_10Q"

    with pytest.raises(ValueError, match="doc_id"):
        validate_events(events)
