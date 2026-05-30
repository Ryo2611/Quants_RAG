import pytest

from src.pdf_loader import parse_pdf_filename


def test_parse_pdf_filename_extracts_metadata() -> None:
    metadata = parse_pdf_filename("MSFT_2024_Q1_10Q.pdf")

    assert metadata == {
        "doc_id": "MSFT_2024_Q1_10Q",
        "ticker": "MSFT",
        "company_name": "Microsoft",
        "fiscal_year": 2024,
        "quarter": "Q1",
        "document_type": "10Q",
    }


def test_parse_pdf_filename_rejects_unexpected_names() -> None:
    with pytest.raises(ValueError):
        parse_pdf_filename("MSFT_2024_10Q.pdf")
