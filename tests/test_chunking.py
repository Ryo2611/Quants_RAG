import pytest

from src.chunking import chunk_text, create_chunks_from_pages
from src.schemas import PageDocument


def test_chunk_text_returns_empty_for_blank_text() -> None:
    assert chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_overlaps_chunks() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, chunk_overlap=3)
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=5, chunk_overlap=5)


def test_create_chunks_from_pages_preserves_metadata() -> None:
    page = PageDocument(
        doc_id="MSFT_2024_Q1_10Q",
        ticker="MSFT",
        company_name="Microsoft",
        fiscal_year=2024,
        quarter="Q1",
        document_type="10Q",
        page=12,
        text="abc def",
    )

    chunks = create_chunks_from_pages([page], chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "MSFT_2024_Q1_10Q_p12_c1"
    assert chunks[0].ticker == "MSFT"
    assert chunks[0].page == 12
