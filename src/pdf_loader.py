from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from src.config import COMPANY_NAMES, PAGES_JSONL, RAW_PDF_DIR, ensure_data_dirs
from src.schemas import PageDocument

FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Za-z]+)_(?P<fiscal_year>\d{4})_(?P<quarter>Q[1-4])_(?P<document_type>[A-Za-z0-9-]+)\.pdf$"
)


def parse_pdf_filename(filename: str) -> dict:
    """Parse filename like MSFT_2024_Q1_10Q.pdf and return metadata."""
    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError("PDF filename must match {ticker}_{fiscal_year}_{quarter}_{document_type}.pdf")

    metadata = match.groupdict()
    ticker = metadata["ticker"].upper()
    fiscal_year = int(metadata["fiscal_year"])
    document_type = metadata["document_type"].upper()
    doc_id = f"{ticker}_{fiscal_year}_{metadata['quarter']}_{document_type}"
    return {
        "doc_id": doc_id,
        "ticker": ticker,
        "company_name": COMPANY_NAMES.get(ticker, ticker),
        "fiscal_year": fiscal_year,
        "quarter": metadata["quarter"],
        "document_type": document_type,
    }


def extract_pages_from_pdf(pdf_path: str | Path) -> list[PageDocument]:
    """Extract text from each page of a PDF using PyMuPDF."""
    path = Path(pdf_path)
    metadata = parse_pdf_filename(path.name)
    pages: list[PageDocument] = []
    try:
        import fitz

        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                pages.append(PageDocument(**metadata, page=page_index, text=page.get_text("text").strip()))
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to extract PDF text. Install it with `pip install pymupdf`.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to extract PDF text from {path}") from exc
    return pages


def save_pages_to_jsonl(pages: list[PageDocument], output_path: str | Path) -> None:
    """Save extracted page documents as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for page in pages:
            file.write(json.dumps(page.model_dump(), ensure_ascii=False) + "\n")


def load_pages_from_jsonl(input_path: str | Path) -> list[PageDocument]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Page JSONL not found: {path}")
    pages = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                pages.append(PageDocument.model_validate_json(line))
    return pages


def extract_all_pdfs(raw_pdf_dir: str | Path = RAW_PDF_DIR) -> list[PageDocument]:
    pdf_paths = sorted(Path(raw_pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_pdf_dir}")
    pages: list[PageDocument] = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting PDFs"):
        pages.extend(extract_pages_from_pdf(pdf_path))
    return pages


def main() -> None:
    ensure_data_dirs()
    pages = extract_all_pdfs(RAW_PDF_DIR)
    save_pages_to_jsonl(pages, PAGES_JSONL)
    print(f"Saved {len(pages)} pages to {PAGES_JSONL}")


if __name__ == "__main__":
    main()
