"""
pdf_loader.py

Responsible for loading PDF files from disk and extracting their text.
Uses PyMuPDF (pymupdf), which we import as `fitz`.
"""

from pathlib import Path
from typing import List, Dict

import fitz  # pymupdf


def load_pdf_pages(pdf_path: str | Path) -> List[Dict]:
    """
    Load a PDF and extract text page by page.

    Returns a list of dicts:
    [
        {
            "page_num": 1,          # 1-based index
            "text": "Full text of page 1 ..."
        },
        ...
    ]
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict] = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")  # simple text extraction
            pages.append(
                {
                    "page_num": i + 1,  # convert 0-based to 1-based
                    "text": text.strip(),
                }
            )

    return pages


if __name__ == "__main__":
    # Simple manual test: python -m autolit.pdf_loader data/pdfs/your.pdf
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m autolit.pdf_loader <pdf_path>")
        raise SystemExit(1)

    pdf = sys.argv[1]
    pages = load_pdf_pages(pdf)
    print(f"Loaded {len(pages)} pages")
    print("First page preview:\n", pages[0]["text"][:1000])
