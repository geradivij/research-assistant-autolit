"""
chunking.py

Takes raw page texts and splits them into overlapping text chunks so that:
- Chunks are not too long for embeddings / LLMs
- Overlap helps preserve context across boundaries
"""

from typing import List, Dict


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Given a list of pages from `pdf_loader.load_pdf_pages`, produce chunks.

    Returns a list of dicts:
    [
        {
            "chunk_id": 0,
            "page_num": 1,
            "text": "chunk text...",
        },
        ...
    ]
    """
    chunks: List[Dict] = []
    chunk_id = 0

    for page in pages:
        page_text = page["text"]
        page_num = page["page_num"]

        if not page_text:
            continue

        start = 0
        text_len = len(page_text)

        while start < text_len:
            end = start + chunk_size
            chunk_text_str = page_text[start:end]

            if chunk_text_str.strip():  # avoid empty chunks
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_num": page_num,
                        "text": chunk_text_str.strip(),
                    }
                )
                chunk_id += 1

            # move start forward with overlap
            start += chunk_size - chunk_overlap

    return chunks


if __name__ == "__main__":
    # quick sanity check using fake data
    dummy_pages = [
        {"page_num": 1, "text": "hello " * 300},  # 1500 chars approx
    ]
    chunks = chunk_text(dummy_pages, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks")
    for c in chunks:
        print(c["chunk_id"], "len:", len(c["text"]))
