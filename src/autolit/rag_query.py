"""
rag_query.py

End-to-end RAG for a *single* PDF:
- Load PDF
- Chunk text
- Build or load FAISS index
- Retrieve top-k chunks for a question
- Construct a RAG prompt and send to an LLM (stub for now)
"""

from __future__ import annotations
import ollama


import argparse
import json
from pathlib import Path
from typing import List, Dict

from .pdf_loader import load_pdf_pages
from .chunking import chunk_text
from .vector_store import FaissVectorStore, DEFAULT_EMBEDDING_MODEL


DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "indexes"


def build_index_for_pdf(
    pdf_path: Path,
    index_output_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Ingest one PDF:
    - Load pages
    - Chunk
    - Build FAISS index
    - Save index + metadata + chunks.json

    Returns the chunks list.
    """
    print(f"Loading PDF: {pdf_path}")
    pages = load_pdf_pages(pdf_path)

    print(f"Chunking {len(pages)} pages...")
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    print("Building FAISS index...")
    store, _embeds = FaissVectorStore.from_chunks(chunks, DEFAULT_EMBEDDING_MODEL)

    print(f"Saving index to: {index_output_dir}")
    store.save(index_output_dir)

    # Also save chunks text so we can reconstruct contexts later
    chunks_path = index_output_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    return chunks


def load_chunks(index_dir: Path) -> List[Dict]:
    """
    Load chunks.json from the index directory.
    """
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.json in {index_dir}")

    return json.loads(chunks_path.read_text(encoding="utf-8"))


def build_rag_prompt(question: str, context_chunks: List[Dict]) -> str:
    """
    Simple RAG prompt template.

    You will feed this to your Llama / LLM of choice.
    """
    context_texts = []
    for c in context_chunks:
        page = c["page_num"]
        text = c["text"].strip().replace("\n", " ")
        context_texts.append(f"[Page {page}] {text}")

    context_block = "\n\n".join(context_texts)

    prompt = f"""
You are an AI assistant that answers questions about a single academic paper.

Use ONLY the context provided below. If the answer is not contained in the context,
say you don't know.

Question:
{question}

Context:
{context_block}

Answer in 3-5 sentences, concise and precise.
"""
    return prompt.strip()


def call_llm(prompt: str) -> str:
    """
    Call a local Llama model running via Ollama.
    """

    response = ollama.chat(
        model="llama3",   # or "llama3.2", or whatever you installed
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that answers questions about a single academic paper."
                    "Use the context below to infer the answer as best as you can."
                    "If the context truly does not provide enough information, honestly say you don't know,"
                    "but first try to infer it from the described goals, methods, and results."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    # response structure:
    # {
    #   'model': 'llama3.1',
    #   'created_at': '...',
    #   'message': {'role': 'assistant', 'content': '...'},
    #   ...
    # }
    return response["message"]["content"]



def rag_query_single_paper(
    pdf_filename: str,
    question: str,
    top_k: int = 5,
) -> str:
    """
    High-level function:
    - Determine index dir for this PDF (using filename as 'paper_id')
    - Build index if needed
    - Run similarity search
    - Build RAG prompt
    - Call LLM
    """
    pdf_path = PDF_DIR / pdf_filename
    paper_id = pdf_path.stem  # e.g., "some_paper" from "some_paper.pdf"
    paper_index_dir = INDEX_DIR / paper_id

    if not paper_index_dir.exists():
        print(f"No index found for {paper_id}, building one...")
        chunks = build_index_for_pdf(pdf_path, paper_index_dir)
    else:
        print(f"Index already exists for {paper_id}, loading...")
        chunks = load_chunks(paper_index_dir)

    print("Loading vector store...")
    store = FaissVectorStore.load(paper_index_dir)

    print(f"Searching top {top_k} chunks for question: {question!r}")
    results = store.search(question, k=top_k)

    # Map FAISS indices to chunk dicts
    index_to_chunk = {i: c for i, c in enumerate(chunks)}
    retrieved_chunks = [index_to_chunk[r["index"]] for r in results]

    # Build RAG prompt and query LLM
    prompt = build_rag_prompt(question, retrieved_chunks)
    answer = call_llm(prompt)

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Run simple RAG over a single local PDF."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="PDF filename located in data/pdfs (e.g., paper.pdf)",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the paper",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )

    args = parser.parse_args()

    answer = rag_query_single_paper(
        pdf_filename=args.pdf,
        question=args.question,
        top_k=args.top_k,
    )

    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
