"""
extraction_agent.py — Per-Field RAG Extraction

Phase 2 “Reader Agent” redesigned for robustness:
- NO giant multi-field JSON prompt.
- NO multi-paragraph mega summary generation.
- Each field extracted independently using:
    (1) field-specific semantic retrieval
    (2) small curated context
    (3) a tiny, strict question to the LLM
- Final output assembled as a clean PaperSummary Pydantic model.

Output JSON includes:
- task (3–5 sentences)
- approach (4–6 sentences)
- datasets (list[str])
- metrics (list[str])
- key_results (4–6 sentences)
- limitations (3–5 sentences)
- notes (150–250 word overview)
"""

from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import json

from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama

from .vector_store import FaissVectorStore
from .rag_query import (
    PDF_DIR,
    INDEX_DIR,
    build_index_for_pdf,
    load_chunks,
)


# ----------------------------
#  Pydantic Schema
# ----------------------------

class PaperSummary(BaseModel):
    task: str
    approach: str
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    key_results: str
    limitations: str
    notes: str


# ----------------------------
#  Load/Ensure Index
# ----------------------------

def _ensure_index_and_chunks(paper_id: str) -> List[Dict]:
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    index_dir = INDEX_DIR / paper_id

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for paper_id={paper_id}: {pdf_path}")

    if not index_dir.exists():
        print(f"[extraction] No index for {paper_id}, building...")
        chunks = build_index_for_pdf(pdf_path, index_dir)
    else:
        print(f"[extraction] Using existing index for {paper_id}")
        chunks = load_chunks(index_dir)

    return chunks


# ----------------------------
#  Field-specific Retrieval
# ----------------------------

FIELD_QUERIES = {
    "task": (
        "What problem or task does this paper study? "
        "Look for sentences that describe the goal or research question."
    ),
    "approach": (
        "What is the main method, model, or approach proposed by this paper? "
        "Look for phrases like 'we propose', 'our method', 'our approach'."
    ),
    "datasets": (
        "Which datasets are used in the experiments? "
        "Look for dataset names like CIFAR, ImageNet, WikiText, etc."
    ),
    "metrics": (
        "What evaluation metrics are used to assess the model? "
        "Look for metric names like accuracy, F1, BLEU, perplexity, loss."
    ),
    "key_results": (
        "What are the main quantitative results and performance numbers reported? "
        "Look for percentages, scores, improvements, or trends."
    ),
    "limitations": (
        "What limitations, weaknesses, or future work does the paper discuss? "
        "Look for sentences about drawbacks or open questions."
    ),
    "notes": (
        "Summarize the entire paper: task, approach, experiments, results, significance. "
        "A 150–250 word overview."
    ),
}


def _retrieve_context_for_field(
    paper_id: str,
    field: str,
    chunks: List[Dict],
    per_query_k: int = 5,
    max_chunks: int = 8,
) -> str:
    """Retrieve small, tight context specific to one field."""
    index_dir = INDEX_DIR / paper_id
    store = FaissVectorStore.load(index_dir)

    query = FIELD_QUERIES[field]
    results = store.search(query, k=per_query_k)

    idx_to_chunk = {i: c for i, c in enumerate(chunks)}

    selected = []
    seen = set()

    for r in results:
        i = r["index"]
        if i == -1:
            continue
        ch = idx_to_chunk[i]
        cid = ch["chunk_id"]
        if cid not in seen:
            selected.append(ch)
            seen.add(cid)

        if len(selected) >= max_chunks:
            break

    # Fallback: if FAISS gives nothing, include abstract chunks
    if not selected:
        page_nums = sorted({c["page_num"] for c in chunks})
        first_page = page_nums[0]
        selected = [c for c in chunks if c["page_num"] == first_page][:3]

    out = []
    for c in selected:
        t = c["text"].strip().replace("\n", " ")
        out.append(f"[Page {c['page_num']}] {t}")

    return "\n".join(out)


# ----------------------------
#  LLM Helpers
# ----------------------------

def _get_llm(model_name: str = "llama3") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=0.2,
    )


def _ask_field_llm(field: str, context: str, model_name: str) -> str:
    """Small, simple prompt strictly for one field."""
    llm = _get_llm(model_name)

    if field == "datasets":
        instruction = (
            "List all datasets mentioned. "
            "Output ONLY dataset names, optionally followed by ' - short description'. "
            "ONE dataset per line. "
            "Do NOT include any introductory sentence or notes."
        )
    elif field == "metrics":
        instruction = (
            "List all metrics mentioned. "
            "Output ONLY metric names, optionally followed by ' - short description'. "
            "ONE metric per line. "
            "Do NOT include any introductory sentence or notes."
        )
    elif field == "notes":
        instruction = (
            "Write a 150–250 word paragraph summarizing the entire paper. "
            "Use full sentences. "
            "Do NOT mention 'context', 'request', or that you are summarizing. "
            "Just write the summary directly."
        )
    else:
        # Text fields
        base = {
            "task": "Write 3–5 full sentences describing the task and motivation.",
            "approach": "Write 4–6 full sentences describing the method and approach.",
            "key_results": "Write 4–6 full sentences summarizing the main quantitative findings.",
            "limitations": "Write 3–5 full sentences describing limitations or future work.",
        }[field]
        instruction = (
            base
            + " Do NOT mention 'context', 'request', or that you are answering a question. "
              "Write directly about the paper."
        )

    prompt = f"""
You are an expert research assistant.

Using ONLY the context below, answer the following request.

==REQUEST==
{instruction}

==CONTEXT==
{context}

If the context does not contain the answer, respond exactly with: Not specified.
"""

    print(f"[extraction] Asking LLM for field: {field}")
    resp = llm.invoke(prompt)
    return resp.content.strip()



def _parse_list_field(text: str) -> List[str]:
    """Convert multi-line bullet list to a Python list."""
    lines = [l.strip("-• ").strip() for l in text.splitlines()]
    lines = [l for l in lines if l and l.lower() != "not specified."]
    return lines


# ----------------------------
#  Main Extraction Pipeline
# ----------------------------

def summarize_paper_structured(
    paper_id: str,
    model_name: str = "llama3",
) -> PaperSummary:
    print(f"[extraction] Loading index & chunks for paper: {paper_id}")
    chunks = _ensure_index_and_chunks(paper_id)

    summary_data = {}

    # Extract each field independently
    for field in ["task", "approach", "datasets", "metrics", "key_results", "limitations", "notes"]:
        print(f"\n=== Extracting field: {field} ===")

        context = _retrieve_context_for_field(paper_id, field, chunks)

        raw_answer = _ask_field_llm(field, context, model_name)

        # Clean up list fields
        if field in ["datasets", "metrics"]:
            summary_data[field] = _parse_list_field(raw_answer)
        else:
            summary_data[field] = (
                raw_answer if raw_answer.strip() else "Not specified."
            )

    # Validate via Pydantic
    try:
        summary = PaperSummary(**summary_data)
    except ValidationError as e:
        print(summary_data)
        raise ValueError(
            f"Generated fields failed schema validation:\n{e}"
        )

    return summary
