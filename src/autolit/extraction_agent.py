"""
extraction_agent.py

Single-paper "Reader Agent" that:
- Ensures a FAISS index exists for the paper
- Gathers good context (abstract/intro + conclusion + key chunks)
- Uses Llama via LangChain + Ollama to extract a structured summary.

Output schema:
- task
- approach
- datasets
- metrics
- key_results
- limitations
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from .vector_store import FaissVectorStore
from .rag_query import (
    DATA_DIR,
    PDF_DIR,
    INDEX_DIR,
    build_index_for_pdf,
    load_chunks,
)


# ---------- Pydantic schema for structured output ----------

class PaperSummary(BaseModel):
    """Structured summary of a single paper."""
    task: str = Field(
        description="What problem or research question does the paper tackle?"
    )
    approach: str = Field(
        description="How does the paper approach the problem? Key methods or models."
    )
    datasets: List[str] = Field(
        default_factory=list,
        description="Names or descriptions of any datasets used.",
    )
    metrics: List[str] = Field(
        default_factory=list,
        description="Evaluation metrics used (e.g., accuracy, BLEU, perplexity).",
    )
    key_results: str = Field(
        description="Main findings or quantitative results of the paper."
    )
    limitations: str = Field(
        description="Stated or implied limitations, caveats, or open questions."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any extra important details not covered above.",
    )


# ---------- Context gathering helpers ----------

def _ensure_index_and_chunks(paper_id: str) -> List[Dict]:
    """
    Make sure FAISS index + chunks exist for this paper.
    Returns the loaded chunks list.
    """
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


def _select_structured_context(
    paper_id: str,
    max_chunks: int = 24,
    per_query_k: int = 4,
) -> str:
    """
    Build a context string by doing field-specific semantic retrieval
    rather than naive first/last-page heuristics.

    We run several targeted queries (task, approach, datasets, metrics, etc.)
    and merge the retrieved chunks.
    """
    index_dir = INDEX_DIR / paper_id

    # Ensure we have chunks + index
    chunks = _ensure_index_and_chunks(paper_id)
    store = FaissVectorStore.load(index_dir)

    idx_to_chunk = {i: c for i, c in enumerate(chunks)}

    # Field-specific retrieval queries
    retrieval_queries = {
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
            "Look for dataset names and descriptions."
        ),
        "metrics": (
            "What evaluation metrics are used to assess the model or method? "
            "Look for metric names such as accuracy, BLEU, F1, perplexity, etc."
        ),
        "key_results": (
            "What are the main quantitative results and performance numbers reported? "
            "Look for tables or sentences with percentages, scores, or improvements."
        ),
        "limitations": (
            "What limitations, weaknesses, or open questions does the paper discuss? "
            "Look for a limitations or future work section, or discussion of drawbacks."
        ),
    }

    seen_ids = set()
    all_chunks: List[Dict] = []

    # Always include a couple of chunks from the first page (abstract/intro)
    page_nums = sorted({c["page_num"] for c in chunks})
    first_page = page_nums[0]
    first_page_chunks = [c for c in chunks if c["page_num"] == first_page][:3]
    for c in first_page_chunks:
        cid = c["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_chunks.append(c)

    # Field-specific semantic retrieval
    for label, query in retrieval_queries.items():
        results = store.search(query, k=per_query_k)
        for r in results:
            i = r["index"]
            if i == -1:
                continue
            c = idx_to_chunk[i]
            cid = c["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(c)

    # Limit total number of chunks
    all_chunks = all_chunks[:max_chunks]

    # Build context block
    context_parts = []
    for c in all_chunks:
        page = c["page_num"]
        text = c["text"].strip().replace("\n", " ")
        context_parts.append(f"[Page {page}] {text}")

    context = "\n\n".join(context_parts)
    return context


# ---------- LLM extraction chain ----------

def _get_structured_llm(model_name: str = "llama3") -> ChatOllama:
    """
    Initialize ChatOllama and wrap it with structured output using PaperSummary.
    """
    base_llm = ChatOllama(
        model=model_name,
        temperature=0.1,
    )
    structured_llm = base_llm.with_structured_output(PaperSummary)
    return structured_llm


def summarize_paper_structured(
    paper_id: str,
    model_name: str = "llama3",
) -> PaperSummary:
    """
    Main entrypoint: given a paper_id (PDF name without .pdf),
    return a PaperSummary Pydantic object.

    - Builds/gathers context
    - Calls Llama via LangChain with structured output
    """
    print(f"[extraction] Building context for paper_id={paper_id}...")
    context = _select_structured_context(paper_id)

    prompt = f"""
You are an AI assistant that reads academic papers and extracts structured information.

You will be given context from a single paper (abstract, introduction, conclusion, and other key parts).
Based ONLY on this context, infer and fill in the following fields in full detailed sentences:

- task: What is the main problem or research question? Write in atleast 2 full sentences
- approach: What methods, models, or techniques are used? Write in atleast 2 full sentences
- datasets: Names/descriptions of datasets used (if any). List everything you can find by name. Empty list if none mentioned.
- metrics: Evaluation metrics used (e.g., accuracy, BLEU, perplexity). List everything you can find by name. Empty list if none mentioned.
- key_results: Main findings or quantitative results, must include numbers if present. Write in full sentences
- limitations: Stated or implied limitations or open questions.
- notes: Include a one paragraph summary of the entire paper

If something is not clearly stated, make a best-effort inference, but don't hallucinate wildly.
Keep text concise but specific.

Context:
{context}
"""

    llm = _get_structured_llm(model_name=model_name)

    # This returns a PaperSummary instance directly thanks to structured output.
    summary: PaperSummary = llm.invoke(prompt)
    return summary
