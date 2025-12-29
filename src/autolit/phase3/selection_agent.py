"""
Selection agent for Phase 3.

This module implements `select_papers_for_topic`, which takes a topic and a
list of `PaperSummary` objects (from `io`) and returns the top-k most
relevant papers according to the LLM. Parsing and repair of the model's
JSON response is delegated to `contracts.parse_or_repair_json` which uses
the project's two-attempt repair policy.
"""
from typing import List, Dict, Any
import json

from src.autolit.phase3 import contracts
from src.autolit.phase3.io import PaperSummary
from src.autolit.llm.client import chat


def select_papers_for_topic(
    topic: str,
    papers: List[PaperSummary],
    top_k: int = 3,
) -> List[PaperSummary]:
    """Select up to `top_k` papers relevant to `topic` using the LLM.

    The function builds a compact representation of each paper's Phase 2
    summary to send to the model. The model MUST return a JSON object with
    a `selected_papers` list of paper_ids. The JSON parsing and repair is
    handled by `contracts.parse_or_repair_json`.
    """

    # Build a compact view of each paper to send to the model
    paper_descriptions: List[Dict[str, Any]] = []
    for p in papers:
        s = p.summary
        desc = {
            "paper_id": p.paper_id,
            "task": s.get("task", ""),
            "approach": s.get("approach", ""),
            "datasets": s.get("datasets", []),
            "metrics": s.get("metrics", []),
            "key_results": s.get("key_results", ""),
            "notes": s.get("notes", ""),
        }
        paper_descriptions.append(desc)

    payload = {
        "topic": topic,
        "top_k": top_k,
        "papers": paper_descriptions,
    }
    user_json = json.dumps(payload, indent=2)

    system_prompt = (
        "You are a Paper Selection Agent for machine learning research.\n"
        "You will be given a research topic and a list of papers with high-level summaries.\n\n"
        "Your job is to select the papers that are most relevant to the topic.\n"
        "Consider how closely their task, approach, datasets, and key results match the topic.\n\n"
        "You MUST return ONLY a valid JSON object of the form:\n"
        "{\n"
        '  "selected_papers": ["paper_id1", "paper_id2", ...]\n'
        "}\n"
        "- The list should be sorted from most relevant to least relevant.\n"
        "- Do not include any explanation or text outside the JSON.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    raw = chat(messages, temperature=0.0)

    # Parse using the project-wide parsing/repair policy (two attempts)
    data = contracts.parse_or_repair_json(raw, max_attempts=2)

    selected_ids = data.get("selected_papers", [])
    if not isinstance(selected_ids, list):
        raise RuntimeError("Model returned 'selected_papers' in an unexpected format.")

    # Map IDs back to PaperSummary objects and preserve ordering from model
    id_set = set(selected_ids)
    selected: List[PaperSummary] = [p for p in papers if p.paper_id in id_set]

    selected_sorted = sorted(selected, key=lambda p: selected_ids.index(p.paper_id))

    return selected_sorted[:top_k]
