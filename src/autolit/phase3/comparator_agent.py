"""
Comparator agent for Phase 3.

Produces a JSON comparison table and a natural-language critique.
Returns pure JSON with two keys: "comparison_table" and "critique".
"""
from typing import List, Dict, Any, Tuple
import json

from src.autolit.phase3 import contracts
from src.autolit.phase3.io import PaperSummary
from src.autolit.llm.client import chat


def compare_papers(papers: List[PaperSummary]) -> Tuple[List[Dict[str, Any]], str]:
    """Compare `papers` and return (comparison_table, critique)."""
    if not papers:
        raise ValueError("compare_papers called with empty paper list.")

    paper_entries: List[Dict[str, Any]] = []
    for p in papers:
        paper_entries.append({"paper_id": p.paper_id, **p.summary})

    user_json = json.dumps({"papers": paper_entries}, indent=2)

    system_prompt = (
        "You are a Comparator Agent for machine learning research papers.\n"
        "Given a list of paper summaries, return a JSON object with exactly two keys:\n\n"
        '{\n'
        '  "comparison_table": [\n'
        '    {"paper_id": "...", "task": "...", "approach": "...", "key_results": "...", "strengths": "...", "weaknesses": "..."},\n'
        '    ...\n'
        '  ],\n'
        '  "critique": "A paragraph comparing and contrasting the papers."\n'
        '}\n\n'
        "Rules:\n"
        "- Output ONLY valid JSON. No markdown, no extra text.\n"
        "- One entry in comparison_table per paper.\n"
        "- The critique key must be a single string (not an object).\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    raw = chat(messages, temperature=0.1)

    data = contracts.parse_or_repair_json(raw, max_attempts=2)

    table = data.get("comparison_table", [])
    if not isinstance(table, list):
        raise RuntimeError("'comparison_table' is not a list in comparator output.")

    critique = data.get("critique", "")
    if not isinstance(critique, str):
        critique = str(critique)

    return table, critique
