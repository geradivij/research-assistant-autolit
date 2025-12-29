"""
Comparator agent for Phase 3.

Produces a JSON comparison table and a natural-language critique. The
agent expects the model to return two delimited blocks: BEGIN_JSON/END_JSON
and BEGIN_CRITIQUE/END_CRITIQUE. The JSON block is parsed using the
project's parsing and repair helpers with two repair attempts.
"""
from typing import List, Dict, Any, Tuple
import json

from src.autolit.phase3 import contracts
from src.autolit.phase3.io import PaperSummary
from src.autolit.llm.client import chat


def compare_papers(papers: List[PaperSummary]) -> Tuple[List[Dict[str, Any]], str]:
    """Compare `papers` and return (comparison_table, critique).

    The comparison_table is a list of dicts describing each paper. The
    critique is a string with the agent's analysis. Raises ValueError for
    empty input and RuntimeError for malformed comparator output.
    """
    if not papers:
        raise ValueError("compare_papers called with empty paper list.")

    paper_entries: List[Dict[str, Any]] = []
    for p in papers:
        paper_entries.append({"paper_id": p.paper_id, **p.summary})

    user_json = json.dumps({"papers": paper_entries}, indent=2)

    system_prompt = (
        "You are a Comparator Agent for machine learning research papers.\n"
        "You will receive a list of paper summaries.\n\n"
        "You must output EXACTLY TWO BLOCKS in this order:\n"
        "1) A valid JSON object containing ONLY the comparison table\n"
        "2) A natural language critique\n\n"
        "FORMAT:\n"
        "BEGIN_JSON\n"
        "{ ... }\n"
        "END_JSON\n\n"
        "BEGIN_CRITIQUE\n"
        "(write critique here)\n"
        "END_CRITIQUE\n\n"
        "Rules:\n"
        "- The JSON block MUST be strictly valid JSON.\n"
        "- Do NOT put the critique inside JSON.\n"
        "- Do NOT output anything outside these blocks.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    raw = chat(messages, temperature=0.1)

    # Extract the delimited blocks
    json_block = contracts._extract_between(raw, "BEGIN_JSON", "END_JSON")
    critique_block = contracts._extract_between(raw, "BEGIN_CRITIQUE", "END_CRITIQUE", allow_missing_end=True)

    # Normalize and parse JSON using the central parser with repair policy
    json_block = contracts._strip_code_fences(json_block)
    # parse_or_repair_json will attempt repairs up to the configured attempts
    data = contracts.parse_or_repair_json(json_block, max_attempts=2)

    table = data.get("comparison_table", [])
    if not isinstance(table, list):
        raise RuntimeError("'comparison_table' is not a list in comparator output.")

    critique = critique_block.strip()
    return table, critique
