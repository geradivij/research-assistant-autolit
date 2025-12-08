# src/autolit/multi_paper_agents.py

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from src.autolit.llm.client import chat


# ---------------------------------------------------------
# Data class to represent each paper and its Phase 2 summary
# ---------------------------------------------------------
@dataclass
class PaperSummary:
    paper_id: str
    summary: Dict[str, Any]


# ---------------------------------------------------------
# STEP 1 — Load all Phase 2 summaries
# ---------------------------------------------------------
def load_all_summaries(summaries_dir: str) -> List[PaperSummary]:
    """
    Load all JSON summaries from the given directory.
    Each file expected to be <paper_id>.json
    """
    papers = []

    if not os.path.isdir(summaries_dir):
        raise RuntimeError(f"Summaries directory does not exist: {summaries_dir}")

    for fname in os.listdir(summaries_dir):
        if not fname.endswith(".json"):
            continue

        paper_id = fname.replace(".json", "")
        path = os.path.join(summaries_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers.append(PaperSummary(paper_id=paper_id, summary=data))

    if not papers:
        raise RuntimeError("No summaries found. Did you run Phase 2?")

    return papers

# ---------------------------------------------------------
# Helper — robust JSON parsing (handles ```json fences)
# ---------------------------------------------------------
def _parse_json_from_model(raw: str) -> Any:
    """
    Llama sometimes wraps JSON in ```json ... ``` fences.
    This helper strips those and parses the JSON.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove first line (``` or ```json)
        lines = text.splitlines()
        # Drop the first line
        lines = lines[1:]
        # If last line is ``` then drop it too
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)

# ---------------------------------------------------------
# STEP 2 — Paper Selection Agent
# ---------------------------------------------------------
def select_papers_for_topic(
    topic: str,
    papers: List[PaperSummary],
    top_k: int = 3,
) -> List[PaperSummary]:
    """
    Use Llama (via chat()) to choose the most relevant papers for a given topic,
    based on their Phase 2 summaries.

    Returns a list of up to top_k PaperSummary objects.
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

    # This is what we send to the model as user content
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
        '  \"selected_papers\": [\"paper_id1\", \"paper_id2\", ...]\n'
        "}\n"
        "- The list should be sorted from most relevant to least relevant.\n"
        "- Do not include any explanation or text outside the JSON.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    # Call Llama
    raw = chat(messages, temperature=0.0)

    # Parse JSON robustly
    data = _parse_json_from_model(raw)

    selected_ids = data.get("selected_papers", [])
    if not isinstance(selected_ids, list):
        raise RuntimeError("Model returned 'selected_papers' in an unexpected format.")

    # Map IDs back to PaperSummary objects
    id_set = set(selected_ids)
    selected: List[PaperSummary] = [p for p in papers if p.paper_id in id_set]

    # Keep order of selected_ids (model's relevance ordering)
    # by sorting according to index in selected_ids
    selected_sorted = sorted(
        selected,
        key=lambda p: selected_ids.index(p.paper_id),
    )

    return selected_sorted[:top_k]

# ---------------------------------------------------------
# STEP 3 — Comparator Agent
# ---------------------------------------------------------
def compare_papers(
    papers: List[PaperSummary],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Ask Llama to:
      1) Build a comparison table across the given papers.
      2) Write a natural-language critique.

    Returns:
      comparison_table: list of dicts (one row per paper)
      critique: string
    """

    if not papers:
        raise ValueError("compare_papers called with empty paper list.")

    # Prepare data for the model – one dict per paper
    paper_entries: List[Dict[str, Any]] = []
    for p in papers:
        entry = {
            "paper_id": p.paper_id,
            **p.summary,   # unpack your Phase 2 JSON fields
        }
        paper_entries.append(entry)

    payload = {"papers": paper_entries}
    user_json = json.dumps(payload, indent=2)

    system_prompt = (
        "You are a Comparator Agent for machine learning research papers.\n"
        "You will receive a list of paper summaries, each including a paper_id and fields like "
        "task, approach, datasets, metrics, key_results, limitations, and notes.\n\n"
        "Your job has two parts:\n"
        "1) Build a COMPARISON TABLE, where each row corresponds to one paper and has keys:\n"
        "   - paper_id (exactly as given)\n"
        "   - title (if available in the summary; otherwise, invent a short descriptive title)\n"
        "   - task\n"
        "   - approach\n"
        "   - datasets\n"
        "   - metrics\n"
        "   - key_results\n"
        "   - strengths\n"
        "   - weaknesses\n"
        "2) Write a NATURAL LANGUAGE CRITIQUE comparing the papers, discussing similarities,\n"
        "   differences, trade-offs, and when one approach might be preferred over another.\n\n"
        "You MUST return ONLY a valid JSON object with exactly two keys:\n"
        "{\n"
        "  \"comparison_table\": [ { ...row1... }, { ...row2... }, ... ],\n"
        "  \"critique\": \"multi-paragraph text here\"\n"
        "}\n"
        "Do not include any explanation or text outside this JSON. No markdown, no comments."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    # Call Llama
    raw = chat(messages, temperature=0.1)

    # Parse JSON robustly
    data = _parse_json_from_model(raw)

    comparison_table = data.get("comparison_table", [])
    critique = data.get("critique", "")

    if not isinstance(comparison_table, list):
        raise RuntimeError("'comparison_table' is not a list in model output.")

    if not isinstance(critique, str):
        raise RuntimeError("'critique' is not a string in model output.")
