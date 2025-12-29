"""
Writer agent for Phase 3.

This module contains `write_survey_markdown` which takes the selected
papers, comparison table, and critique and asks the LLM to produce a
Markdown-formatted mini-survey. The function returns the Markdown string.
"""
from typing import List, Dict, Any
import json

from src.autolit.phase3.io import PaperSummary
from src.autolit.llm.client import chat


def write_survey_markdown(
    topic: str,
    papers: List[PaperSummary],
    comparison_table: List[Dict[str, Any]],
    critique: str,
) -> str:
    """Generate a Markdown mini-survey for `topic` given `papers`.

    The writer agent is asked to produce a specific section structure
    (Introduction, Individual Papers, Comparison, Limitations, Future Work)
    and to output ONLY Markdown. This function forwards that instruction
    to the LLM and returns the trimmed Markdown string.
    """
    payload = {
        "topic": topic,
        "papers": [{"paper_id": p.paper_id, **p.summary} for p in papers],
        "comparison_table": comparison_table,
        "critique": critique,
    }
    user_json = json.dumps(payload, indent=2)

    system_prompt = (
        "You are a Writer Agent drafting a short survey-style report in Markdown.\n"
        "Using the provided topic, paper summaries, comparison table, and critique, "
        "write a mini survey in MARKDOWN with these sections EXACTLY:\n\n"
        "# Introduction\n"
        "## Individual Papers\n"
        "## Comparison\n"
        "## Limitations\n"
        "## Future Work\n\n"
        "Rules:\n"
        "- Use clear headings and subheadings.\n"
        "- Under 'Individual Papers', include one subsection per paper, using the paper_id in the heading.\n"
        "- In 'Comparison', incorporate the comparison_table and critique.\n"
        "- Output ONLY Markdown (no JSON, no extra commentary).\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    md = chat(messages, temperature=0.3)
    return md.strip()
