"""
Pipeline orchestrator for Phase 3.

This module wires together the selection, comparator, and writer agents
and persists the final Markdown output using the IO helpers.
"""
from typing import List

from src.autolit.phase3.io import load_all_summaries, save_survey_markdown, PaperSummary
from src.autolit.phase3.selection_agent import select_papers_for_topic
from src.autolit.phase3.comparator_agent import compare_papers
from src.autolit.phase3.writer_agent import write_survey_markdown


def run_survey_pipeline(
    topic: str,
    summaries_dir: str = "outputs/summaries",
    surveys_dir: str = "outputs/surveys",
    top_k: int = 3,
) -> str:
    """End-to-end pipeline: load -> select -> compare -> write -> save.

    Returns the path to the written Markdown file.
    """
    # 1) Load
    all_papers: List[PaperSummary] = load_all_summaries(summaries_dir)

    # 2) Select
    selected = select_papers_for_topic(topic, all_papers, top_k=top_k)
    if not selected:
        raise RuntimeError("No papers selected for the topic.")

    # 3) Compare
    comparison_table, critique = compare_papers(selected)

    # 4) Write
    md = write_survey_markdown(topic, selected, comparison_table, critique)

    # 5) Save
    out_path = save_survey_markdown(md, topic, surveys_dir)
    return out_path
