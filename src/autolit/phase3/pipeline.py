"""
Pipeline orchestrator for Phase 3.

This module wires together the selection, comparator, and writer agents
and persists the final Markdown output using the IO helpers.
"""
from typing import Callable, List, Optional

from src.autolit.phase3.io import load_all_summaries, save_survey_markdown, PaperSummary
from src.autolit.phase3.selection_agent import select_papers_for_topic
from src.autolit.phase3.comparator_agent import compare_papers
from src.autolit.phase3.writer_agent import write_survey_markdown


def run_survey_pipeline(
    topic: str,
    summaries_dir: str = "outputs/summaries",
    surveys_dir: str = "outputs/surveys",
    top_k: int = 3,
    paper_ids: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """End-to-end pipeline: load -> select -> compare -> write -> save.

    Args:
        topic: The research topic to survey.
        summaries_dir: Directory containing Phase 2 JSON summaries.
        surveys_dir: Directory to write the final Markdown survey.
        top_k: Number of papers to select.
        progress_callback: Optional callable(agent, message) fired at each
            major step. Useful for streaming progress to a UI. The ``agent``
            argument is one of "pipeline", "selector", "comparator", "writer".

    Returns the path to the written Markdown file.
    """
    cb = progress_callback or (lambda agent, msg: None)

    # 1) Load
    cb("pipeline", f"Loading summaries from {summaries_dir}...")
    all_papers: List[PaperSummary] = load_all_summaries(summaries_dir, paper_ids=paper_ids)
    cb("pipeline", f"Loaded {len(all_papers)} paper summaries")

    # 2) Select — skip LLM call if we already have top_k or fewer papers
    if len(all_papers) <= top_k:
        selected = all_papers
        cb("selector", f"Using all {len(selected)} papers (pool ≤ top_k={top_k})")
    else:
        cb("selector", f"Selecting top {top_k} from {len(all_papers)} papers for topic: \"{topic}\"")
        selected = select_papers_for_topic(topic, all_papers, top_k=top_k)
        if not selected:
            raise RuntimeError("No papers selected for the topic.")
        cb("selector", f"Selected: {', '.join(p.paper_id for p in selected)}")

    # 3) Compare
    cb("comparator", f"Comparing {len(selected)} papers...")
    comparison_table, critique = compare_papers(selected)
    cb("comparator", f"Comparison table: {len(comparison_table)} rows — critique written")

    # 4) Write
    cb("writer", "Writing Markdown survey...")
    md = write_survey_markdown(topic, selected, comparison_table, critique)
    cb("writer", f"Survey written ({len(md):,} characters)")

    # 5) Save
    out_path = save_survey_markdown(md, topic, surveys_dir)
    cb("pipeline", f"Saved to: {out_path}")

    return out_path
