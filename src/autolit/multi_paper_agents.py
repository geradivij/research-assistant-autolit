"""
Compatibility shim for the Phase 3 refactor.

This module re-exports the public Phase‑3 API from the new
`src.autolit.phase3` package so existing scripts that import
`src.autolit.multi_paper_agents` continue to work during migration.

Once callers are updated to import from `src.autolit.phase3.*` this
shim can be removed.
"""

# Re-export commonly used Phase‑3 symbols from the new package.
from src.autolit.phase3.io import load_all_summaries, save_survey_markdown, PaperSummary
from src.autolit.phase3.selection_agent import select_papers_for_topic
from src.autolit.phase3.comparator_agent import compare_papers
from src.autolit.phase3.writer_agent import write_survey_markdown
from src.autolit.phase3.pipeline import run_survey_pipeline

# Public API provided by this shim
__all__ = [
    "PaperSummary",
    "load_all_summaries",
    "save_survey_markdown",
    "select_papers_for_topic",
    "compare_papers",
    "write_survey_markdown",
    "run_survey_pipeline",
]