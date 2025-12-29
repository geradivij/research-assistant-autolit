"""
IO helpers for Phase 3.

This module contains functions to load Phase 2 summaries from disk and
to save the generated Markdown survey. Functions are intentionally
small and focused so they can be tested in isolation.
"""
from dataclasses import dataclass
import os
import json
from typing import List, Dict, Any


def slugify(text: str) -> str:
    """Create a filesystem-safe slug from `text`.

    This is used to name survey output files deterministically.
    """
    s = text.lower().strip()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    for ch in [":", "?", "."]:
        s = s.replace(ch, "")
    return s


@dataclass
class PaperSummary:
    """Lightweight container for a paper id and its Phase 2 summary.

    Using a small dataclass here mirrors the original structure and keeps
    IO concerns separate from agent logic.
    """
    paper_id: str
    summary: Dict[str, Any]


def load_all_summaries(summaries_dir: str) -> List[PaperSummary]:
    """Load all JSON summaries from `summaries_dir`.

    Each file is expected to be named `<paper_id>.json`. Returns a list of
    `PaperSummary` instances. Raises a RuntimeError when no summaries are
    found or the directory doesn't exist.
    """
    papers: List[PaperSummary] = []

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


def save_survey_markdown(md: str, topic: str, surveys_dir: str) -> str:
    """Save the Markdown survey `md` for `topic` into `surveys_dir`.

    Returns the path written to. Creates `surveys_dir` if needed.
    """
    os.makedirs(surveys_dir, exist_ok=True)

    slug = slugify(topic)
    out_path = os.path.join(surveys_dir, f"{slug}_survey.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")

    return out_path
