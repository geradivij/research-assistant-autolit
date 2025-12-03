"""
summarize_paper.py

Usage:
    python scripts/summarize_paper.py --paper_id testpdf

This will:
- Load / build the index for data/pdfs/testpdf.pdf
- Run the extraction agent
- Save outputs/summaries/testpdf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# --- Make sure Python can see the src/ directory ---
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolit.extraction_agent import summarize_paper_structured

# --- Output directory for summaries ---
OUTPUT_DIR = ROOT_DIR / "outputs" / "summaries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a single paper into structured JSON."
    )
    parser.add_argument(
        "--paper_id",
        required=True,
        help="Paper ID (PDF filename without .pdf). Example: testpdf",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model name (default: llama3)",
    )

    args = parser.parse_args()

    paper_id = args.paper_id
    model_name = args.model

    print(f"[summarize_paper] Summarizing paper_id={paper_id} using model={model_name}...")

    summary = summarize_paper_structured(
        paper_id=paper_id,
        model_name=model_name,
    )

    # Convert Pydantic model to plain dict
    summary_dict = summary.model_dump()

    out_path = OUTPUT_DIR / f"{paper_id}.json"
    out_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    print(f"[summarize_paper] Saved summary to: {out_path}")
    print("\n--- Summary (preview) ---")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
