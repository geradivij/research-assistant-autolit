# scripts/test_comparator_agent.py

import os
import sys

# Make sure Python can find src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.autolit.multi_paper_agents import (
    load_all_summaries,
    select_papers_for_topic,
    compare_papers,
)


def main():
    summaries_dir = "outputs/summaries"
    topic = "differentially private language models"
    top_k = 2

    print(f"Loading summaries from: {summaries_dir}")
    papers = load_all_summaries(summaries_dir)
    print(f"Loaded {len(papers)} summaries.")

    # First, select top_k papers
    print(f"\nSelecting top {top_k} papers for topic: {topic!r}\n")
    selected = select_papers_for_topic(topic, papers, top_k=top_k)

    if not selected:
        print("No papers selected – can't compare.")
        return

    print("Selected papers:")
    for p in selected:
        print(f" - {p.paper_id}")
    print()

    # Now run comparator
    print("Running comparator agent...\n")
    comparison_table, critique = compare_papers(selected)

    print("Comparison table (showing first row if available):\n")
    if comparison_table:
        first = comparison_table[0]
        for k, v in first.items():
            print(f"{k}: {v}")
    else:
        print("Comparison table is empty.")

    print("\nCritique (first 500 chars):\n")
    print(critique[:500])
    print("\n--- END OF CRITIQUE PREVIEW ---")


if __name__ == "__main__":
    main()
