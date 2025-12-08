# scripts/test_selection_agent.py

import os
import sys

# Make sure Python can find src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.autolit.multi_paper_agents import (
    load_all_summaries,
    select_papers_for_topic,
)


def main():
    summaries_dir = "outputs/summaries"

    print(f"Loading summaries from: {summaries_dir}")
    papers = load_all_summaries(summaries_dir)
    print(f"Loaded {len(papers)} summaries.")

    # 👇 Change this to any topic you care about
    topic = "differentially private language models"
    top_k = 1

    print(f"\nSelecting top {top_k} papers for topic: {topic!r}\n")

    selected = select_papers_for_topic(topic, papers, top_k=top_k)

    if not selected:
        print("No papers selected. Something is off.")
        return

    print("Selected papers (in model-chosen order):\n")
    for i, p in enumerate(selected, start=1):
        task = p.summary.get("task", "(no task field)")
        print(f"{i}. paper_id = {p.paper_id}")
        print(f"   task     = {task}")
        print()

if __name__ == "__main__":
    main()

