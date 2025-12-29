"""
CLI wrapper: run only the comparator step and print JSON table + critique.

Useful for debugging comparator output formatting.
"""
import argparse
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.autolit.multi_paper_agents import load_all_summaries, select_papers_for_topic, compare_papers


def main():
    parser = argparse.ArgumentParser(description="Run Phase3 comparator step")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--summaries_dir", default="outputs/summaries")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    papers = load_all_summaries(args.summaries_dir)
    selected = select_papers_for_topic(args.topic, papers, top_k=args.top_k)
    table, critique = compare_papers(selected)

    print(json.dumps(table, indent=2))
    print("\n--- CRITIQUE ---\n")
    print(critique)


if __name__ == "__main__":
    main()
