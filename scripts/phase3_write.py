"""
CLI wrapper: run only the writer step and print the generated Markdown.

Useful for validating writer formatting and prompt behavior.
"""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.autolit.multi_paper_agents import load_all_summaries, select_papers_for_topic, compare_papers, write_survey_markdown


def main():
    parser = argparse.ArgumentParser(description="Run Phase3 writer step")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--summaries_dir", default="outputs/summaries")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    papers = load_all_summaries(args.summaries_dir)
    selected = select_papers_for_topic(args.topic, papers, top_k=args.top_k)
    table, critique = compare_papers(selected)

    md = write_survey_markdown(args.topic, selected, table, critique)
    print(md)


if __name__ == "__main__":
    main()
