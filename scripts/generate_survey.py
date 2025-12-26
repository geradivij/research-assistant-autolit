import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.autolit.multi_paper_agents import run_survey_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate a mini survey from multiple paper summaries.")
    parser.add_argument("--topic", type=str, required=True, help="Topic string to search/compare papers for")
    parser.add_argument("--top_k", type=int, default=3, help="Number of papers to select")
    parser.add_argument("--summaries_dir", type=str, default="outputs/summaries", help="Where Phase 2 summaries live")
    parser.add_argument("--surveys_dir", type=str, default="outputs/surveys", help="Where to write the Markdown survey")

    args = parser.parse_args()

    out_path = run_survey_pipeline(
        topic=args.topic,
        summaries_dir=args.summaries_dir,
        surveys_dir=args.surveys_dir,
        top_k=args.top_k,
    )

    print(f"\n✅ Survey written to: {out_path}\n")


if __name__ == "__main__":
    main()
