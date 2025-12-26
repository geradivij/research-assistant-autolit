# src/autolit/multi_paper_agents.py

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from src.autolit.llm.client import chat


# ---------------------------------------------------------
# Data class to represent each paper and its Phase 2 summary
# ---------------------------------------------------------
@dataclass
class PaperSummary:
    paper_id: str
    summary: Dict[str, Any]


# ---------------------------------------------------------
# STEP 1 — Load all Phase 2 summaries
# ---------------------------------------------------------
def load_all_summaries(summaries_dir: str) -> List[PaperSummary]:
    """
    Load all JSON summaries from the given directory.
    Each file expected to be <paper_id>.json
    """
    papers = []

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

# ---------------------------------------------------------
# Helper — robust JSON parsing (handles ```json fences)
# ---------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]  # drop ``` or ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object {...} from a string.
    This helps if the model adds any stray text before/after.
    """
    s = text.strip()
    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    # If we never closed, return best-effort substring
    return s[start:]


def _parse_json_from_model(raw: str) -> Any:
    """
    Robust JSON parsing:
      - strips ```json fences
      - extracts first {...} JSON object
      - tries json.loads
    """
    text = _strip_code_fences(raw)
    text = _extract_json_object(text)

    return json.loads(text)

def _parse_or_repair_json(raw: str) -> Any:
    """
    Try parse; if it fails, ask the model to repair into valid JSON.
    """
    try:
        return _parse_json_from_model(raw)
    except json.JSONDecodeError:
        # Print raw for debugging once
        print("\n========== MODEL RAW OUTPUT (JSON PARSE FAILED) ==========\n")
        print(raw)
        print("\n========== END MODEL RAW OUTPUT ==========\n")

        # Repair attempt
        repaired = _repair_json_with_model(raw)
        return repaired



def _repair_json_with_model(raw: str) -> Any:
    system_prompt = (
        "You are a strict JSON repair assistant.\n"
        "You will receive text intended to be a JSON object, but it may be invalid.\n"
        "Fix it to become STRICT valid JSON (RFC 8259).\n"
        "Rules:\n"
        "- Output ONLY the JSON object.\n"
        "- Escape all newline characters inside strings as \\n.\n"
        "- Do not use trailing commas.\n"
        "- Use double quotes for all keys and string values.\n"
        "- No markdown fences.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw},
    ]

    fixed = chat(messages, temperature=0.0)
    fixed = _strip_code_fences(fixed)
    fixed = _extract_json_object(fixed)
    return json.loads(fixed)


def _extract_between(text: str, start_tag: str, end_tag: str, *, allow_missing_end: bool = False) -> str:
    s = text

    start = s.find(start_tag)
    if start == -1:
        raise RuntimeError(f"Could not find start tag: {start_tag}")
    start += len(start_tag)

    end = s.find(end_tag, start)
    if end == -1:
        if allow_missing_end:
            # Take everything after the start tag
            return s[start:].strip()
        raise RuntimeError(f"Could not find end tag: {end_tag}")

    return s[start:end].strip()

# ---------------------------------------------------------
# STEP 2 — Paper Selection Agent
# ---------------------------------------------------------
def select_papers_for_topic(
    topic: str,
    papers: List[PaperSummary],
    top_k: int = 3,
) -> List[PaperSummary]:
    """
    Use Llama (via chat()) to choose the most relevant papers for a given topic,
    based on their Phase 2 summaries.

    Returns a list of up to top_k PaperSummary objects.
    """

    # Build a compact view of each paper to send to the model
    paper_descriptions: List[Dict[str, Any]] = []
    for p in papers:
        s = p.summary
        desc = {
            "paper_id": p.paper_id,
            "task": s.get("task", ""),
            "approach": s.get("approach", ""),
            "datasets": s.get("datasets", []),
            "metrics": s.get("metrics", []),
            "key_results": s.get("key_results", ""),
            "notes": s.get("notes", ""),
        }
        paper_descriptions.append(desc)

    # This is what we send to the model as user content
    payload = {
        "topic": topic,
        "top_k": top_k,
        "papers": paper_descriptions,
    }
    user_json = json.dumps(payload, indent=2)

    system_prompt = (
        "You are a Paper Selection Agent for machine learning research.\n"
        "You will be given a research topic and a list of papers with high-level summaries.\n\n"
        "Your job is to select the papers that are most relevant to the topic.\n"
        "Consider how closely their task, approach, datasets, and key results match the topic.\n\n"
        "You MUST return ONLY a valid JSON object of the form:\n"
        "{\n"
        '  \"selected_papers\": [\"paper_id1\", \"paper_id2\", ...]\n'
        "}\n"
        "- The list should be sorted from most relevant to least relevant.\n"
        "- Do not include any explanation or text outside the JSON.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    # Call Llama
    raw = chat(messages, temperature=0.0)

    # Parse JSON robustly
    try:
        data = _parse_json_from_model(raw)
    except json.JSONDecodeError:
        # One repair attempt
        data = _repair_json_with_model(raw)


    selected_ids = data.get("selected_papers", [])
    if not isinstance(selected_ids, list):
        raise RuntimeError("Model returned 'selected_papers' in an unexpected format.")

    # Map IDs back to PaperSummary objects
    id_set = set(selected_ids)
    selected: List[PaperSummary] = [p for p in papers if p.paper_id in id_set]

    # Keep order of selected_ids (model's relevance ordering)
    # by sorting according to index in selected_ids
    selected_sorted = sorted(
        selected,
        key=lambda p: selected_ids.index(p.paper_id),
    )

    return selected_sorted[:top_k]

# ---------------------------------------------------------
# STEP 3 — Comparator Agent
# ---------------------------------------------------------
def compare_papers(papers: List[PaperSummary]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Comparator Agent (delimiter-based to avoid JSON failures):
      - Outputs a JSON comparison table inside BEGIN_JSON/END_JSON
      - Outputs critique text inside BEGIN_CRITIQUE/END_CRITIQUE
    """
    if not papers:
        raise ValueError("compare_papers called with empty paper list.")

    paper_entries: List[Dict[str, Any]] = []
    for p in papers:
        paper_entries.append({"paper_id": p.paper_id, **p.summary})

    user_json = json.dumps({"papers": paper_entries}, indent=2)

    system_prompt = (
        "You are a Comparator Agent for machine learning research papers.\n"
        "You will receive a list of paper summaries.\n\n"
        "You must output EXACTLY TWO BLOCKS in this order:\n"
        "1) A valid JSON object containing ONLY the comparison table\n"
        "2) A natural language critique\n\n"
        "FORMAT:\n"
        "BEGIN_JSON\n"
        "{\n"
        "  \"comparison_table\": [\n"
        "    {\n"
        "      \"paper_id\": \"...\",\n"
        "      \"title\": \"...\",\n"
        "      \"task\": \"...\",\n"
        "      \"approach\": \"...\",\n"
        "      \"datasets\": [\"...\"],\n"
        "      \"metrics\": [\"...\"],\n"
        "      \"key_results\": \"...\",\n"
        "      \"strengths\": \"...\",\n"
        "      \"weaknesses\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "END_JSON\n\n"
        "BEGIN_CRITIQUE\n"
        "(write critique here)\n"
        "END_CRITIQUE\n\n"
        "Rules:\n"
        "- The JSON block MUST be strictly valid JSON.\n"
        "- Do NOT put the critique inside JSON.\n"
        "- Do NOT output anything outside these blocks.\n"
        "The final line of your response MUST be: END_CRITIQUE"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    raw = chat(messages, temperature=0.1)

    # Extract JSON block
    json_block = _extract_between(raw, "BEGIN_JSON", "END_JSON")
    critique_block = _extract_between(raw, "BEGIN_CRITIQUE", "END_CRITIQUE", allow_missing_end=True)


    # Parse JSON safely (strip fences just in case)
    json_block = _strip_code_fences(json_block)
    json_block = _extract_json_object(json_block)
    data = json.loads(json_block)

    table = data.get("comparison_table", [])
    if not isinstance(table, list):
        raise RuntimeError("'comparison_table' is not a list in comparator output.")

    critique = critique_block.strip()
    return table, critique


# ---------------------------------------------------------
# STEP 4 — Writer Agent
# ---------------------------------------------------------
def write_survey_markdown(
    topic: str,
    papers: List[PaperSummary],
    comparison_table: List[Dict[str, Any]],
    critique: str,
) -> str:
    """
    Writer Agent:
      - Takes summaries + comparison table + critique
      - Produces a mini survey in Markdown
    """
    payload = {
        "topic": topic,
        "papers": [{"paper_id": p.paper_id, **p.summary} for p in papers],
        "comparison_table": comparison_table,
        "critique": critique,
    }
    user_json = json.dumps(payload, indent=2)

    system_prompt = (
        "You are a Writer Agent drafting a short survey-style report in Markdown.\n"
        "Using the provided topic, paper summaries, comparison table, and critique, "
        "write a mini survey in MARKDOWN with these sections EXACTLY:\n\n"
        "# Introduction\n"
        "## Individual Papers\n"
        "## Comparison\n"
        "## Limitations\n"
        "## Future Work\n\n"
        "Rules:\n"
        "- Use clear headings and subheadings.\n"
        "- Under 'Individual Papers', include one subsection per paper, using the paper_id in the heading.\n"
        "- In 'Comparison', incorporate the comparison_table and critique.\n"
        "- Output ONLY Markdown (no JSON, no extra commentary).\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    md = chat(messages, temperature=0.3)
    return md.strip()

# ---------------------------------------------------------
# STEP 5 — Orchestrator: select → compare → write → save
# ---------------------------------------------------------
def run_survey_pipeline(
    topic: str,
    summaries_dir: str = "outputs/summaries",
    surveys_dir: str = "outputs/surveys",
    top_k: int = 3,
) -> str:
    """
    End-to-end Phase 3 pipeline:
      1) Load all Phase 2 summaries
      2) Select top_k relevant papers for topic
      3) Compare selected papers
      4) Write survey Markdown
      5) Save markdown to outputs/surveys/<slug>_survey.md

    Returns the output file path.
    """
    os.makedirs(surveys_dir, exist_ok=True)

    # 1) Load
    all_papers = load_all_summaries(summaries_dir)

    # 2) Select
    selected = select_papers_for_topic(topic, all_papers, top_k=top_k)
    if not selected:
        raise RuntimeError("No papers selected for the topic.")

    # 3) Compare
    comparison_table, critique = compare_papers(selected)

    # 4) Write
    md = write_survey_markdown(topic, selected, comparison_table, critique)

    # 5) Save
    slug = (
        topic.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
        .replace("?", "")
        .replace(".", "")
    )
    out_path = os.path.join(surveys_dir, f"{slug}_survey.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")

    return out_path
