"""
Selection agent for Phase 3.

This module implements `select_papers_for_topic`, which takes a topic and a
list of `PaperSummary` objects (from `io`) and returns the top-k most
relevant papers according to the LLM.

Parsing and repair of the model's JSON response is delegated to
`contracts.parse_or_repair_json` which uses the project's two-attempt
repair policy.
"""
from __future__ import annotations

from typing import List, Dict, Any
import json
import re

from src.autolit.phase3 import contracts
from src.autolit.phase3.io import PaperSummary
from src.autolit.llm.client import chat


def _normalize_text(x: Any) -> str:
    """Best-effort convert unknown values to searchable text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(_normalize_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(f"{k} {_normalize_text(v)}" for k, v in x.items())
    return str(x)


def _fallback_rank_by_overlap(topic: str, papers: List[PaperSummary], top_k: int) -> List[PaperSummary]:
    """
    Deterministic fallback ranking if the LLM returns unusable output.
    Uses simple token overlap between topic and concatenated summary fields.
    """
    topic_tokens = set(re.findall(r"[a-z0-9]+", topic.lower()))
    if not topic_tokens:
        # If topic is empty, just return first top_k deterministically
        return papers[:top_k]

    scored = []
    for p in papers:
        s = p.summary if isinstance(p.summary, dict) else {}
        blob = " ".join(
            [
                _normalize_text(p.paper_id),
                _normalize_text(s.get("task", "")),
                _normalize_text(s.get("approach", "")),
                _normalize_text(s.get("datasets", "")),
                _normalize_text(s.get("metrics", "")),
                _normalize_text(s.get("key_results", "")),
                _normalize_text(s.get("notes", "")),
            ]
        ).lower()
        blob_tokens = set(re.findall(r"[a-z0-9]+", blob))
        overlap = len(topic_tokens & blob_tokens)
        scored.append((overlap, p))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def select_papers_for_topic(
    topic: str,
    papers: List[PaperSummary],
    top_k: int = 3,
) -> List[PaperSummary]:
    """Select up to `top_k` papers relevant to `topic` using the LLM.

    The function builds a compact representation of each paper's Phase 2
    summary to send to the model.

    The model MUST return a JSON object with a `selected_papers` list of paper_ids.
    JSON parsing and repair is handled by `contracts.parse_or_repair_json`.
    """

    if not isinstance(topic, str):
        raise TypeError(f"topic must be a string, got {type(topic)}")
    topic = topic.strip()

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")

    if not papers:
        return []

    print(f"[selection_agent] topic={topic!r}, top_k={top_k}, num_papers={len(papers)}")

    # Build a compact view of each paper to send to the model
    paper_descriptions: List[Dict[str, Any]] = []
    allowed_ids: List[str] = []
    for p in papers:
        allowed_ids.append(p.paper_id)

        s = p.summary
        if not isinstance(s, dict):
            # Fail fast; otherwise s.get(...) will throw later or silently degrade.
            raise TypeError(f"PaperSummary.summary for {p.paper_id!r} must be a dict; got {type(s)}")

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
        print(f"Prepared paper for selection agent: {p.paper_id}")

    payload = {"topic": topic, "top_k": top_k, "papers": paper_descriptions}
    user_json = json.dumps(payload, ensure_ascii=False)

    allowed_ids_set = set(allowed_ids)

    # IMPORTANT: keep this prompt SHORT and STRICT.
    # Long prompts + examples often trigger "explaining the JSON" behavior.
    system_prompt = (
        "You are a Paper Selection Agent.\n"
        "Given a topic and a list of papers (each has paper_id + short summary fields), "
        "return the most relevant paper_ids.\n\n"
        "Return ONLY valid JSON with exactly one key:\n"
        '{"selected_papers": ["paper_id1", "paper_id2", "..."]}\n\n'
        "Constraints:\n"
        f"- Only use these paper_id values: {json.dumps(allowed_ids, ensure_ascii=False)}\n"
        "- Sort from most relevant to least relevant.\n"
        f"- Return at least {top_k} ids.\n"
        "- No extra keys. No explanations. No markdown.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]

    # If your local `chat()` wrapper supports stop sequences, add them there.
    # Even if it doesn't, leaving this as-is is fine; it won't break anything.
    raw = chat(messages, temperature=0)
    print("Raw selection agent output:")
    print(raw)

    # Parse using the project-wide parsing/repair policy (two attempts)
    try:
        data = contracts.parse_or_repair_json(raw, max_attempts=2)
    except Exception as e:
        print(f"[selection_agent] parse_or_repair_json failed: {e!r}")
        return _fallback_rank_by_overlap(topic, papers, top_k)

    selected_ids = data.get("selected_papers", [])
    if not isinstance(selected_ids, list) or not all(isinstance(x, str) for x in selected_ids):
        print("[selection_agent] 'selected_papers' missing or not a list[str]; using fallback ranker.")
        return _fallback_rank_by_overlap(topic, papers, top_k)

    # Filter to allowed ids + dedupe + preserve order + enforce top_k
    final_ids: List[str] = []
    seen = set()
    for pid in selected_ids:
        if pid in allowed_ids_set and pid not in seen:
            final_ids.append(pid)
            seen.add(pid)
        if len(final_ids) >= top_k:
            break

    if not final_ids:
        print("[selection_agent] model returned no usable ids; using fallback ranker.")
        return _fallback_rank_by_overlap(topic, papers, top_k)

    print(f"[selection_agent] selected_ids={final_ids}")

    # Map back to PaperSummary objects in ranked order
    by_id = {p.paper_id: p for p in papers}
    return [by_id[pid] for pid in final_ids]
