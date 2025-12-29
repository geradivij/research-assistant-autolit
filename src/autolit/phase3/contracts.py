"""
Contracts, parsing, and repair helpers for Phase 3.

This module centralizes robust parsing of LLM outputs and schema
validation helpers. It implements a two-attempt repair policy using the
LLM: if initial parsing fails, it will call the LLM up to two times to
repair the output into strict JSON. If both repairs fail, a
`ContractRepairError` is raised with diagnostic information.

Keep these helpers small and deterministic so callers can apply
retry/backoff policies around higher-level agent functions.
"""
import json
from typing import Any
from dataclasses import dataclass

from src.autolit.llm.client import chat


class ContractRepairError(RuntimeError):
    """Raised when automatic JSON repair attempts have failed.

    The exception message should include the original raw model output
    and any intermediate repaired attempts to aid debugging.
    """


def _strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences like ``` or ```json."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object {...} from `text`.

    This returns a best-effort substring that starts at the first '{'
    and ends at the matching closing '}'. If no close is found it returns
    the suffix starting from the first '{'.
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

    return s[start:]


def _parse_json_from_model(raw: str) -> Any:
    """Parse JSON robustly from raw model text.

    Steps:
    - strip code fences
    - extract the first JSON object
    - json.loads
    """
    text = _strip_code_fences(raw)
    text = _extract_json_object(text)
    return json.loads(text)


def _repair_json_with_model(raw: str) -> Any:
    """Ask the LLM to repair a raw string into strict JSON and parse it.

    This is a single repair attempt; callers may call this multiple times
    according to their retry policy.
    """
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


def parse_or_repair_json(raw: str, *, max_attempts: int = 2) -> Any:
    """Try to parse `raw` as JSON, otherwise attempt up to `max_attempts`
    LLM repairs before raising `ContractRepairError`.

    Returns the parsed Python object on success.
    """
    try:
        return _parse_json_from_model(raw)
    except json.JSONDecodeError:
        # Store attempts for diagnostics
        attempts = []

        for attempt in range(1, max_attempts + 1):
            try:
                repaired = _repair_json_with_model(raw)
                return repaired
            except Exception as e:
                attempts.append({"attempt": attempt, "error": str(e)})

        # If we reach here, all repairs failed
        msg = (
            "JSON parse failed and automatic repair attempts exhausted.\n"
            f"Raw output:\n{raw}\n\n"
            f"Attempts: {attempts}"
        )
        raise ContractRepairError(msg)


def _extract_between(text: str, start_tag: str, end_tag: str, *, allow_missing_end: bool = False) -> str:
    """Extract a substring between start_tag and end_tag from text.

    If `allow_missing_end` is True and the end tag is not found, return the
    suffix after the start tag.
    """
    s = text

    start = s.find(start_tag)
    if start == -1:
        raise RuntimeError(f"Could not find start tag: {start_tag}")
    start += len(start_tag)

    end = s.find(end_tag, start)
    if end == -1:
        if allow_missing_end:
            return s[start:].strip()
        raise RuntimeError(f"Could not find end tag: {end_tag}")

    return s[start:end].strip()
