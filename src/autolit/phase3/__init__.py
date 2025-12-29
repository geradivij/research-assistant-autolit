"""
Phase 3 package for multi-paper survey generation.

This package contains modular components extracted from the original
`src/autolit/multi_paper_agents.py` script. Modules:
 - io: filesystem I/O for summaries and surveys
 - contracts: parsing/validation/repair helpers and custom errors
 - selection_agent: selects relevant papers for a topic
 - comparator_agent: produces comparison table + critique
 - writer_agent: generates Markdown surveys
 - pipeline: orchestrates selection -> compare -> write -> save

All modules use absolute imports rooted at `src.autolit` to match the
project's existing import style.
"""

__all__ = [
    "io",
    "contracts",
    "selection_agent",
    "comparator_agent",
    "writer_agent",
    "pipeline",
]
