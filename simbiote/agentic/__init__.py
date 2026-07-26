"""Natural-language agentic control (spec §6b) — Teammate 4 / Andrew.

command_parser -> task_executor -> robot_tools, backed by a pluggable
`llm_backend`. `agentic_session.py` is the entry point most callers want.
"""
