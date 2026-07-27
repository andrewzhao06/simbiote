"""Natural language -> validated tool calls — spec §6b.5.

``parse_instruction`` is the single LLM call in Step 4. Everything after it is
plain Python: the FSM in :mod:`task_executor` decides when a skill is complete,
not the model (§6b.4).
"""

from __future__ import annotations

from simbiote.agentic.llm_backend import LLMBackend, LLMError
from simbiote.agentic.scene_query import SceneGraph
from simbiote.agentic.tool_schema import (
    SchemaError,
    ToolCall,
    parse_tool_calls,
    tool_schema_text,
    validate_calls,
)

__all__ = ["ParseError", "parse_instruction", "build_system_prompt"]


class ParseError(RuntimeError):
    """The instruction could not be turned into a valid plan."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


_SYSTEM_TEMPLATE = """You control a mobile manipulator robot (an omnidirectional \
4-wheel base with a 7-DOF arm and a 2-finger gripper) working in a hospital.

Turn the operator's instruction into a sequence of tool calls.

AVAILABLE TOOLS
{tools}

SCENE
{inventory}

RULES
1. Reply with JSON only. No prose, no explanation, no markdown fence.
2. Shape: {{"tool_calls": [{{"tool": "<name>", "args": {{...}}}}]}}
3. Use only the exact ids listed under SCENE. Never invent an id.
4. The robot must be at an object's location before grasping it: emit
   navigate_to first unless the instruction says it is already there.
5. To move a wheeled payload such as a wheelchair, emit the full sequence in
   this order: approach_wheelchair, align_gripper, attach_handle,
   nav_with_payload, detach. Never skip a step.
6. If the instruction cannot be carried out with these tools and this scene,
   reply with {{"tool_calls": []}}.
"""


def build_system_prompt(scene: SceneGraph, include_wheelchair: bool = True) -> str:
    """The system prompt, carrying both the tool schema and the live scene.

    Listing real ids is what keeps a small local model from inventing
    ``room_7``; ``validate_calls`` is the backstop for when it does anyway.
    """
    return _SYSTEM_TEMPLATE.format(
        tools=tool_schema_text(include_wheelchair=include_wheelchair),
        inventory=scene.inventory_text(),
    )


def parse_instruction(
    text: str,
    scene: SceneGraph,
    backend: LLMBackend,
    *,
    include_wheelchair: bool = True,
) -> list[ToolCall]:
    """Decompose an instruction into a validated tool-call sequence.

    On a schema or scene-validation failure the model gets exactly one retry
    with the error text appended. One, not a loop: a retry loop that hangs on
    stage is worse than a clean failure the operator can retype.
    """
    if not text or not text.strip():
        raise ParseError("empty instruction")

    system = build_system_prompt(scene, include_wheelchair=include_wheelchair)
    user = text.strip()
    last_raw: str | None = None
    last_errors: list[str] = []

    for attempt in (1, 2):
        try:
            raw = backend.complete(system, user)
        except LLMError as exc:
            raise ParseError(f"model backend failed: {exc}") from exc
        last_raw = raw

        try:
            calls = parse_tool_calls(raw)
        except SchemaError as exc:
            last_errors = [str(exc)]
        else:
            errors = validate_calls(calls, scene)
            if not errors:
                return calls
            last_errors = errors

        if attempt == 1:
            user = (
                f"{text.strip()}\n\n"
                "Your previous reply was rejected:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\nReply again with corrected JSON only."
            )

    raise ParseError(
        "could not produce a valid plan for "
        f"{text.strip()!r} after 2 attempts:\n" + "\n".join(f"  - {e}" for e in last_errors),
        raw=last_raw,
    )
