"""The tool-call contract between the LLM and the executor — master doc 6b.5.

The parser is told to emit JSON matching this schema; every emitted call is then
validated against it *and against the live scene graph* before anything reaches
the robot. Validating ids against the scene is what stops a hallucinated
``room_7`` from becoming a navigation goal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ToolCall",
    "ToolSpec",
    "TOOL_SPECS",
    "SchemaError",
    "parse_tool_calls",
    "validate_call",
    "validate_calls",
    "tool_schema_text",
]


class SchemaError(ValueError):
    """The model's output was not a usable tool-call list."""


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.tool}({inner})"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    #: arg name -> one of "location_id", "object_id". Empty means no args.
    args: dict[str, str]
    #: Only emitted when Step 2's wheelchair stretch task (5.5) is available.
    wheelchair_only: bool = False


TOOL_SPECS: dict[str, ToolSpec] = {
    "navigate_to": ToolSpec(
        name="navigate_to",
        description="Drive the mobile base to a location. Runs the trained navigation policy.",
        args={"location_id": "location_id"},
    ),
    "pick_up": ToolSpec(
        name="pick_up",
        description=(
            "Approach and grasp a graspable object. Assumes the robot is already "
            "at the object's location — emit navigate_to first if it is not."
        ),
        args={"object_id": "object_id"},
    ),
    "approach_wheelchair": ToolSpec(
        name="approach_wheelchair",
        description="Drive the base into grasp range of a wheelchair's handle.",
        args={"object_id": "object_id"},
        wheelchair_only=True,
    ),
    "align_gripper": ToolSpec(
        name="align_gripper",
        description=(
            "Bring the gripper into alignment with the payload's handle. Separate "
            "from attach_handle so a misalignment fails before a constraint is formed."
        ),
        args={"object_id": "object_id"},
        wheelchair_only=True,
    ),
    "attach_handle": ToolSpec(
        name="attach_handle",
        description="Form a fixed constraint between the end-effector and the payload's handle.",
        args={"object_id": "object_id"},
        wheelchair_only=True,
    ),
    "nav_with_payload": ToolSpec(
        name="nav_with_payload",
        description="Navigate to a location while a payload is attached, using the co-navigation policy.",
        args={"location_id": "location_id"},
        wheelchair_only=True,
    ),
    "detach": ToolSpec(
        name="detach",
        description="Release the fixed constraint and park the payload.",
        args={},
        wheelchair_only=True,
    ),
}


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract a tool-call list from raw model output.

    Tolerates a ```json fence and surrounding prose, because small local models
    add both. Accepts either ``{"tool_calls": [...]}`` or a bare list.
    """
    blob = _extract_json(text)
    if blob is None:
        raise SchemaError(f"no JSON object or array found in model output: {text[:200]!r}")

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"model output was not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        raw_calls = data.get("tool_calls")
        if raw_calls is None:
            raise SchemaError("JSON object has no 'tool_calls' key")
    elif isinstance(data, list):
        raw_calls = data
    else:
        raise SchemaError(f"expected an object or array, got {type(data).__name__}")

    if not isinstance(raw_calls, list):
        raise SchemaError("'tool_calls' must be an array")

    calls: list[ToolCall] = []
    for i, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            raise SchemaError(f"tool_calls[{i}] must be an object, got {type(item).__name__}")
        tool = item.get("tool") or item.get("name")
        if not isinstance(tool, str):
            raise SchemaError(f"tool_calls[{i}] is missing a string 'tool'")
        args = item.get("args", item.get("arguments", {}))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise SchemaError(f"tool_calls[{i}].args must be an object")
        calls.append(ToolCall(tool=tool, args=args))
    return calls


def _extract_json(text: str) -> str | None:
    """Return the first balanced JSON object or array in ``text``."""
    fence = text.find("```")
    if fence != -1:
        end = text.find("```", fence + 3)
        if end != -1:
            inner = text[fence + 3 : end]
            # drop a leading language tag such as "json\n"
            if "\n" in inner:
                first, rest = inner.split("\n", 1)
                if first.strip().isalpha():
                    inner = rest
            found = _first_balanced(inner)
            if found:
                return found
    return _first_balanced(text)


def _first_balanced(text: str) -> str | None:
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_call(call: ToolCall, scene: Any) -> list[str]:
    """Check one call against the schema and the scene. Returns error strings."""
    errors: list[str] = []
    spec = TOOL_SPECS.get(call.tool)
    if spec is None:
        return [f"unknown tool {call.tool!r}; valid tools: {', '.join(sorted(TOOL_SPECS))}"]

    missing = set(spec.args) - set(call.args)
    extra = set(call.args) - set(spec.args)
    for name in sorted(missing):
        errors.append(f"{call.tool}: missing required argument {name!r}")
    for name in sorted(extra):
        errors.append(f"{call.tool}: unexpected argument {name!r}")

    for name, kind in spec.args.items():
        if name not in call.args:
            continue
        value = call.args[name]
        if not isinstance(value, str):
            errors.append(f"{call.tool}.{name} must be a string id, got {type(value).__name__}")
            continue
        if kind == "location_id":
            if not scene.has_location(value):
                errors.append(
                    f"{call.tool}.{name}={value!r} is not a location in the scene "
                    f"(valid: {', '.join(sorted(scene.locations))})"
                )
            else:
                loc = scene.get_location(value)
                if loc is not None and not loc.navigable:
                    errors.append(f"{call.tool}.{name}={value!r} is tagged non-navigable")
        elif kind == "object_id":
            if not scene.has_object(value):
                errors.append(
                    f"{call.tool}.{name}={value!r} is not an object in the scene "
                    f"(valid: {', '.join(sorted(scene.objects))})"
                )
            else:
                obj = scene.get_object(value)
                if obj is not None and not obj.is_graspable:
                    errors.append(
                        f"{call.tool}.{name}={value!r} ({obj.label}) is tagged is_graspable=false"
                    )
    return errors


def validate_calls(calls: list[ToolCall], scene: Any) -> list[str]:
    """Validate a whole sequence. Also rejects an empty plan.

    An empty list means the model did not understand the instruction. Treating
    it as a valid no-op plan would show a robot doing nothing and report success.
    """
    if not calls:
        return ["the plan is empty — no tool calls were produced for this instruction"]
    errors: list[str] = []
    for i, call in enumerate(calls):
        errors.extend(f"tool_calls[{i}]: {e}" for e in validate_call(call, scene))
    return errors


def tool_schema_text(include_wheelchair: bool = True) -> str:
    """Human/LLM-readable tool list for the system prompt."""
    lines = []
    for spec in TOOL_SPECS.values():
        if spec.wheelchair_only and not include_wheelchair:
            continue
        args = ", ".join(f"{n}: <{k}>" for n, k in spec.args.items()) or ""
        lines.append(f"- {spec.name}({args}) — {spec.description}")
    return "\n".join(lines)
