"""Verification 3 & 4 — instruction parsing, and rejection of bad model output."""

from __future__ import annotations

import json

import pytest

from factoryflow.agentic.command_parser import ParseError, parse_instruction
from factoryflow.agentic.tool_schema import ToolCall, parse_tool_calls, validate_calls


class ScriptedBackend:
    """Replays fixed strings, to test the parser against hostile model output."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


def _names(calls: list[ToolCall]) -> list[str]:
    return [c.tool for c in calls]


# ---- the two acceptance instructions from 6b.5 ---------------------------


def test_simple_instruction(scene, fake_llm):
    calls = parse_instruction("pick up the tray in the supply room", scene, fake_llm)
    assert _names(calls) == ["navigate_to", "pick_up"]
    assert calls[0].args == {"location_id": "supply_room"}
    assert calls[1].args == {"object_id": "tray_01"}


def test_compound_instruction_emits_all_five_states(scene, fake_llm):
    """6b.4's hierarchy, in order — the executor then gates each transition."""
    calls = parse_instruction("move the wheelchair to Room 2", scene, fake_llm)
    assert _names(calls) == [
        "approach_wheelchair", "align_gripper", "attach_handle",
        "nav_with_payload", "detach",
    ]
    assert calls[3].args == {"location_id": "room_2"}


def test_compound_destination_is_not_the_source_room(scene, fake_llm):
    """The wheelchair sits in room_1; 'to Room 2' must win."""
    calls = parse_instruction("move the wheelchair to Room 2", scene, fake_llm)
    assert calls[3].args["location_id"] == "room_2"


def test_pure_navigation(scene, fake_llm):
    calls = parse_instruction("go to the nurse station", scene, fake_llm)
    assert _names(calls) == ["navigate_to"]
    assert calls[0].args == {"location_id": "nurse_station"}


def test_fetch_and_deliver(scene, fake_llm):
    calls = parse_instruction("take the tray to Room 1", scene, fake_llm)
    assert _names(calls) == ["navigate_to", "pick_up", "navigate_to"]
    assert calls[0].args["location_id"] == "supply_room"
    assert calls[2].args["location_id"] == "room_1"


def test_every_plan_validates_against_the_scene(scene, fake_llm):
    for text in (
        "pick up the tray in the supply room",
        "move the wheelchair to Room 2",
        "go to the nurse station",
    ):
        assert validate_calls(parse_instruction(text, scene, fake_llm), scene) == []


# ---- rejection paths -----------------------------------------------------


def test_unparseable_instruction_raises(scene, fake_llm):
    with pytest.raises(ParseError):
        parse_instruction("recalibrate the flux capacitor", scene, fake_llm)


def test_empty_instruction_raises(scene, fake_llm):
    with pytest.raises(ParseError, match="empty instruction"):
        parse_instruction("   ", scene, fake_llm)


def test_hallucinated_id_is_rejected(scene):
    bad = json.dumps({"tool_calls": [{"tool": "navigate_to", "args": {"location_id": "room_7"}}]})
    with pytest.raises(ParseError, match="room_7"):
        parse_instruction("go to room 7", scene, ScriptedBackend(bad))


def test_non_graspable_target_is_rejected(scene):
    bad = json.dumps({"tool_calls": [{"tool": "pick_up", "args": {"object_id": "iv_stand_01"}}]})
    with pytest.raises(ParseError, match="is_graspable"):
        parse_instruction("pick up the IV stand", scene, ScriptedBackend(bad))


def test_malformed_json_is_rejected(scene):
    with pytest.raises(ParseError):
        parse_instruction("do something", scene, ScriptedBackend("I think I should drive north."))


def test_empty_plan_is_a_failure_not_a_silent_noop(scene):
    with pytest.raises(ParseError, match="empty"):
        parse_instruction("do something", scene, ScriptedBackend('{"tool_calls": []}'))


def test_parser_retries_once_with_the_error(scene):
    """A bad first reply gets exactly one corrective retry, not a loop."""
    bad = json.dumps({"tool_calls": [{"tool": "navigate_to", "args": {"location_id": "room_7"}}]})
    good = json.dumps({"tool_calls": [{"tool": "navigate_to", "args": {"location_id": "room_2"}}]})
    backend = ScriptedBackend(bad, good)
    calls = parse_instruction("go to Room 2", scene, backend)
    assert calls[0].args["location_id"] == "room_2"
    assert len(backend.calls) == 2
    assert "was rejected" in backend.calls[1][1]


def test_parser_gives_up_after_two_attempts(scene):
    bad = json.dumps({"tool_calls": [{"tool": "navigate_to", "args": {"location_id": "room_7"}}]})
    backend = ScriptedBackend(bad, bad, bad)
    with pytest.raises(ParseError, match="after 2 attempts"):
        parse_instruction("go to room 7", scene, backend)
    assert len(backend.calls) == 2


# ---- tolerance for real small-model output -------------------------------


def test_tolerates_markdown_fence_and_prose():
    raw = (
        "Sure! Here is the plan:\n```json\n"
        '{"tool_calls": [{"tool": "detach", "args": {}}]}\n'
        "```\nHope that helps."
    )
    assert _names(parse_tool_calls(raw)) == ["detach"]


def test_tolerates_a_bare_array():
    assert _names(parse_tool_calls('[{"tool": "detach", "args": {}}]')) == ["detach"]


def test_tolerates_openai_style_name_and_arguments_keys():
    raw = '{"tool_calls": [{"name": "navigate_to", "arguments": {"location_id": "room_2"}}]}'
    calls = parse_tool_calls(raw)
    assert calls[0].tool == "navigate_to"
    assert calls[0].args == {"location_id": "room_2"}
