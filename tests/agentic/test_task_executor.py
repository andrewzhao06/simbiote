"""Verification 5 & 6 — FSM ordering, and what happens when a skill fails."""

from __future__ import annotations

import time

import pytest

from simbiote.agentic.robot_tools import RobotTools, SkillResult, StubBackend
from simbiote.agentic.task_executor import StepStatus, execute
from simbiote.agentic.tool_schema import ToolCall

COMPOUND = [
    ToolCall("approach_wheelchair", {"object_id": "wheelchair_01"}),
    ToolCall("align_gripper", {"object_id": "wheelchair_01"}),
    ToolCall("attach_handle", {"object_id": "wheelchair_01"}),
    ToolCall("nav_with_payload", {"location_id": "room_2"}),
    ToolCall("detach", {}),
]


def test_compound_run_advances_in_order(scene, tools):
    order: list[str] = []
    report = execute(COMPOUND, tools, on_step=lambda s: order.append(s.tool))

    assert report.ok
    assert order == [c.tool for c in COMPOUND]
    assert all(s.status is StepStatus.SUCCEEDED for s in report.steps)
    assert tools.attached_to is None  # detach ran, nothing left held


def test_each_step_completes_before_the_next_starts(scene, tools):
    """The FSM, not the LLM, decides when to advance — so no step may begin
    before the previous one has reported."""
    events: list[tuple[str, str]] = []

    class TracingBackend(StubBackend):
        def run_skill(self, skill, args, scene, on_action=None):
            events.append(("start", skill))
            result = super().run_skill(skill, args, scene, on_action)
            events.append(("end", skill))
            return result

    tools = RobotTools(scene, TracingBackend())
    execute(COMPOUND, tools)

    for i in range(0, len(events), 2):
        assert events[i][0] == "start"
        assert events[i + 1] == ("end", events[i][1])


def test_actions_stream_as_they_are_emitted(scene, tools):
    seen: list[str] = []
    report = execute(COMPOUND, tools, on_action=lambda action, skill: seen.append(skill))
    assert len(seen) == report.action_count
    assert seen[0] == "approach_wheelchair"
    assert seen[-1] == "detach"


# ---- failure handling ----------------------------------------------------


def test_failure_stops_the_run_and_names_the_step(scene):
    tools = RobotTools(scene, StubBackend(fail_skills=("attach_handle",)))
    report = execute(COMPOUND, tools)

    assert not report.ok
    assert report.failed_step == 2
    assert "attach_handle" in report.failure_reason

    statuses = [s.status for s in report.steps]
    assert statuses[:2] == [StepStatus.SUCCEEDED, StepStatus.SUCCEEDED]
    assert statuses[2] is StepStatus.FAILED
    assert statuses[3:] == [StepStatus.SKIPPED, StepStatus.SKIPPED]


def test_nothing_is_left_attached_after_a_failure(scene):
    """A failed run must not leave the arm welded to the payload."""
    tools = RobotTools(scene, StubBackend(fail_skills=("nav_with_payload",)))
    report = execute(COMPOUND, tools)

    assert not report.ok
    assert report.compensated is True
    assert tools.attached_to is None


def test_no_compensating_detach_when_nothing_is_held(scene):
    tools = RobotTools(scene, StubBackend(fail_skills=("approach_wheelchair",)))
    report = execute(COMPOUND, tools)
    assert not report.ok
    assert report.compensated is False


def test_a_backend_that_raises_is_reported_not_propagated(scene):
    class ExplodingBackend(StubBackend):
        def run_skill(self, skill, args, scene, on_action=None):
            raise RuntimeError("policy crashed")

    tools = RobotTools(scene, ExplodingBackend())
    report = execute([ToolCall("navigate_to", {"location_id": "room_2"})], tools)
    assert not report.ok
    assert "policy crashed" in report.failure_reason
    assert report.steps[0].status is StepStatus.FAILED


# ---- preconditions -------------------------------------------------------


def test_nav_with_payload_is_blocked_without_an_attachment(scene, tools):
    report = execute([ToolCall("nav_with_payload", {"location_id": "room_2"})], tools)
    assert not report.ok
    assert report.steps[0].status is StepStatus.BLOCKED
    assert "no payload is attached" in report.steps[0].detail


def test_plain_navigation_is_blocked_while_a_payload_is_held(scene, tools):
    plan = [
        ToolCall("approach_wheelchair", {"object_id": "wheelchair_01"}),
        ToolCall("align_gripper", {"object_id": "wheelchair_01"}),
        ToolCall("attach_handle", {"object_id": "wheelchair_01"}),
        ToolCall("navigate_to", {"location_id": "room_2"}),  # wrong skill while loaded
    ]
    report = execute(plan, tools)
    assert not report.ok
    assert report.steps[3].status is StepStatus.BLOCKED
    assert report.compensated is True


def test_double_attach_is_blocked(scene, tools):
    plan = [
        ToolCall("attach_handle", {"object_id": "wheelchair_01"}),
        ToolCall("attach_handle", {"object_id": "cart_01"}),
    ]
    report = execute(plan, tools)
    assert report.steps[0].status is StepStatus.SUCCEEDED
    assert report.steps[1].status is StepStatus.BLOCKED


# ---- timeout -------------------------------------------------------------


def test_a_hung_skill_times_out_instead_of_hanging_the_demo(scene):
    class HangingBackend(StubBackend):
        def run_skill(self, skill, args, scene, on_action=None):
            time.sleep(5.0)
            return SkillResult(True, "eventually")

    tools = RobotTools(scene, HangingBackend())
    started = time.monotonic()
    report = execute(
        [ToolCall("navigate_to", {"location_id": "room_2"})], tools, timeout_s=0.2
    )
    elapsed = time.monotonic() - started

    assert not report.ok
    assert report.steps[0].status is StepStatus.TIMED_OUT
    assert elapsed < 3.0, "execute() must return on timeout, not wait for the skill"
