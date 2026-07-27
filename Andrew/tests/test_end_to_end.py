"""Verification 7 — parse -> execute -> log, wired together the way the demo
runs it, writing to a real (temp) staging directory rather than mocks.
"""

from __future__ import annotations

import json

from simbiote import demo_logger
from simbiote.agentic.agentic_session import run_session
from simbiote.agentic.llm_backend import FakeBackend, FallbackBackend
from simbiote.agentic.robot_tools import StubBackend
from simbiote.agentic.task_executor import StepStatus


def test_successful_instruction_produces_a_trajectory_and_report(scene, fake_llm, stage):
    result = run_session(
        "pick up the tray in the supply room",
        scene,
        fake_llm,
        StubBackend(),
        session_id="test-session-ok",
        stage=stage,
    )

    assert result.ok
    assert result.error is None
    assert [c.tool for c in result.calls] == ["navigate_to", "pick_up"]
    assert result.report is not None and result.report.ok
    assert result.trajectory is not None
    assert len(result.trajectory) > 0
    assert all(step.source == "agentic" for step in result.trajectory.steps)

    traj_path = demo_logger.session_path("test-session-ok", stage)
    assert traj_path.exists()
    lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(result.trajectory)

    report_path = traj_path.with_suffix(".report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["instruction"] == "pick up the tray in the supply room"


def test_failed_instruction_still_logs_a_partial_trajectory(scene, fake_llm, stage):
    """A demo beat that fails mid-plan must still leave usable log data,
    not nothing."""
    robot = StubBackend(fail_skills=("attach_handle",))
    result = run_session(
        "move the wheelchair to Room 2",
        scene,
        fake_llm,
        robot,
        session_id="test-session-fail",
        stage=stage,
    )

    assert not result.ok
    assert result.error is None  # this failed during execution, not parsing
    assert result.report is not None and not result.report.ok
    assert result.report.failed_step == 2
    assert result.trajectory is not None and len(result.trajectory) > 0

    report_path = demo_logger.session_path("test-session-fail", stage).with_suffix(".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "attach_handle" in report["failure_reason"]


def test_a_parse_failure_short_circuits_before_any_execution(scene, fake_llm, stage):
    result = run_session(
        "recalibrate the flux capacitor",
        scene,
        fake_llm,
        StubBackend(),
        session_id="test-session-parse-fail",
        stage=stage,
    )

    assert not result.ok
    assert result.error is not None
    assert result.report is None
    assert result.calls == []
    # Nothing was logged, since nothing ever executed.
    assert not demo_logger.session_path("test-session-parse-fail", stage).exists()


def test_on_step_callback_observes_every_step_of_a_compound_plan(scene, fake_llm, stage):
    seen: list[StepStatus] = []
    result = run_session(
        "move the wheelchair to Room 2",
        scene,
        fake_llm,
        StubBackend(),
        session_id="test-session-steps",
        stage=stage,
        on_step=lambda step: seen.append(step.status),
    )
    assert result.ok
    assert seen == [StepStatus.SUCCEEDED] * 5


def test_a_degraded_llm_is_reported_in_the_session_result(scene, stage):
    class BrokenPrimary:
        backend_name = "broken"

        def complete(self, system: str, user: str) -> str:
            from simbiote.agentic.llm_backend import LLMError

            raise LLMError("server unreachable")

    llm = FallbackBackend(BrokenPrimary(), FakeBackend(scene))
    result = run_session(
        "go to the nurse station",
        scene,
        llm,
        StubBackend(),
        session_id="test-session-degraded",
        stage=stage,
    )

    assert result.ok
    assert result.degraded is True
    assert result.llm["served_by"] == "fake"

    report_path = demo_logger.session_path("test-session-degraded", stage).with_suffix(".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["llm"]["degraded"] is True


def test_trajectory_read_back_matches_what_was_logged_live(scene, fake_llm, stage):
    """export_trajectory's disk fallback path, exercised via a fresh read
    rather than the in-memory session that run_session already populated."""
    result = run_session(
        "go to the nurse station",
        scene,
        fake_llm,
        StubBackend(),
        session_id="test-session-readback",
        stage=stage,
    )
    demo_logger.end_session("test-session-readback")

    reloaded = demo_logger.export_trajectory("test-session-readback", stage=stage)
    assert len(reloaded) == len(result.trajectory)
    assert reloaded.source == "agentic"
