"""Verification 7 — the acceptance tests from 6b.5, end to end.

A trajectory file comes out and reads back cleanly, which is the contract Step
2's ``ingest_demo()`` codes against.
"""

from __future__ import annotations

import json

from factoryflow import demo_logger
from factoryflow.agentic.agentic_session import main, run_session
from factoryflow.agentic.robot_tools import StubBackend


def test_simple_acceptance_case(scene, fake_llm, stage):
    result = run_session(
        "pick up the tray in the supply room", scene, fake_llm, StubBackend(), stage=stage
    )

    assert result.ok
    assert [c.tool for c in result.calls] == ["navigate_to", "pick_up"]
    assert result.trajectory is not None and len(result.trajectory) > 0
    assert result.trajectory.skills() == ["navigate_to", "pick_up"]
    assert all(step.source == "agentic" for step in result.trajectory)


def test_compound_acceptance_case(scene, fake_llm, stage):
    result = run_session("move the wheelchair to Room 2", scene, fake_llm, StubBackend(), stage=stage)

    assert result.ok
    assert result.trajectory.skills() == [
        "approach_wheelchair", "align_gripper", "attach_handle",
        "nav_with_payload", "detach",
    ]


def test_trajectory_reads_back_identically(scene, fake_llm, stage):
    """export_trajectory must reproduce exactly what was logged — this is the
    interface Step 2 ingests."""
    result = run_session("pick up the tray in the supply room", scene, fake_llm, StubBackend(), stage=stage)
    reloaded = demo_logger.export_trajectory(result.session_id, stage=stage)

    assert len(reloaded) == len(result.trajectory)
    assert [s.action for s in reloaded] == [s.action for s in result.trajectory]
    assert [s.skill for s in reloaded] == [s.skill for s in result.trajectory]


def test_trajectory_file_is_one_json_object_per_line(scene, fake_llm, stage):
    """JSONL, so a session killed mid-run still leaves a readable partial."""
    result = run_session("go to Room 2", scene, fake_llm, StubBackend(), stage=stage)
    path = demo_logger.session_path(result.session_id, stage)

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(result.trajectory)
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"t", "source", "skill", "action", "ok"}


def test_execution_report_sidecar_is_written(scene, fake_llm, stage):
    result = run_session("move the wheelchair to Room 2", scene, fake_llm, StubBackend(), stage=stage)
    sidecar = demo_logger.session_path(result.session_id, stage).with_suffix(".report.json")

    report = json.loads(sidecar.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["instruction"] == "move the wheelchair to Room 2"
    assert [s["tool"] for s in report["steps"]] == [c["tool"] for c in report["plan"]]


def test_failed_run_still_logs_the_actions_it_managed(scene, fake_llm, stage):
    """Partial demonstrations are still useful training data — and a run that
    fails must not silently produce an empty file."""
    result = run_session(
        "move the wheelchair to Room 2",
        scene,
        fake_llm,
        StubBackend(fail_skills=("nav_with_payload",)),
        stage=stage,
    )

    assert not result.ok
    assert len(result.trajectory) > 0
    assert "nav_with_payload" not in result.trajectory.skills()
    assert "detach" in result.trajectory.skills()  # the compensating release


# ---- CLI -----------------------------------------------------------------


def test_cli_success_path(stage, capsys):
    code = main(["pick up the tray in the supply room", "--stage", str(stage)])
    out = capsys.readouterr().out
    assert code == 0
    assert "result: SUCCESS" in out
    assert "navigate_to(location_id=supply_room)" in out


def test_cli_reports_a_failed_skill(stage, capsys):
    code = main(
        ["move the wheelchair to Room 2", "--stage", str(stage), "--fail-skill", "attach_handle"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "result: FAILED" in captured.err


def test_cli_reports_an_unparseable_instruction(stage, capsys):
    code = main(["recalibrate the flux capacitor", "--stage", str(stage)])
    assert code == 2
    assert "parse failed" in capsys.readouterr().err
