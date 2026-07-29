import pytest

from simbiote import demo_logger
from simbiote.robot_iface.actions import GripperState, RobotAction


@pytest.fixture(autouse=True)
def _clean_sessions(tmp_path, monkeypatch):
    # Isolate the on-disk JSONL persistence to a per-test tmp dir, so the
    # suite never writes into the real repo's ./stage or /var/simbiote/stage.
    monkeypatch.setenv("SIMBIOTE_STAGE", str(tmp_path / "stage"))
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None
    yield
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None


def test_start_log_export_session():
    demo_logger.start_session("sess-1", source="teleop", task="nav")
    demo_logger.log_action(
        RobotAction(base_velocity=(1, 0, 0)), source="teleop", observation=[0.1, 0.2]
    )
    demo_logger.log_action(
        RobotAction(base_velocity=(0, 1, 0)), source="teleop", observation=[0.3, 0.4]
    )

    traj = demo_logger.export_trajectory("sess-1")
    assert traj.session_id == "sess-1"
    assert len(traj) == 2
    assert traj.steps[0].observation == [0.1, 0.2]
    assert traj.steps[1].action.base_velocity == (0, 1, 0)


def test_log_action_without_active_session_auto_creates_one():
    """Teleop/agentic call sites don't always call start_session() first --
    log_action() creates one implicitly rather than raising."""
    step = demo_logger.log_action(RobotAction(), source="teleop")
    assert step.source == "teleop"
    traj = demo_logger.export_trajectory()
    assert len(traj) == 1


def test_log_action_source_mismatch_raises():
    demo_logger.start_session("sess-2", source="teleop", task="grasp")
    with pytest.raises(ValueError):
        demo_logger.log_action(RobotAction(), source="agentic", session_id="sess-2")


def test_end_session_removes_it_from_memory():
    demo_logger.start_session("sess-3", source="agentic", task="nav")
    demo_logger.log_action(RobotAction(), source="agentic")
    traj = demo_logger.end_session("sess-3")
    assert traj.session_id == "sess-3"
    assert "sess-3" not in demo_logger._active_sessions

    # log_action() persists every step to disk as it's logged (JSONL), so a
    # session that has ended in-memory still reads back cleanly from disk --
    # durability across a crashed/restarted process, not just within one run.
    recovered = demo_logger.export_trajectory("sess-3")
    assert recovered.session_id == "sess-3"
    assert len(recovered) == 1


def test_multiple_concurrent_sessions():
    demo_logger.start_session("a", source="teleop", task="nav")
    demo_logger.log_action(RobotAction(base_velocity=(1, 0, 0)), source="teleop", session_id="a")
    demo_logger.start_session("b", source="agentic", task="grasp")
    demo_logger.log_action(RobotAction(base_velocity=(0, 1, 0)), source="agentic", session_id="b")

    traj_a = demo_logger.export_trajectory("a")
    traj_b = demo_logger.export_trajectory("b")
    assert traj_a.steps[0].action.base_velocity == (1, 0, 0)
    assert traj_b.steps[0].action.base_velocity == (0, 1, 0)
    assert traj_a.source == "teleop"
    assert traj_b.source == "agentic"


def test_log_action_persists_to_jsonl_on_disk(tmp_path):
    stage = tmp_path / "custom_stage"
    demo_logger.start_session("sess-disk", source="teleop", task="nav")
    demo_logger.log_action(
        RobotAction(base_velocity=(1, 0, 0), gripper_state=GripperState.CLOSED),
        source="teleop",
        stage=stage,
    )
    path = demo_logger.session_path("sess-disk", stage)
    assert path.exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


def test_export_trajectory_recovers_from_disk_when_not_in_memory(tmp_path):
    """Simulates a crashed process: the in-memory session is gone, but the
    JSONL file on disk still reads back cleanly."""
    stage = tmp_path / "stage"
    demo_logger.start_session("sess-crash", source="agentic", task="nav")
    demo_logger.log_action(RobotAction(base_velocity=(1, 0, 0)), source="agentic", stage=stage)
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None

    recovered = demo_logger.export_trajectory("sess-crash", stage=stage)
    assert recovered.session_id == "sess-crash"
    assert len(recovered) == 1
    assert recovered.steps[0].source == "agentic"


def test_export_trajectory_unknown_session_raises(tmp_path):
    with pytest.raises(KeyError):
        demo_logger.export_trajectory("does-not-exist", stage=tmp_path / "stage")
