import pytest

from simbiote import demo_logger
from simbiote.robot_iface.actions import GripperState, RobotAction


@pytest.fixture(autouse=True)
def _clean_sessions():
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None
    yield
    demo_logger._active_sessions.clear()
    demo_logger._current_session_id = None


def test_start_log_export_session():
    demo_logger.start_session("sess-1", source="teleop", task="nav")
    demo_logger.log_action(RobotAction(base_velocity=(1, 0, 0)), source="teleop", observation=[0.1, 0.2])
    demo_logger.log_action(RobotAction(base_velocity=(0, 1, 0)), source="teleop", observation=[0.3, 0.4])

    traj = demo_logger.export_trajectory("sess-1")
    assert traj.session_id == "sess-1"
    assert len(traj) == 2
    assert traj.steps[0].observation == [0.1, 0.2]
    assert traj.steps[1].action.base_velocity == (0, 1, 0)


def test_log_action_without_active_session_raises():
    with pytest.raises(RuntimeError):
        demo_logger.log_action(RobotAction(), source="teleop")


def test_log_action_source_mismatch_raises():
    demo_logger.start_session("sess-2", source="teleop", task="grasp")
    with pytest.raises(ValueError):
        demo_logger.log_action(RobotAction(), source="agentic", session_id="sess-2")


def test_end_session_removes_it():
    demo_logger.start_session("sess-3", source="agentic", task="nav")
    demo_logger.log_action(RobotAction(), source="agentic")
    traj = demo_logger.end_session("sess-3")
    assert traj.session_id == "sess-3"
    with pytest.raises(KeyError):
        demo_logger.export_trajectory("sess-3")


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
