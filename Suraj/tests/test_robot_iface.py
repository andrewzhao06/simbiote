import json

from simbiote.robot_iface.actions import ACTION_VECTOR_DIM, GripperState, Pose, RobotAction
from simbiote.robot_iface.trajectory import Trajectory, TrajectoryStep, load_trajectories, make_toy_trajectory


def test_pose_roundtrip():
    pose = Pose(position=(1.0, 2.0, 3.0), orientation=(0.1, 0.2, 0.3, 0.9))
    restored = Pose.from_tuple(pose.as_tuple())
    assert restored == pose


def test_robot_action_vector_roundtrip():
    action = RobotAction(
        base_velocity=(0.5, -0.2, 0.1),
        arm_target_pose=Pose(position=(0.4, 0.0, 0.3), orientation=(0, 0, 0, 1)),
        gripper_state=GripperState.CLOSED,
    )
    vec = action.to_vector()
    assert len(vec) == ACTION_VECTOR_DIM
    restored = RobotAction.from_vector(vec)
    assert restored.base_velocity == action.base_velocity
    assert restored.gripper_state == GripperState.CLOSED


def test_robot_action_default_is_open_gripper():
    action = RobotAction()
    assert action.gripper_state == GripperState.OPEN
    assert action.arm_target_pose is None
    assert action.to_vector()[-1] == 0.0


def test_robot_action_dict_roundtrip_preserves_none_arm_pose():
    """A nav-only action's `arm_target_pose=None` must survive the trip --
    fabricating a pose here would corrupt bc_pretrain.py's training data."""
    action = RobotAction(base_velocity=(0.6, 0.0, 0.0))
    restored = RobotAction.from_dict(action.to_dict())
    assert restored == action
    assert restored.arm_target_pose is None


def test_pose_from_dict_accepts_flat_xyzq_form():
    """Some scene-graph fixtures use flat x/y/z/qx.. keys rather than the
    canonical nested position/orientation lists."""
    flat = Pose.from_dict({"x": 1.0, "y": 2.0, "z": 3.0, "qw": 1.0})
    assert flat.position == (1.0, 2.0, 3.0)
    assert flat.x == 1.0 and flat.y == 2.0 and flat.z == 3.0


def test_trajectory_step_dict_roundtrip():
    action = RobotAction(base_velocity=(1, 0, 0), gripper_state=GripperState.CLOSED)
    step = TrajectoryStep(timestamp=1.0, observation=[0.1, 0.2], action=action, source="teleop", reward=0.5)
    restored = TrajectoryStep.from_dict(step.to_dict())
    assert restored.observation == step.observation
    assert restored.action.gripper_state == GripperState.CLOSED
    assert restored.reward == 0.5
    assert restored.source == "teleop"


def test_trajectory_step_skill_and_ok_roundtrip():
    action = RobotAction(base_velocity=(0, 0, 0))
    step = TrajectoryStep(timestamp=2.0, action=action, source="agentic", skill="navigate_to", ok=False)
    restored = TrajectoryStep.from_dict(step.to_dict())
    assert restored.skill == "navigate_to"
    assert restored.ok is False


def test_trajectory_save_and_load(tmp_path):
    traj = make_toy_trajectory("session-1", obs_dim=5, length=10, task="nav")
    path = tmp_path / "traj.json"
    traj.save(path)

    loaded = Trajectory.load(path)
    assert loaded.session_id == "session-1"
    assert loaded.task == "nav"
    assert len(loaded) == 10
    assert loaded.observations()[0] == traj.observations()[0]

    # sanity: file is plain readable JSON, not some opaque blob
    raw = json.loads(path.read_text())
    assert raw["session_id"] == "session-1"


def test_load_trajectories_multiple(tmp_path):
    paths = []
    for i in range(3):
        traj = make_toy_trajectory(f"s{i}", obs_dim=4, length=5, seed=i)
        p = tmp_path / f"s{i}.json"
        traj.save(p)
        paths.append(p)

    loaded = load_trajectories(paths)
    assert len(loaded) == 3
    assert all(len(t) == 5 for t in loaded)


def test_trajectory_action_vectors_shape():
    traj = make_toy_trajectory("s", obs_dim=6, length=4)
    vectors = traj.action_vectors()
    assert len(vectors) == 4
    assert all(len(v) == ACTION_VECTOR_DIM for v in vectors)
