import numpy as np
import torch

from simbiote.robot_iface.actions import GripperState, Pose, RobotAction
from simbiote.robot_iface.trajectory import Trajectory, TrajectoryStep, make_toy_trajectory
from simbiote.sim_env.grasp_task import MAX_EE_STEP
from simbiote.training.bc_pretrain import task_action_vector, train_bc, trajectories_to_dataset
from simbiote.training.policy_net import ActorCriticMLP, PolicyMeta


def test_trajectories_to_dataset_nav_shapes():
    trajs = [
        make_toy_trajectory(f"s{i}", obs_dim=7, length=10, task="nav", seed=i) for i in range(3)
    ]
    obs_arr, act_arr = trajectories_to_dataset(trajs, task="nav")
    assert obs_arr.shape == (30, 7)
    assert act_arr.shape == (30, 3)  # base_velocity is 3-dim


def _grasp_trajectory(session_id: str, length: int = 6) -> Trajectory:
    steps = []
    for t in range(length):
        action = RobotAction(
            arm_target_pose=Pose(position=(0.1 * t, 0.0, 0.2)),
            gripper_state=GripperState.CLOSED if t > 3 else GripperState.OPEN,
        )
        steps.append(
            TrajectoryStep(
                timestamp=float(t), observation=[0.0] * 17, action=action, source="teleop"
            )
        )
    return Trajectory(session_id=session_id, source="teleop", task="grasp", steps=steps)


def test_task_action_vector_grasp_delta():
    prev = None
    action = RobotAction(
        arm_target_pose=Pose(position=(0.1, 0.0, 0.2)), gripper_state=GripperState.OPEN
    )
    vec, new_target = task_action_vector(prev, action)
    assert vec.shape == (4,)
    assert (vec[:3] == 0).all()  # first step: no previous target, delta is zero
    assert vec[3] == -1.0  # open -> -1

    # Requested jump (0.1) exceeds GraspEnv's per-step limit (`MAX_EE_STEP`)
    # -- task_action_vector() must clip to it, and track the resulting
    # (clipped) running target rather than the raw teleop pose.
    action2 = RobotAction(
        arm_target_pose=Pose(position=(0.2, 0.0, 0.2)), gripper_state=GripperState.CLOSED
    )
    vec2, new_target2 = task_action_vector(new_target, action2)
    assert abs(vec2[0] - MAX_EE_STEP) < 1e-6
    assert vec2[3] == 1.0  # closed -> 1
    assert (
        abs(new_target2[0] - (new_target[0] + MAX_EE_STEP)) < 1e-6
    )  # running target reflects the clipped delta, not the raw pose


def test_task_action_vector_handles_no_arm_command():
    """A step logged with `arm_target_pose=None` (e.g. a navigate_to leg
    inside an otherwise grasp-relevant agentic run) must not crash, and must
    not move the running EE target."""
    prev = np.array([0.2, 0.0, 0.3], dtype=np.float32)
    action = RobotAction(
        base_velocity=(0.5, 0.0, 0.0), arm_target_pose=None, gripper_state=GripperState.OPEN
    )
    vec, new_target = task_action_vector(prev, action)
    assert vec.shape == (4,)
    assert (vec[:3] == 0).all()
    assert (new_target == prev).all()


def test_trajectories_to_dataset_grasp_shapes():
    trajs = [_grasp_trajectory("g1"), _grasp_trajectory("g2")]
    obs_arr, act_arr = trajectories_to_dataset(trajs, task="grasp")
    assert obs_arr.shape == (12, 17)
    assert act_arr.shape == (12, 4)


def test_train_bc_produces_loadable_checkpoint(tmp_path):
    trajs = [
        make_toy_trajectory(f"s{i}", obs_dim=5, length=20, task="nav", seed=i) for i in range(4)
    ]
    out_path = tmp_path / "nav_bc.pt"
    result_path = train_bc(trajs, task="nav", epochs=3, batch_size=16, out_path=out_path)

    assert result_path == out_path
    assert out_path.exists()
    loaded = ActorCriticMLP.load(out_path)
    assert loaded.meta.obs_dim == 5
    assert loaded.meta.act_dim == 3


def test_train_bc_fits_a_simple_deterministic_target(tmp_path):
    """A stronger check than 'doesn't crash': BC should actually reduce
    regression error on an easy, noise-free (obs -> action) mapping."""
    import numpy as np

    obs_dim = 3
    trajs = []
    for i in range(6):
        steps = []
        for t in range(15):
            obs = list(np.random.RandomState(i * 100 + t).uniform(-1, 1, size=obs_dim))
            # deterministic linear target the MLP can actually learn
            target = [obs[0] + obs[1], obs[2] * 0.5]
            action = RobotAction(base_velocity=(target[0], target[1], 0.0))
            steps.append(
                TrajectoryStep(timestamp=float(t), observation=obs, action=action, source="teleop")
            )
        trajs.append(Trajectory(session_id=f"lin{i}", source="teleop", task="nav", steps=steps))

    out_path = tmp_path / "linear_bc.pt"
    train_bc(trajs, task="nav", epochs=200, lr=1e-2, batch_size=32, out_path=out_path, seed=1)
    model = ActorCriticMLP.load(out_path)

    obs_arr, act_arr = trajectories_to_dataset(trajs, task="nav")
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
    act_t = torch.as_tensor(act_arr[:, :2], dtype=torch.float32)  # only the 2 meaningful dims
    with torch.no_grad():
        pred = model.actor_mean(obs_t)[:, :2]
    mse = ((pred - act_t) ** 2).mean().item()
    assert mse < 0.05, f"BC failed to fit a simple deterministic target (mse={mse})"


def test_train_bc_warm_starts_existing_policy(tmp_path):
    meta = PolicyMeta(obs_dim=5, act_dim=3, hidden_sizes=(16, 16))
    seed_policy = ActorCriticMLP(meta)
    before = seed_policy.actor_mean[0].weight.clone()

    trajs = [make_toy_trajectory("s", obs_dim=5, length=20, task="nav")]
    train_bc(trajs, policy_net=seed_policy, task="nav", epochs=3, out_path=tmp_path / "warm.pt")

    after = seed_policy.actor_mean[0].weight
    assert not torch.allclose(before, after), (
        "train_bc should update the passed-in policy_net in place"
    )
