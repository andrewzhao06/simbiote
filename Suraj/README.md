# Suraj — physics, sim, and training

Owns everything under `simbiote/sim_env/`, `simbiote/training/`, and the
shared `simbiote/robot_iface/` schemas that the other two roles build on top
of. Tests live in `Suraj/tests/` and run as part of the root `pytest` suite
(see `pyproject.toml`'s `testpaths`).

## What's here

- `simbiote/sim_env/` — PyBullet Gymnasium environments:
  - `nav_task.py` — `NavEnv`, base navigation with obstacle/wall collision
    penalties.
  - `grasp_task.py` — `GraspEnv`, arm reach-and-grasp with a workspace-radius
    limit measured relative to the arm's base link, not the world origin.
  - `pybullet_scene.py` — shared scene/robot loading utilities
    (`RobotHandles`, arena builders).
- `simbiote/training/` — PPO + Behavioral Cloning:
  - `policy_net.py` — `ActorCriticMLP`, the actor-critic network shared by BC
    pretraining and PPO fine-tuning. `act()` clamps to the action bounds
    *before* computing `log_prob`, so PPO's importance-sampling ratio is
    evaluated at the same point on both the old and new policy.
  - `bc_pretrain.py` — behavioral cloning from logged demonstrations.
    Tolerates `RobotAction.arm_target_pose is None` (nav-only actions) by
    treating the arm delta as zero.
  - `train_nav.py` / `train_grasp.py` — PPO training entry points, each
    parallel environment seeded uniquely (`base_seed + env_index`) so
    parallel rollouts aren't identical.
  - `retrain.py` — `ingest_demo()` / `finetune_policy()`, the loop that turns
    a logged demo (teleop or agentic) into a BC + PPO fine-tune. Sanitizes
    `trajectory.task` / `session_id` before using them as path components.
- `simbiote/robot_iface/` — canonical cross-team schemas:
  - `actions.py` — `Pose` (tuple-based position/orientation, with
    `to_dict`/`from_dict` and flat `x,y,z,qx,qy,qz,qw` accessors) and
    `RobotAction` (`arm_target_pose: Optional[Pose]`, `to_vector()`
    substitutes a neutral pose when `None` so ML models always see a
    fixed-length vector).
  - `trajectory.py` — `Trajectory` / `TrajectoryStep`, with per-step
    `observation`/`reward` (for BC), `source` (`"teleop"` or `"agentic"`),
    and optional `skill`/`ok` (for agentic runs).
  - `skills.py` — `navigate_to()` / `pick_up()`, thin wrappers that load a
    trained checkpoint and run a short PyBullet rollout to completion; this
    is what Andrew's `CheckpointBackend` calls into.

## Tests

`Suraj/tests/` covers the sim environments, policy network, BC/PPO training
loops, the `RobotAction`/`Pose`/`Trajectory` schemas, and `demo_logger.py`'s
in-memory + JSONL-on-disk session handling. Tests that need real physics are
marked with the shared `require_pybullet` fixture (root `conftest.py`) and
skip cleanly on Windows, where `pybullet` has no PyPI wheel.

## GB10 next steps

- Swap the PyBullet backend for Isaac Sim / PhysX 5 behind the same
  `sim_env` interface, so `training/` doesn't change.
- Export trained checkpoints for `skills.py` to load on the GB10.
- Verify parallel-env seeding and workspace-radius clipping still hold at
  Isaac Sim's higher fidelity.
