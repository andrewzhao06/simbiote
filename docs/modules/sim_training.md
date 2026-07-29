# Physics, simulation, and training

This module covers `simbiote/sim_env/`, `simbiote/training/`, and the shared
`simbiote/robot_iface/` schemas that the other modules build on. Tests live in
the corresponding domain folders under `tests/` and run as part of the root `pytest` suite
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

`tests/` covers the sim environments, policy network, BC/PPO training
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

## Running the RL policies (PyBullet tier)

```bash
source scripts/gb10/env.gb10.sh

# navigation
$FF_PY -m simbiote.training.train_nav --num_envs 8 --timesteps 80000 \
    --out checkpoints/nav_ppo.pt
$FF_PY -m simbiote.training.play --checkpoint checkpoints/nav_ppo.pt \
    --task nav --episodes 30          # --gui to watch

# grasp: BC from scripted demos first, PPO cannot find this one cold
$FF_PY scripts/training/oracle_demos.py --episodes 60 --out checkpoints/grasp_bc.pt
$FF_PY -m simbiote.training.play --checkpoint checkpoints/grasp_bc.pt \
    --task grasp --episodes 30
```

**Grasp: use the BC checkpoint.** PPO from a cold start never finds the lift
(peaks ~4%), and PPO fine-tuning *on top of* BC actively destroys it — measured
going 0.19 → 0.26 → 0.00 by update 8 and never recovering. BC alone scores 71%,
matching the scripted demonstrator it learned from. Fine-tuning needs a lower
learning rate or a KL penalty against the BC policy before it is worth running.

Both trainers now keep the **best** checkpoint rather than the last, scored on
success rate (falling back to mean return before any episode finishes).
Previously a run that peaked and then collapsed silently saved the collapsed
weights while the log still showed it had once worked.

### Both tasks were unwinnable; that is fixed

Worth knowing, because in each case the training curve looked plausible while
the objective was unreachable:

- **NavEnv** placed obstacles and goals by unconstrained uniform draw. With the
  robot issuing a zero action, 23% of episodes collided on step 1 (nearest
  obstacle spawned 0.04 m away) and 3% started inside the goal — about a
  quarter of any measured success rate was decided at reset. `reset()` now
  rejection-samples against `spawn_clearance`, `obstacle_spacing`,
  `min_goal_distance` and `goal_clearance`. This is also the most likely cause
  of `test_ppo_improves_nav_success_rate_over_random_baseline` being flaky.
- **The stand-in arm was planar.** `shoulder_joint` and `elbow_joint` both had
  axis `(0 1 0)` (the wrist's Z roll is at the chain's end and only spins the
  gripper), so the EE's world y was pinned at exactly 0.000 while GraspEnv
  spawns objects at y ∈ [-0.2, 0.2]. Added `shoulder_yaw_joint`.
- **`GRASP_DIST_THRESHOLD` was tighter than the collision geometry allows.**
  0.03 m object half-extent plus a 0.04×0.08×0.04 `ee_link` puts the floor on
  centre-to-centre distance around 0.05–0.07 m; the threshold was 0.06 m. A
  scripted oracle pressing into the object bottomed out at 0.061 m and never
  triggered a grasp. Now 0.10 m.
- **IK solved cold every step.** With the added yaw the arm is redundant, so
  the solver hopped between branches and the arm thrashed. Now seeded with the
  current configuration via the null-space form (limits + ranges + rest poses
  together — passing `currentPositions` alone is silently ignored).

`tests/sim_env/test_env_solvability.py` guards all of this, including a scripted
oracle that must actually complete the grasp. A success rate converging to zero
is indistinguishable from "needs more timesteps" unless something asserts the
task is winnable at all.

## Isaac Sim bring-up — `check_isaac_hospital.py`

Proves the substrate before any of the above: hospital scene loads with real
colliders, the Ridgeback+Franka articulation initialises, and both the base and
the arm track commands under PhysX 5.

```bash
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/isaac/check_isaac_hospital.py          # --gui to watch
```

Expect it to take several minutes: PhysX cooks collision meshes for the whole
76 x 42 m environment on the first run.

To look at it rather than assert on it:

```bash
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/isaac/view_isaac_hospital.py           # --play to run physics
```

Same asset resolution, anchor and spawn point as the check, so what you see is
what the checks ran against. Give the viewport ~90 s after the window appears —
RTX streams the hospital's materials in gradually and the view is blank white
until it finishes, which looks like a broken camera but is not.

### Things the asset does that will bite you

**You must pin the `world` link yourself, or the robot never moves.** The chain
is `world -> dummy_base_x -> dummy_base_y -> base_link`, and the asset ships no
joint anchoring `world` to the static frame. Left alone the articulation is
floating-base, so driving `dummy_base_prismatic_x_joint` to +1 m slides the
*anchor* to −1 m and leaves `base_link` and the whole arm exactly where they
were. Every command reports success and the robot goes nowhere — measured:

```
              unanchored              anchored
world         delta [-1, 0, 0]        delta [0, 0, 0]
dummy_base_x  delta [ 0, 0, 0]        delta [1, 0, 0]
base_link     delta [ 0, 0, 0]        delta [1, 0, 0]
panda_link0   delta [ 0, 0, 0]        delta [1, 0, 0]
```

The fix is one prim:

```python
anchor = UsdPhysics.FixedJoint.Define(stage, "/base_anchor")
anchor.CreateBody1Rel().SetTargets([f"{robot_path}/world"])
```

Follow-on consequences once anchored:

- The robot cannot tip over or fall through the floor. Reward terms that
  penalise toppling are dead code here.
- `articulation.get_world_pose()` returns the **anchor**, which never moves.
  Read the base pose from `base_link` instead, or you will train a nav policy
  against a position that is always the spawn point.
- The base is a 3-DOF planar joint (prismatic x/y + revolute z), not four
  mecanum wheels — nav actions map straight to base velocities, no wheel IK.
- **Do not hardcode DOF indices.** Adding the anchor reordered them
  (`dummy_base_prismatic_x_joint` moved from index 9 to 0). Always resolve via
  `get_dof_index(name)`.

**The base position drive is stiff enough to bulldoze through walls.**
Measured: commanded 10 m along X from a clear spawn, the base travelled the
full 10.00 m straight through the building. At ~1e7 stiffness the drive simply
overpowers contact. So **the physics will not enforce collision avoidance for
you** — it has to come from the nav reward, or you lower the base drive
stiffness / switch to velocity control. `check_isaac_hospital.py` runs this
test every time and reports which way it went.

**Spawn points need a clearance test, not just a floor tile.** The hospital has
785 prims sitting in the robot's height band (z 0.15–1.6 m). Spawning at the
centre of a floor tile put the robot inside a prop: contact jammed the base
joints at −0.31 m before the first command, then *every* drive test failed —
base and arm alike — which reads exactly like broken drives rather than a bad
spawn. Intersect candidate points against those prim bounds and require ~1.2 m
of clearance. `(7.81, 8.25)` has 2.75 m and is the default in the script.

**Assets resolve through Isaac Sim's own asset root**, not hardcoded paths.
`check_isaac_hospital.py` calls `get_assets_root_path()` and composes
`/Isaac/Environments/Hospital/hospital.usd` and
`/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd` onto it.

Two things about that root on this box:

- Out of the box it is an **S3 URL**, so every asset streams over the network —
  which the event is explicitly air-gapped against. Setting it in
  `user.config.json` is *not* enough: `isaacsim.storage.native`'s
  `config/extension.toml` declares
  `persistent.isaac.asset_root.default` as an extension setting, and that wins
  over the user config. It has been repointed there to
  `/home/dell/AI/assets/isaac-5.1` (original kept as `extension.toml.bak`).
  Since that is a vendored NVIDIA file, an Isaac Sim reinstall will revert it —
  `check_isaac_hospital.py` therefore also detects a remote root at runtime,
  falls back to the local pack, and says so.
- The pack was a **partial download**: `Isaac/Environments/Hospital` and
  `Isaac/Robots/Clearpath` only, with no `/NVIDIA` tree, so the hospital's dome
  light silently had no sky texture. The missing HDR has been fetched. The
  script now resolves every asset-valued attribute on the stage and fails if
  any is unresolved, rather than letting it degrade quietly.

Never point at a *symlinked copy* of `hospital.usd`: USD resolves the scene's
relative `./Props` and `./Materials` references against the link's own
directory, so a bare file symlink loads a hospital with no walls and the robot
drives straight through the building. The check counts `Geo_` prims
(expect ~1064) as the canary.

**Coordinate conventions differ between the two environments.** `hospital.usd`
is Z-up, so PhysX's default -Z gravity is already right. Teammate 1's scanned
scenes are Y-up per the Step 2 contract, and there Isaac Sim silently gives you
-Z gravity unless you re-assert the up axis after `initialize_physics()` — see
`docs/modules/mapper.md`.

## Navigating the real hospital (Isaac tier)

`simbiote/sim_env/isaac_nav.py` drives the trained nav policy through
`hospital.usd` at true scale, and `simbiote/sim_env/hospital_map.py` is the
occupancy grid + A* that makes it possible.

```bash
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/isaac/eval_hospital_nav.py --checkpoint checkpoints/nav_bc.pt   # --gui
/home/dell/IsaacSim/_build/linux-aarch64/release/python.sh \
    scripts/gb10/isaac/eval_hospital_nav.py --controller pursuit   # reference, no policy
```

**Measured: 16/20 ordered location pairs, mean 0.96x path efficiency** (i.e.
the driven distance is within 4% of the planned path), including a 75 m
traversal end to end. The straight-at-the-carrot reference controller also
scores 16/20, so the policy is not yet *better* than a dumb controller here —
it is equal to it, and both are limited by the same tight corridor.

`nav_bc.pt` is the checkpoint to use. Re-running PPO on top of it
(`checkpoints/nav_hospital.pt`, 600k steps, 16 envs) left the arena success
rate unchanged at 53% — the same "PPO on top of BC doesn't help" result already
recorded above for grasp.

### Why the 4 x 4 m policy can drive a 76 x 42 m building

It isn't asked to. `HospitalMap` A*s the long-range route; the policy is fed a
carrot 1.6 m ahead on that route, with the robot placed at the *local* origin
and the goal delta clipped to 1.8 m, so every observation looks like the arena
it trained in. Its output is a world-frame velocity, which is why the local
frame is not rotated.

### Things that cost hours, recorded so they don't again

- **Every `Geo_` prim in hospital.usd is `instanceable = true`.** A plain
  `stage.Traverse()` finds 23 meshes in the entire building instead of 2059,
  and the occupancy grid comes out empty — which looks exactly like a
  successful build. Use `Usd.TraverseInstanceProxies`.
- **`S_WetFloorSign` matches a `"Floor" in name` filter.** It is a 0.79 m tall
  obstacle in the middle of a corridor, and skipping it as floor geometry left
  a hole the planner routed straight through. The base then jammed on it in
  what the grid reported as 2.5 m of open space. The floor test now also
  requires the prim to be flat and on the ground.
- **Integrating the base position target off the *measured* joint position
  cannot work.** The drives are position PD, so commanded-minus-actual is the
  entire force budget; re-reading the encoder each tick caps the error at one
  tick of motion and the base stalls against any contact at all. Hold the
  target internally and clamp how far it may lead (`max_target_lead`, 0.35 m).
  That clamp is also what keeps the 1e7 stiffness from bulldozing walls.
- **Inflate the grid by the robot's real half-diagonal (0.62 m), not less.**
  At 0.45 m, A* returned paths through gaps the base cannot fit and traversals
  clipped corners at 0.00-0.22 m measured clearance. The building is open-plan
  enough that 0.65 m costs nothing: all 30 pairs still plan at 0.75 m.
- **The policy has no behaviour for walls.** `NavEnv` shows it three *point*
  obstacles in an open box and never puts a wall in the observation, so in a
  corridor it drifts into the surface and grinds along it. A cross-track term
  pulling the base back onto the planned line (`cross_track_gain`) took the
  measured result from 5/20 to 14/20; nothing else moved the number as much.
- **`teleport()` must set joint state, not a position target.** Driving to a
  40 m target inside the settle window is 16 m/s, the base never arrives, and
  the next run starts wherever it stopped — which shows up as a 73 m traversal
  reporting a 2.9 m path.
