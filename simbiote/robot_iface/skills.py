"""The skill API Step 2 produces for Steps 3 & 4 — spec §5.7:

    "You produce for Step 3 (teleop) and Step 4 (agentic): exported
    ONNX/TorchScript checkpoints, callable via navigate_to(location_id) /
    pick_up(object_id) -- and, if the wheelchair stretch task is attempted,
    approach_wheelchair() / attach_handle() / nav_with_payload() / detach()."

§5.4's build-spec file tree didn't previously list a home for these callables
(only the raw checkpoints); this module is that home. Step 4's own
`robot_tools.py` (§6b.5) is expected to import straight from here rather
than re-implementing policy loading.

Each call spins up its own short-lived PyBullet scene, runs the
task-appropriate trained policy to completion, and reports a plain
success/failure dict -- exactly the kind of atomic, self-contained result
`task_executor.py`'s FSM (§6b.4) needs per skill. Location/object name ->
world-position resolution is a placeholder registry here; swap
`DEFAULT_LOCATIONS`/`DEFAULT_OBJECTS` for real calls into Step 1's
`scene_query.py` once that scene graph exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from simbiote.sim_env.grasp_attach import GraspConstraint, attach, release
from simbiote.sim_env.grasp_task import GraspEnv
from simbiote.sim_env.nav_task import NavEnv
from simbiote.sim_env.register_envs import register
from simbiote.sim_env.spawn import SpawnedRobot
from simbiote.sim_env.wheelchair_task import WheelchairEnv

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

# Placeholder scene registry -- real values come from Step 1's scene graph
# (`scene_query.py`'s `list_locations()`/`list_objects()`) once it exists.
DEFAULT_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "room_1": (1.2, 0.8),
    "room_2": (-1.2, 0.8),
    "supply_room": (0.0, -1.4),
    "nurses_station": (1.4, -1.2),
}
DEFAULT_OBJECTS: Dict[str, Tuple[float, float, float]] = {
    "tray_1": (0.45, 0.05, 0.25),
    "tray_2": (0.4, -0.15, 0.25),
}


def _load_inference_fn(checkpoint_path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Returns `f(obs_batch) -> action_batch` for either a `.pt`
    (ActorCriticMLP) or `.onnx` (export_policy.py output) checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(checkpoint_path))
        input_name = session.get_inputs()[0].name

        def infer(obs_batch: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: obs_batch.astype(np.float32)})[0]

        return infer

    import torch

    from simbiote.training.policy_net import ActorCriticMLP

    policy = ActorCriticMLP.load(checkpoint_path)
    policy.eval()

    def infer(obs_batch: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        action_t, _, _ = policy.act(obs_t, deterministic=True)
        return action_t.numpy()

    return infer


def _rollout(env, infer_fn: Callable[[np.ndarray], np.ndarray], max_steps: int) -> dict:
    obs, _ = env.reset()
    info: dict = {}
    for _ in range(max_steps):
        action = infer_fn(obs[None, :])[0]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return info


def fit_locations_to_arena(
    locations: Dict[str, Tuple[float, float]],
    room_size: float = 4.0,
    margin: float = 0.6,
) -> Dict[str, Tuple[float, float]]:
    """Map scene-graph world coordinates into the nav arena, preserving layout.

    Step 1's scene graph is in real building coordinates -- the hospital
    fixture spans about 10 x 13 m -- while `NavEnv` is a 4 x 4 m arena with
    walls at +/-1.8 and a policy trained on goals no further than ~2.4 m away.
    Feeding a raw world coordinate through `goal_override` puts the goal
    *outside the walls*, so the robot drives into one and the skill reports a
    collision (measured: nurse_station came out 6.17 m away).

    Scaling is uniform and about the scene's centre, so relative bearings
    survive: the supply room stays diagonally opposite the nurse station. This
    is a stand-in-tier fix -- the Isaac Sim port navigates the real hospital at
    true scale and won't need it.

    `margin` buys clearance from the arena walls. It is generous on purpose:
    at 0.3 the mapped corners landed 0.3 m from a wall and the approach clipped
    it, which showed up as "go to the supply room" failing on collision while
    the centre-ish destinations passed.
    """
    if not locations:
        return {}
    usable = max(room_size / 2.0 - margin - 0.2, 0.1)
    xs = [xy[0] for xy in locations.values()]
    ys = [xy[1] for xy in locations.values()]
    centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_x, half_y = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
    reach = max(half_x, half_y)
    scale = 1.0 if reach <= usable else usable / reach
    return {
        key: ((x - centre_x) * scale, (y - centre_y) * scale)
        for key, (x, y) in locations.items()
    }


def navigate_to(
    location_id: str,
    checkpoint_path: str | Path = CHECKPOINT_DIR / "nav_ppo.pt",
    locations: Dict[str, Tuple[float, float]] = DEFAULT_LOCATIONS,
    gui: bool = False,
    max_steps: int = 300,
    room_size: float = 4.0,
) -> dict:
    """OpenClaw/agentic tool: drive the trained nav policy to `location_id`."""
    if location_id not in locations:
        raise KeyError(f"navigate_to: unknown location_id '{location_id}'. Known: {list(locations)}")
    register()
    infer_fn = _load_inference_fn(checkpoint_path)
    goal = fit_locations_to_arena(locations, room_size=room_size)[location_id]
    env = NavEnv(
        goal_override=goal, gui=gui, max_steps=max_steps, room_size=room_size
    )
    info = _rollout(env, infer_fn, max_steps)
    return {
        "skill": "navigate_to",
        "location_id": location_id,
        "goal_xy": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
        **info,
    }


def pick_up(
    object_id: str,
    checkpoint_path: str | Path = CHECKPOINT_DIR / "grasp_ppo.pt",
    objects: Dict[str, Tuple[float, float, float]] = DEFAULT_OBJECTS,
    gui: bool = False,
    max_steps: int = 200,
) -> dict:
    """OpenClaw/agentic tool: drive the trained grasp policy to pick up `object_id`."""
    if object_id not in objects:
        raise KeyError(f"pick_up: unknown object_id '{object_id}'. Known: {list(objects)}")
    register()
    infer_fn = _load_inference_fn(checkpoint_path)
    env = GraspEnv(object_position_override=objects[object_id], gui=gui, max_steps=max_steps)
    info = _rollout(env, infer_fn, max_steps)
    return {"skill": "pick_up", "object_id": object_id, **info}


# ---------------------------------------------------------------------------
# Wheelchair stretch-task skills (spec §5.5/§6b.4) -- the FSM states beyond
# navigate_to()/pick_up(). approach_wheelchair()/nav_with_payload() reuse
# the same rollout pattern; attach_handle()/detach() are direct
# grasp_attach.py calls against a live SpawnedRobot handle since "attach" and
# "detach" aren't themselves learned behaviors, just the trigger the FSM
# pulls between Align&Grasp and Constrained-Nav (§5.5's diagram).
# ---------------------------------------------------------------------------

def approach_wheelchair(
    location_id: str,
    checkpoint_path: str | Path = CHECKPOINT_DIR / "nav_ppo.pt",
    locations: Dict[str, Tuple[float, float]] = DEFAULT_LOCATIONS,
    gui: bool = False,
    max_steps: int = 300,
) -> dict:
    result = navigate_to(location_id, checkpoint_path=checkpoint_path, locations=locations, gui=gui, max_steps=max_steps)
    return {**result, "skill": "approach_wheelchair"}


def attach_handle(robot: SpawnedRobot, wheelchair_body_id: int) -> GraspConstraint:
    """FSM state 2->3 transition (§6b.4: AttachHandleConstraint). Welds the
    robot's end-effector to the wheelchair at its current relative pose --
    the caller (task_executor.py) is responsible for having already driven
    the EE to the handle via `robot_tools`/IK before calling this."""
    return attach(robot.robot_id, robot.handles.ee_link_index, wheelchair_body_id, physics_client=robot.physics_client)


def nav_with_payload(
    location_id: str,
    checkpoint_path: str | Path = CHECKPOINT_DIR / "wheelchair_ppo.pt",
    locations: Dict[str, Tuple[float, float]] = DEFAULT_LOCATIONS,
    gui: bool = False,
    max_steps: int = 300,
) -> dict:
    """FSM state 3 (§5.5: Co-Nav Policy, Base + Wheelchair)."""
    if location_id not in locations:
        raise KeyError(f"nav_with_payload: unknown location_id '{location_id}'. Known: {list(locations)}")
    register()
    infer_fn = _load_inference_fn(checkpoint_path)
    env = WheelchairEnv(goal_override=locations[location_id], gui=gui, max_steps=max_steps)
    info = _rollout(env, infer_fn, max_steps)
    return {"skill": "nav_with_payload", "location_id": location_id, **info}


def detach(constraint: GraspConstraint) -> None:
    """FSM state 4 (§5.5: Release & Park)."""
    release(constraint)
