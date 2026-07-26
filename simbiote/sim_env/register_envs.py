"""Registers Simbiote's tasks as Gymnasium envs (spec §5.4).

    from simbiote.sim_env.register_envs import register, TASK_IDS
    register()
    env = gymnasium.make(TASK_IDS["nav"])

Called once at process start by `train_nav.py`/`train_grasp.py`/`play.py`.
Idempotent — safe to call multiple times (e.g. from tests).
"""

from __future__ import annotations

TASK_IDS = {
    "nav": "Simbiote-Nav-v0",
    "grasp": "Simbiote-Grasp-v0",
    "wheelchair": "Simbiote-Wheelchair-v0",
}

def register() -> None:
    """Idempotent: skips any id already present in gymnasium's registry, so
    it's safe to call repeatedly across a test session or notebook re-runs."""
    import gymnasium as gym

    specs = {
        TASK_IDS["nav"]: "simbiote.sim_env.nav_task:NavEnv",
        TASK_IDS["grasp"]: "simbiote.sim_env.grasp_task:GraspEnv",
        TASK_IDS["wheelchair"]: "simbiote.sim_env.wheelchair_task:WheelchairEnv",
    }
    for env_id, entry_point in specs.items():
        if env_id not in gym.envs.registry:
            gym.register(id=env_id, entry_point=entry_point, max_episode_steps=None)


def make_env(task: str, **kwargs):
    """Convenience helper: `make_env("nav", gui=True)`."""
    register()
    import gymnasium as gym

    if task not in TASK_IDS:
        raise ValueError(f"Unknown task '{task}'. Known: {list(TASK_IDS)}")
    return gym.make(TASK_IDS[task], **kwargs)
