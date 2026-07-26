"""The atomic skills — master doc 6b.5.

Each skill is a thin, stateful wrapper over a :class:`RobotBackend`:

* :class:`StubBackend` — tonight. Synthesises a plausible action sequence and
  reports success, so the parser -> FSM -> logger chain is fully exercisable
  without Isaac Sim or any trained policy. Can be told to fail a named skill,
  which is how the executor's failure paths get tested.
* :class:`CheckpointBackend` — tomorrow. Same interface, running inference on
  Teammate 2's exported ONNX/TorchScript checkpoints.

The wheelchair skills are implemented even though Step 2's wheelchair task is a
stretch goal (5.5): 6b.4 is explicit that the task hierarchy is the right shape
for any multi-step instruction regardless, and against the stub they are nearly
free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from factoryflow.agentic.scene_query import SceneGraph
from factoryflow.agentic.tool_schema import TOOL_SPECS, ToolCall
from factoryflow.robot_iface.actions import GripperState, Pose, RobotAction

__all__ = [
    "SkillResult",
    "RobotBackend",
    "StubBackend",
    "CheckpointBackend",
    "RobotTools",
    "ActionSink",
]

#: Called with each action as the skill emits it, so the session can log live
#: rather than in a batch at the end.
ActionSink = Callable[[RobotAction, str], None]


@dataclass
class SkillResult:
    ok: bool
    detail: str
    actions: list[RobotAction] = field(default_factory=list)


@runtime_checkable
class RobotBackend(Protocol):
    """What actually moves the robot."""

    def run_skill(
        self,
        skill: str,
        args: dict[str, Any],
        scene: SceneGraph,
        on_action: ActionSink | None = None,
    ) -> SkillResult: ...


class StubBackend:
    """Deterministic stand-in used until Step 2's checkpoints exist.

    Tracks a crude base position so the emitted trajectories move sensibly
    through the scene instead of being noise — useful when eyeballing a logged
    run. No randomness anywhere, so tests are stable.
    """

    def __init__(
        self,
        *,
        fail_skills: tuple[str, ...] | set[str] = (),
        steps: int = 4,
        start_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.fail_skills = set(fail_skills)
        self.steps = max(1, steps)
        self.base_xy = start_xy
        self._gripper = GripperState.OPEN

    def run_skill(
        self,
        skill: str,
        args: dict[str, Any],
        scene: SceneGraph,
        on_action: ActionSink | None = None,
    ) -> SkillResult:
        if skill in self.fail_skills:
            # Fail before emitting anything, mirroring a policy that rejects
            # its own preconditions rather than one that fails mid-motion.
            return SkillResult(False, f"stub backend configured to fail {skill!r}")

        if skill in ("navigate_to", "nav_with_payload"):
            loc = scene.get_location(args["location_id"])
            if loc is None:
                return SkillResult(False, f"unknown location {args['location_id']!r}")
            actions = self._drive_to(loc.pose, skill, on_action)
            return SkillResult(True, f"arrived at {loc.id}", actions)

        if skill in ("pick_up", "approach_wheelchair", "align_gripper", "attach_handle"):
            obj = scene.get_object(args["object_id"])
            if obj is None:
                return SkillResult(False, f"unknown object {args['object_id']!r}")
            target = obj.handle_pose or obj.pose

            if skill == "approach_wheelchair":
                actions = self._drive_to(obj.pose, skill, on_action)
                return SkillResult(True, f"in grasp range of {obj.id}", actions)

            if skill == "align_gripper":
                # Arm moves to the pre-grasp standoff and stops. No constraint
                # is formed here, so a bad alignment fails cheaply.
                actions = self._align(target, skill, on_action)
                return SkillResult(True, f"gripper aligned to {obj.id}", actions)

            actions = self._grasp(target, skill, on_action)
            verb = "attached to" if skill == "attach_handle" else "grasped"
            return SkillResult(True, f"{verb} {obj.id}", actions)

        if skill == "detach":
            actions = self._release(skill, on_action)
            return SkillResult(True, "constraint released", actions)

        return SkillResult(False, f"stub backend has no implementation for {skill!r}")

    # ---- action synthesis -------------------------------------------------

    def _emit(
        self, action: RobotAction, skill: str, sink: ActionSink | None, out: list[RobotAction]
    ) -> None:
        out.append(action)
        if sink is not None:
            sink(action, skill)

    def _drive_to(
        self, target: Pose, skill: str, sink: ActionSink | None
    ) -> list[RobotAction]:
        actions: list[RobotAction] = []
        for i in range(self.steps):
            dx = target.x - self.base_xy[0]
            dy = target.y - self.base_xy[1]
            dist = math.hypot(dx, dy)
            remaining = self.steps - i
            if dist < 1e-9:
                vx = vy = 0.0
            else:
                # Constant-ish cruise velocity along the bearing to the goal.
                speed = min(0.6, dist)
                vx, vy = speed * dx / dist, speed * dy / dist
            self.base_xy = (
                self.base_xy[0] + dx / remaining,
                self.base_xy[1] + dy / remaining,
            )
            self._emit(
                RobotAction(
                    base_velocity=(round(vx, 4), round(vy, 4), 0.0),
                    arm_target_pose=None,
                    gripper_state=self._gripper,
                ),
                skill,
                sink,
                actions,
            )
        return actions

    @staticmethod
    def _standoff(target: Pose) -> Pose:
        return Pose(
            target.x, target.y, target.z + 0.15, target.qx, target.qy, target.qz, target.qw
        )

    def _align(self, target: Pose, skill: str, sink: ActionSink | None) -> list[RobotAction]:
        actions: list[RobotAction] = []
        self._emit(
            RobotAction(arm_target_pose=self._standoff(target), gripper_state=GripperState.OPEN),
            skill,
            sink,
            actions,
        )
        return actions

    def _grasp(self, target: Pose, skill: str, sink: ActionSink | None) -> list[RobotAction]:
        actions: list[RobotAction] = []
        # Pre-grasp standoff, then the grasp pose, then close.
        self._emit(
            RobotAction(arm_target_pose=self._standoff(target), gripper_state=GripperState.OPEN),
            skill,
            sink,
            actions,
        )
        self._emit(
            RobotAction(arm_target_pose=target, gripper_state=GripperState.OPEN),
            skill,
            sink,
            actions,
        )
        self._gripper = GripperState.CLOSED
        self._emit(
            RobotAction(arm_target_pose=target, gripper_state=GripperState.CLOSED),
            skill,
            sink,
            actions,
        )
        return actions

    def _release(self, skill: str, sink: ActionSink | None) -> list[RobotAction]:
        actions: list[RobotAction] = []
        self._gripper = GripperState.OPEN
        self._emit(
            RobotAction(gripper_state=GripperState.OPEN), skill, sink, actions
        )
        return actions


class CheckpointBackend:
    """Runs Step 2's exported policies. Filled in on the GB10 tomorrow.

    Left as an explicit stub with the real call signature so tomorrow's work is
    one file, not a new design. Per 6b.7 the inference call signatures still
    need to be settled with Teammate 2.
    """

    def __init__(self, nav_checkpoint: str, grasp_checkpoint: str) -> None:
        self.nav_checkpoint = nav_checkpoint
        self.grasp_checkpoint = grasp_checkpoint

    def run_skill(
        self,
        skill: str,
        args: dict[str, Any],
        scene: SceneGraph,
        on_action: ActionSink | None = None,
    ) -> SkillResult:
        raise NotImplementedError(
            "CheckpointBackend needs Step 2's exported checkpoints "
            f"(nav={self.nav_checkpoint!r}, grasp={self.grasp_checkpoint!r}). "
            "Use StubBackend until export_policy.py has produced them."
        )


class RobotTools:
    """The six skills, plus the attachment state the executor gates on."""

    def __init__(self, scene: SceneGraph, backend: RobotBackend) -> None:
        self.scene = scene
        self.backend = backend
        self._attached_to: str | None = None

    @property
    def attached_to(self) -> str | None:
        """Id of the object currently welded to the end-effector, if any."""
        return self._attached_to

    # ---- core skills ------------------------------------------------------

    def navigate_to(self, location_id: str, on_action: ActionSink | None = None) -> SkillResult:
        return self.backend.run_skill(
            "navigate_to", {"location_id": location_id}, self.scene, on_action
        )

    def pick_up(self, object_id: str, on_action: ActionSink | None = None) -> SkillResult:
        return self.backend.run_skill(
            "pick_up", {"object_id": object_id}, self.scene, on_action
        )

    # ---- wheelchair skills (Step 2 stretch task, 5.5) ---------------------

    def approach_wheelchair(
        self, object_id: str, on_action: ActionSink | None = None
    ) -> SkillResult:
        return self.backend.run_skill(
            "approach_wheelchair", {"object_id": object_id}, self.scene, on_action
        )

    def align_gripper(self, object_id: str, on_action: ActionSink | None = None) -> SkillResult:
        return self.backend.run_skill(
            "align_gripper", {"object_id": object_id}, self.scene, on_action
        )

    def attach_handle(self, object_id: str, on_action: ActionSink | None = None) -> SkillResult:
        result = self.backend.run_skill(
            "attach_handle", {"object_id": object_id}, self.scene, on_action
        )
        if result.ok:
            self._attached_to = object_id
        return result

    def nav_with_payload(
        self, location_id: str, on_action: ActionSink | None = None
    ) -> SkillResult:
        return self.backend.run_skill(
            "nav_with_payload", {"location_id": location_id}, self.scene, on_action
        )

    def detach(self, on_action: ActionSink | None = None) -> SkillResult:
        result = self.backend.run_skill("detach", {}, self.scene, on_action)
        if result.ok:
            self._attached_to = None
        return result

    # ---- dispatch ---------------------------------------------------------

    def call(self, tool_call: ToolCall, on_action: ActionSink | None = None) -> SkillResult:
        """Run a validated :class:`ToolCall`."""
        # Whitelisted against TOOL_SPECS rather than resolved by bare getattr, so
        # a malformed tool name can never reach an unrelated attribute.
        if tool_call.tool not in TOOL_SPECS:
            return SkillResult(False, f"no such skill: {tool_call.tool!r}")
        handler = getattr(self, tool_call.tool)
        return handler(**tool_call.args, on_action=on_action)
