"""The atomic skills — spec §6b.5.

Each skill is a thin, stateful wrapper over a :class:`RobotBackend`:

* :class:`StubBackend` — tonight. Synthesises a plausible action sequence and
  reports success, so the parser -> FSM -> logger chain is fully exercisable
  without Isaac Sim or any trained policy. Can be told to fail a named skill,
  which is how the executor's failure paths get tested.
* :class:`CheckpointBackend` — Step 2's real trained policies
  (`simbiote.robot_iface.skills`). `navigate_to`/`pick_up` are wired to the
  real checkpoints; the wheelchair-stretch skills still need a *persistent*
  robot/env handle carried across a whole compound plan (approach -> align ->
  attach -> nav_with_payload -> detach), which per-call `skills.py` helpers
  don't yet support -- see the `NotImplementedError` below for what's left.

The wheelchair skills are implemented even though Step 2's wheelchair task is a
stretch goal (§5.5): §6b.4 is explicit that the task hierarchy is the right shape
for any multi-step instruction regardless, and against the stub they are nearly
free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from simbiote.agentic.scene_query import SceneGraph
from simbiote.agentic.tool_schema import TOOL_SPECS, ToolCall
from simbiote.robot_iface.actions import GripperState, Pose, RobotAction

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
            position=(target.x, target.y, target.z + 0.15),
            orientation=(target.qx, target.qy, target.qz, target.qw),
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
    """Runs Step 2's exported policies via `simbiote.robot_iface.skills`.

    `navigate_to`/`pick_up` map directly onto `skills.navigate_to()` /
    `skills.pick_up()` -- both take a plain string id and run a short-lived
    PyBullet rollout to completion, exactly the atomic, self-contained call
    this backend's per-skill interface needs.

    The wheelchair skills (`approach_wheelchair`, `align_gripper`,
    `attach_handle`, `nav_with_payload`, `detach`) don't fit that shape yet:
    `skills.attach_handle()`/`detach()` need a *live* `SpawnedRobot` handle and
    an already-formed `GraspConstraint` carried across the whole compound
    plan, not just a string id per call. Wiring that up needs `RobotTools` (or
    this backend) to own a persistent env/robot handle across a session --
    left as an explicit `NotImplementedError` with the real shape rather than
    faked, since `StubBackend` already exercises the FSM/executor logic for
    that path end to end.
    """

    def __init__(self, nav_checkpoint: str, grasp_checkpoint: str, *, gui: bool = False) -> None:
        self.nav_checkpoint = nav_checkpoint
        self.grasp_checkpoint = grasp_checkpoint
        self.gui = gui

    def run_skill(
        self,
        skill: str,
        args: dict[str, Any],
        scene: SceneGraph,
        on_action: ActionSink | None = None,
    ) -> SkillResult:
        from simbiote.robot_iface import skills as robot_skills

        if skill == "navigate_to":
            loc_id = args["location_id"]
            loc = scene.get_location(loc_id)
            if loc is None:
                return SkillResult(False, f"unknown location {loc_id!r}")
            # Pass every location, not just the requested one: navigate_to
            # rescales the scene into the nav arena and needs the whole set to
            # compute that mapping. Handing it a single point would make the
            # scale depend on which place was asked for.
            locations = {
                other.id: (other.pose.x, other.pose.y)
                for other in scene.list_locations()
            }
            locations.setdefault(loc_id, (loc.pose.x, loc.pose.y))
            info = robot_skills.navigate_to(
                loc_id, checkpoint_path=self.nav_checkpoint, locations=locations, gui=self.gui
            )
            ok = bool(info.get("success", info.get("terminated", True)))
            return SkillResult(ok, f"navigate_to({loc_id}) -> {info}")

        if skill == "pick_up":
            obj_id = args["object_id"]
            obj = scene.get_object(obj_id)
            if obj is None:
                return SkillResult(False, f"unknown object {obj_id!r}")
            objects = {obj_id: (obj.pose.x, obj.pose.y, obj.pose.z)}
            info = robot_skills.pick_up(
                obj_id, checkpoint_path=self.grasp_checkpoint, objects=objects, gui=self.gui
            )
            ok = bool(info.get("success", info.get("terminated", True)))
            return SkillResult(ok, f"pick_up({obj_id}) -> {info}")

        raise NotImplementedError(
            f"CheckpointBackend has no wiring for {skill!r} yet -- it needs a "
            "persistent robot/env handle carried across a whole compound plan "
            "(approach_wheelchair -> align_gripper -> attach_handle -> "
            "nav_with_payload -> detach), which simbiote.robot_iface.skills' "
            "per-call helpers don't expose. Use StubBackend for this skill "
            "until that lands."
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

    # ---- wheelchair skills (Step 2 stretch task, §5.5) ---------------------

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
