"""Vertical end-effector control for the Ridgeback's Franka, by measured Jacobian.

Teleop's manipulate mode asks for one thing: move the claw up and down. That's
a 1-DOF task on a 7-DOF arm, which does not need a full IK stack -- and there
isn't one available offline anyway. Isaac Sim's Lula/RMPflow solvers want
robot-descriptor YAMLs that this asset pack doesn't ship, and cuRobo (staged in
/home/dell/AI/repos/curobo) isn't wired up.

So this measures the one row of the Jacobian it actually needs. At startup it
nudges each arm joint and watches how far the end-effector moves in world z,
giving dz/dq for every joint. Vertical motion is then the minimum-norm joint
step that produces the requested dz:

    dq = s * (dz / (s . s))        where s_i = dz/dq_i

That's the pseudo-inverse of a 1xN Jacobian row. It's the same "nudge and
measure rather than assume" approach `IsaacHospital._calibrate_base_axes` uses
for the base axes, and for the same reason: the asset authors its own
orientation and joint layout, so hardcoding a mapping is a guess.

Limitation worth naming: the sensitivities are measured once, at the folded
pose, and a Jacobian is only valid locally. Joint travel is therefore clamped
to `MAX_JOINT_TRAVEL` around that pose, which keeps the linearisation honest
and stops the arm from unfolding into the building.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Candidate end-effector bodies, best first. Which exist depends on how the
# Franka was composed into the Ridgeback asset.
EE_CANDIDATES = ("panda_hand", "panda_link8", "panda_link7")

# Per-joint radians of nudge used to measure dz/dq. Big enough to move the
# end-effector clear of solver noise, small enough to stay a local estimate.
CALIBRATION_NUDGE = 0.12

# Physics steps to let a nudge settle before measuring.
CALIBRATION_SETTLE = 12

# How far any arm joint may travel from the folded pose, radians.
MAX_JOINT_TRAVEL = 0.9

# Cap on a single tick's joint step, radians -- limits how hard the arm lunges
# at a target the operator moved suddenly.
MAX_JOINT_STEP = 0.05

# Metres of end-effector travel the operator's full hand range maps onto,
# centred on the folded pose's height.
LIFT_RANGE = 0.45


class ArmLift:
    """Servos the Franka's end-effector height by writing into the drive target.

    Writes into `hospital._target` rather than issuing its own
    `ArticulationAction`, because `_apply_velocity` already applies that whole
    vector every tick -- a second action would just fight it, last-write-wins.
    """

    def __init__(self, hospital, ee_candidates: Sequence[str] = EE_CANDIDATES):
        self.hospital = hospital
        self.robot = hospital.robot

        self.arm_dofs: list[int] = [
            self.robot.get_dof_index(name)
            for name in self.robot.dof_names
            if name.startswith("panda_joint")
        ]
        self.finger_dofs: list[int] = [
            self.robot.get_dof_index(name)
            for name in self.robot.dof_names
            if "finger" in name.lower()
        ]
        if not self.arm_dofs:
            raise RuntimeError("no panda_joint DOFs found -- is this the Franka asset?")

        self._ee = self._resolve_ee(ee_candidates)
        self.home = np.array(
            [float(self.hospital._target[dof]) for dof in self.arm_dofs], dtype=np.float64
        )
        self.home_z = self.ee_z()
        self.sensitivity = self._calibrate()
        self.commanded_z = self.home_z

    # -- setup ---------------------------------------------------------------

    def _resolve_ee(self, candidates: Sequence[str]):
        from isaacsim.core.prims import SingleRigidPrim

        errors = []
        for name in candidates:
            path = f"{self.hospital.robot_path}/{name}"
            try:
                prim = SingleRigidPrim(path)
                prim.initialize(self.hospital.sim.physics_sim_view)
                prim.get_world_pose()  # fail now rather than mid-session
                return prim
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError(
            "could not bind an end-effector body. Tried:\n  " + "\n  ".join(errors)
        )

    def ee_z(self) -> float:
        position, _ = self._ee.get_world_pose()
        return float(position[2])

    def _settle(self, steps: int = CALIBRATION_SETTLE) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(ArticulationAction(joint_positions=self.hospital._target))
        for _ in range(steps):
            self.hospital.sim.step(render=False)

    def _calibrate(self) -> np.ndarray:
        """Measure dz/dq for each arm joint at the folded pose."""

        sensitivity = np.zeros(len(self.arm_dofs), dtype=np.float64)
        for slot, dof in enumerate(self.arm_dofs):
            original = float(self.hospital._target[dof])

            self.hospital._target[dof] = original + CALIBRATION_NUDGE
            self._settle()
            z_up = self.ee_z()

            self.hospital._target[dof] = original - CALIBRATION_NUDGE
            self._settle()
            z_down = self.ee_z()

            self.hospital._target[dof] = original
            self._settle()

            # Central difference: less biased than one-sided, and the arm is
            # already being driven both ways here anyway.
            sensitivity[slot] = (z_up - z_down) / (2.0 * CALIBRATION_NUDGE)

        norm = float(sensitivity @ sensitivity)
        if norm < 1e-8:
            raise RuntimeError(
                "arm lift calibration failed -- no joint moved the end-effector "
                f"vertically (sensitivities {sensitivity.round(4).tolist()}). "
                "Check the arm is not jammed against geometry at spawn."
            )
        return sensitivity

    # -- control -------------------------------------------------------------

    def height_for(self, commanded: float, lo: float, hi: float) -> float:
        """Map a teleop workspace height onto a reachable band around home.

        The teleop workspace box is expressed in the robot's own frame, but the
        Franka sits on the Ridgeback's deck, so its end-effector is nowhere near
        those absolute numbers. Rather than guess the offset, the operator's
        range is mapped onto `LIFT_RANGE` metres centred on the pose the arm
        actually folded into -- which is reachable by construction.
        """

        if hi <= lo:
            return self.home_z
        fraction = (float(commanded) - lo) / (hi - lo)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return self.home_z + (fraction - 0.5) * LIFT_RANGE

    def servo_to(self, target_z: float) -> float:
        """Step the arm one tick toward `target_z`. Returns the current height."""

        self.commanded_z = float(target_z)
        current_z = self.ee_z()
        error = self.commanded_z - current_z

        norm = float(self.sensitivity @ self.sensitivity)
        step = self.sensitivity * (error / norm)  # min-norm solution of s.dq = error
        step = np.clip(step, -MAX_JOINT_STEP, MAX_JOINT_STEP)

        for slot, dof in enumerate(self.arm_dofs):
            proposed = float(self.hospital._target[dof]) + float(step[slot])
            # Keep the pose near where the sensitivities were measured.
            low = self.home[slot] - MAX_JOINT_TRAVEL
            high = self.home[slot] + MAX_JOINT_TRAVEL
            self.hospital._target[dof] = float(np.clip(proposed, low, high))

        return current_z

    def set_gripper(self, closed: bool, open_width: float = 0.04) -> None:
        """Franka fingers: 0.0 closed, ~0.04 m each when open."""

        for dof in self.finger_dofs:
            self.hospital._target[dof] = 0.0 if closed else open_width

    def hold(self) -> None:
        """Keep the arm where it is (drive mode) -- targets simply aren't changed."""

        return
