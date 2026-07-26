"""Shared robot interface types, produced by both Step 3 (teleop) and Step 4
(agentic control) and consumed by Step 2's fine-tune loop.
"""

from factoryflow.robot_iface.actions import GripperState, Pose, RobotAction

__all__ = ["GripperState", "Pose", "RobotAction"]
