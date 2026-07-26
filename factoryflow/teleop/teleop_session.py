"""Ties the teleop chain together:

    camera_source -> hand_tracking -> ik_bridge -> RobotAction -> robot AND demo_logger

Runs one frame per iteration: grab a camera frame, find hand landmarks,
retarget to a RobotAction, hand it to whatever RobotSink is driving the
robot (a real sim/robot binding, or tonight's toy PyBullet stand-in), and
log it for Step 2's fine-tune loop.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Protocol

from factoryflow import demo_logger
from factoryflow.robot_iface.actions import RobotAction
from factoryflow.teleop.camera_source import FrameSource, open_camera
from factoryflow.teleop.hand_tracking import HandTracker
from factoryflow.teleop.ik_bridge import IKBridge


class RobotSink(Protocol):
    """Anything that can consume a RobotAction each frame.

    Implemented tonight by sim_stub.toy_robot_env.ToyRobotEnv (PyBullet
    visualization) and ConsoleRobotSink (no dependencies, prints only).
    Step 2's real robot/sim binding implements the same interface later
    without teleop_session.py needing to change.
    """

    def apply_action(self, action: RobotAction) -> None: ...

    def step(self) -> None: ...

    def close(self) -> None: ...


class ConsoleRobotSink:
    """A dependency-free RobotSink that just prints each action.

    Useful for headless smoke-testing the teleop chain (e.g. CI, or a
    machine without pybullet installed) without needing a camera or a
    physics viewer.
    """

    def apply_action(self, action: RobotAction) -> None:
        vx, vy, omega = action.base_velocity
        x, y, z = action.arm_target_pose.position
        print(
            f"[teleop] base=({vx:+.2f},{vy:+.2f},{omega:+.2f}) "
            f"arm=({x:.2f},{y:.2f},{z:.2f}) gripper={action.gripper_state.value}"
        )

    def step(self) -> None:
        pass

    def close(self) -> None:
        pass


class TeleopSession:
    def __init__(
        self,
        robot_sink: RobotSink,
        camera_index: Optional[int] = None,
        target_fps: int = 30,
        session_id: Optional[str] = None,
        show_preview: bool = False,
    ):
        self.robot_sink = robot_sink
        self.target_fps = target_fps
        self.show_preview = show_preview
        self.camera: FrameSource = open_camera(device_index=camera_index, fps=target_fps)
        self.hand_tracker = HandTracker()
        self.ik_bridge = IKBridge()
        self.session_id = demo_logger.start_session(session_id)
        self._running = False

    def run(self, max_frames: Optional[int] = None) -> str:
        """Runs the teleop loop until stopped or max_frames is reached.

        Returns the session_id so the caller can export/save the trajectory.
        """

        self._running = True
        frame_period = 1.0 / self.target_fps
        frame_count = 0

        try:
            while self._running:
                loop_start = time.time()

                frame = self.camera.read()
                if frame is None:
                    continue

                landmarks = self.hand_tracker.get_hand_landmarks(frame)
                action = self.ik_bridge.landmarks_to_action(landmarks)

                self.robot_sink.apply_action(action)
                self.robot_sink.step()
                demo_logger.log_action(action, source="teleop", session_id=self.session_id)

                if self.show_preview:
                    self._show_preview(frame, landmarks)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break

                elapsed = time.time() - loop_start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)
        finally:
            self.stop()

        return self.session_id

    def _show_preview(self, frame, landmarks) -> None:
        import cv2

        if landmarks is not None:
            h, w = frame.shape[:2]
            for x, y, _ in landmarks.points:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
        cv2.imshow("FactoryFlow Teleop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.camera.release()
        self.hand_tracker.close()
        if self.show_preview:
            import cv2

            cv2.destroyAllWindows()


def _build_robot_sink(kind: str, gui: bool) -> RobotSink:
    if kind == "console":
        return ConsoleRobotSink()
    if kind == "pybullet":
        from factoryflow.sim_stub.toy_robot_env import ToyRobotEnv

        return ToyRobotEnv(gui=gui)
    raise ValueError(f"unknown robot sink kind: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FactoryFlow hand-tracking teleop session.")
    parser.add_argument("--camera-index", type=int, default=None, help="OS camera index (Iriun Webcam).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (omit to run until Ctrl+C or 'q').")
    parser.add_argument(
        "--sink", choices=["console", "pybullet"], default="pybullet",
        help="'pybullet' opens a GUI viewer with the toy stand-in robot; 'console' just prints actions (no extra deps).",
    )
    parser.add_argument("--no-gui", action="store_true", help="Run the pybullet sink headless (DIRECT mode).")
    parser.add_argument("--preview", action="store_true", help="Show the camera feed with hand landmarks overlaid.")
    parser.add_argument("--save", action="store_true", help="Save the trajectory to disk when the session ends.")
    args = parser.parse_args()

    robot_sink = _build_robot_sink(args.sink, gui=not args.no_gui)
    session = TeleopSession(
        robot_sink=robot_sink,
        camera_index=args.camera_index,
        target_fps=args.fps,
        session_id=args.session_id,
        show_preview=args.preview,
    )

    print(f"[teleop] session_id={session.session_id} sink={args.sink} -- Ctrl+C or 'q' to stop")
    try:
        session_id = session.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        session_id = session.session_id
        session.stop()

    trajectory = demo_logger.export_trajectory(session_id)
    print(f"[teleop] logged {len(trajectory.steps)} action(s) for session {session_id}")
    if args.save:
        path = demo_logger.get_logger().save(session_id)
        print(f"[teleop] saved trajectory to {path}")

    robot_sink.close()


if __name__ == "__main__":
    main()
