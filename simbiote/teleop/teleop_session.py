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

from simbiote import demo_logger
from simbiote.robot_iface.actions import RobotAction
from simbiote.teleop.action_bridge import DEFAULT_HOST, DEFAULT_PORT
from simbiote.teleop.camera_source import CameraSource, FrameSource, open_camera
from simbiote.teleop.hand_tracking import create_tracker, resolve_backend
from simbiote.teleop.ik_bridge import IKBridge
from simbiote.teleop.preview import PreviewWindow

# Consecutive empty reads before we conclude the camera is gone rather than
# just slow.
MAX_CONSECUTIVE_DROPS = 30


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
        pose = action.arm_target_pose
        arm = f"({pose.position[0]:.2f},{pose.position[1]:.2f},{pose.position[2]:.2f})" if pose else "none"
        print(
            f"[teleop] base=({vx:+.2f},{vy:+.2f},{omega:+.2f}) "
            f"arm={arm} gripper={action.gripper_state.value}"
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
        camera_source: Optional[CameraSource] = None,
        backend: str = "auto",
        sink_name: str = "console",
        mirror: bool = True,
        rotate: int = 0,
        threaded_camera: bool = True,
        camera_retries: int = 5,
        max_reconnects: int = 20,
    ):
        self.robot_sink = robot_sink
        self.target_fps = target_fps
        self.show_preview = show_preview
        # A front-facing phone camera is not mirrored by the capture pipeline,
        # so "move my hand right" reads as image-left without this flip. The
        # flip is applied to the frame itself so preview and tracking agree.
        self.mirror = mirror
        self.camera = open_camera(
            device_index=camera_index,
            fps=target_fps,
            source=camera_source,
            rotate=rotate,
            threaded=threaded_camera,
            open_retries=camera_retries,
        )
        # Remembered so a reconnect can rebuild the same source. Take it from
        # the opened camera, which has already resolved env-var fallbacks.
        self._camera_source_resolved = self.camera.source
        self._rotate = rotate
        self._threaded_camera = threaded_camera
        self._camera_retries = camera_retries
        self.max_reconnects = max_reconnects
        self._reconnects = 0
        self.backend = resolve_backend(backend)
        self.hand_tracker = create_tracker(self.backend)
        self.ik_bridge = IKBridge()
        self.preview = PreviewWindow(
            backend=self.backend, sink=sink_name, enabled=show_preview, mirrored=mirror
        )
        self.session_id = demo_logger.start_session(session_id, source="teleop", task="nav")
        self._fps = 0.0
        self._running = False

    def run(self, max_frames: Optional[int] = None) -> str:
        """Runs the teleop loop until stopped or max_frames is reached.

        Returns the session_id so the caller can export/save the trajectory.
        """

        self._running = True
        frame_period = 1.0 / self.target_fps
        frame_count = 0
        dropped = 0

        try:
            while self._running:
                loop_start = time.time()

                frame = self.camera.read()
                if frame is None:
                    # A dropped frame is normal; a run of them means the stream
                    # died. For a phone that's routine rather than fatal -- iOS
                    # suspends a backgrounded DroidCam and its socket stops
                    # delivering -- so try to pick the stream back up instead of
                    # ending the session and losing the trajectory so far.
                    dropped += 1
                    if dropped >= MAX_CONSECUTIVE_DROPS:
                        if not self._reconnect_camera():
                            break
                        dropped = 0
                    continue
                dropped = 0

                if self.mirror:
                    import cv2

                    frame = cv2.flip(frame, 1)

                landmarks = self.hand_tracker.get_hand_landmarks(frame)
                action = self.ik_bridge.landmarks_to_action(landmarks)

                self.robot_sink.apply_action(action)
                self.robot_sink.step()
                demo_logger.log_action(action, source="teleop", session_id=self.session_id)

                if self.show_preview:
                    if not self.preview.show(frame, landmarks, action, self._fps):
                        self._running = False

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break

                elapsed = time.time() - loop_start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)

                # Smoothed so the readout doesn't flicker frame to frame.
                instant = 1.0 / max(time.time() - loop_start, 1e-6)
                self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
        finally:
            self.stop()

        return self.session_id

    def _reconnect_camera(self) -> bool:
        """Reopen the camera after the stream went dead. False = give up.

        The robot keeps whatever it was last told until frames resume; the
        Isaac side independently stops the base once commands go stale, so a
        reconnect gap is safe rather than a runaway.
        """

        if self._reconnects >= self.max_reconnects:
            print(
                f"[teleop] camera stayed dead after {self._reconnects} reconnect "
                f"attempts ({self.camera.source}) -- stopping."
            )
            return False

        self._reconnects += 1
        print(
            f"[teleop] camera stopped delivering frames -- reconnecting "
            f"({self._reconnects}/{self.max_reconnects}). Is the phone app foregrounded?"
        )
        try:
            self.camera.release()
        except Exception:  # noqa: BLE001 - it's already broken
            pass

        try:
            self.camera = open_camera(
                source=self._camera_source_resolved,
                fps=self.target_fps,
                rotate=self._rotate,
                threaded=self._threaded_camera,
                open_retries=self._camera_retries,
            )
        except RuntimeError as exc:
            print(f"[teleop] reconnect failed: {exc}")
            return False

        print(f"[teleop] camera back: {self.camera.width}x{self.camera.height}")
        return True

    def stop(self) -> None:
        self._running = False
        self.camera.release()
        self.hand_tracker.close()
        self.preview.close()


def _build_robot_sink(kind: str, gui: bool, udp_host: str, udp_port: int) -> RobotSink:
    if kind == "console":
        return ConsoleRobotSink()
    if kind == "pybullet":
        from simbiote.sim_stub.toy_robot_env import ToyRobotEnv

        return ToyRobotEnv(gui=gui)
    if kind == "udp":
        from simbiote.teleop.action_bridge import UdpActionSink

        return UdpActionSink(host=udp_host, port=udp_port)
    raise ValueError(f"unknown robot sink kind: {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Simbiote hand-tracking teleop session.")
    parser.add_argument(
        "--camera-index", type=int, default=None,
        help="OS camera index, for a phone-webcam client publishing to /dev/videoN.",
    )
    parser.add_argument(
        "--camera-url", type=str, default=None,
        help="Stream URL instead of a local device, e.g. http://<phone-ip>:4747/video. "
             "Needs no v4l2loopback module and no root.",
    )
    parser.add_argument(
        "--backend", choices=["auto", "mediapipe", "wilor"], default="auto",
        help="Hand-tracking backend. 'auto' picks mediapipe when installed, else wilor "
             "(the GB10 case -- mediapipe has no aarch64 wheel).",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Don't horizontally flip the feed (flip is on by default so the preview reads like a mirror).",
    )
    parser.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help="Rotate the feed clockwise. Use 90/270 when the phone streams portrait-side-up, "
             "otherwise your up/down hand motion drives the arm's left/right axis.",
    )
    parser.add_argument(
        "--no-threaded-camera", action="store_true",
        help="Read frames inline instead of on a background thread. Threading is on by "
             "default (it stops camera latency and inference time from adding up); turn it "
             "off when feeding a video file, where dropping frames loses data.",
    )
    parser.add_argument(
        "--camera-retries", type=int, default=5,
        help="Retries (2s apart) when the camera won't open. Phone webcam servers accept "
             "one client at a time, so raise this to wait for the app to start serving "
             "rather than health-checking the stream first -- a probe takes the slot.",
    )
    parser.add_argument(
        "--max-reconnects", type=int, default=20,
        help="How many times to reopen the camera if the stream dies mid-session. "
             "iOS suspending a backgrounded phone-webcam app is routine, so teleop "
             "waits for it to come back rather than ending the session.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (omit to run until Ctrl+C or 'q').")
    parser.add_argument(
        "--sink", choices=["console", "pybullet", "udp"], default="pybullet",
        help="'pybullet' opens a GUI viewer with the toy stand-in robot; 'console' just prints "
             "actions (no extra deps); 'udp' streams them to a simulator in another process "
             "(Isaac Sim -- see scripts/gb10/teleop_hospital.py).",
    )
    parser.add_argument("--udp-host", default=DEFAULT_HOST, help="Host for --sink udp.")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_PORT, help="Port for --sink udp.")
    parser.add_argument("--no-gui", action="store_true", help="Run the pybullet sink headless (DIRECT mode).")
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True,
        help="Show the live window: camera feed + hand skeleton + the RobotAction it retargets to. "
             "On by default; --no-preview runs headless.",
    )
    parser.add_argument("--save", action="store_true", help="Print where the trajectory was saved when the session ends.")
    args = parser.parse_args()

    robot_sink = _build_robot_sink(
        args.sink, gui=not args.no_gui, udp_host=args.udp_host, udp_port=args.udp_port
    )
    session = TeleopSession(
        robot_sink=robot_sink,
        camera_index=args.camera_index,
        target_fps=args.fps,
        session_id=args.session_id,
        show_preview=args.preview,
        camera_source=args.camera_url,
        backend=args.backend,
        sink_name=args.sink,
        mirror=not args.no_mirror,
        rotate=args.rotate,
        threaded_camera=not args.no_threaded_camera,
        camera_retries=args.camera_retries,
        max_reconnects=args.max_reconnects,
    )

    print(
        f"[teleop] session_id={session.session_id} sink={args.sink} "
        f"backend={session.backend} camera={session.camera.source} "
        f"({session.camera.width}x{session.camera.height}) -- Ctrl+C or 'q' to stop"
    )
    try:
        session_id = session.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        session_id = session.session_id
        session.stop()

    trajectory = demo_logger.export_trajectory(session_id)
    print(f"[teleop] logged {len(trajectory.steps)} action(s) for session {session_id}")
    if args.save:
        path = demo_logger.session_path(session_id)
        print(f"[teleop] trajectory saved to {path}")

    robot_sink.close()


if __name__ == "__main__":
    main()
