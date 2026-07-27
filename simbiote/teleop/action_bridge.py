"""Ship `RobotAction`s between two Python interpreters over UDP.

Why a bridge instead of one process
-----------------------------------
The teleop chain and Isaac Sim cannot share an interpreter on this box:

* Teleop needs `cv2`, `ultralytics`, `smplx` and torch 2.13 -- present in the
  repo `.venv`, absent from Isaac Sim's bundled python.
* `IsaacHospital` needs `isaacsim`/`omni`/`pxr`, which only exist inside Isaac
  Sim's bundled python (torch 2.9) and are not pip-installable into a venv.

Rather than force one interpreter to grow the other's dependency tree -- which
would mean pip-installing OpenCV and ultralytics into the Kit runtime and
hoping nothing shifts underneath Isaac Sim -- teleop stays where it works and
speaks to Isaac over a socket.

Why UDP
-------
Teleop is a stream of *states*, not a stream of *events*: only the newest hand
pose matters, and a dropped frame is better than a late one. UDP gives exactly
that -- no head-of-line blocking, no reconnect logic, and a slow consumer can
never stall the producer's camera loop. The receiver drains its socket every
tick and keeps the last datagram, so it always acts on the freshest command.

Deliberately stdlib-only (`socket`, `json`, `dataclasses`) plus
`robot_iface.actions`, which is itself stdlib-only. That's what lets the same
module import cleanly under both interpreters.
"""

from __future__ import annotations

import json
import random
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from simbiote.robot_iface.actions import RobotAction

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47800

# Generous vs a ~14 FPS teleop stream: one dropped frame shouldn't trip it,
# but a backgrounded phone or a killed sender should stop the robot quickly.
DEFAULT_STALE_AFTER = 0.5


def _encode(action: RobotAction, sequence: int, sender: int) -> bytes:
    return json.dumps(
        {"seq": sequence, "sender": sender, "action": action.to_dict()}
    ).encode("utf-8")


def _decode(payload: bytes) -> Tuple[int, int, RobotAction]:
    message = json.loads(payload.decode("utf-8"))
    return (
        int(message.get("seq", 0)),
        int(message.get("sender", 0)),
        RobotAction.from_dict(message["action"]),
    )


@dataclass
class UdpActionSink:
    """A `RobotSink` that forwards each action to a listening simulator.

    Satisfies teleop_session's RobotSink protocol (`apply_action`/`step`/
    `close`), so it drops into `--sink udp` alongside console and pybullet.
    Sends are fire-and-forget: if nothing is listening, the frames go nowhere
    and teleop keeps running rather than dying mid-session.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    _socket: Optional[socket.socket] = None
    _sequence: int = 0
    _sender_id: int = 0

    def __post_init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        # Identifies this process instance. Sequence numbers restart from zero
        # every time teleop restarts, so without a way to tell one run from the
        # next the receiver's "never go backwards" rule would reject the whole
        # new session until it counted past the old one's final sequence.
        if not self._sender_id:
            self._sender_id = random.getrandbits(32)

    def apply_action(self, action: RobotAction) -> None:
        self._sequence += 1
        try:
            self._socket.sendto(
                _encode(action, self._sequence, self._sender_id), (self.host, self.port)
            )
        except (BlockingIOError, OSError):
            # Buffer full or no route -- the next frame supersedes this one.
            pass

    def step(self) -> None:
        pass

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class ActionReceiver:
    """Simulator-side endpoint: hands back the newest action received.

    `latest()` never blocks. It drains everything queued on the socket and
    returns only the last datagram, so a simulator running slower than the
    camera acts on current input instead of working through a backlog.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        stale_after: float = DEFAULT_STALE_AFTER,
    ):
        self.stale_after = stale_after
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.setblocking(False)
        self._last_action: Optional[RobotAction] = None
        self._last_time: float = 0.0
        self._last_sequence: int = -1
        self._last_sender: int = 0
        self.received = 0
        self.dropped = 0

    def latest(self) -> Optional[RobotAction]:
        """Drain the socket and return the freshest action, or None if stale."""

        while True:
            try:
                payload, _addr = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                sequence, sender, action = _decode(payload)
            except (ValueError, KeyError):
                continue  # ignore anything that isn't ours
            # A new sender means teleop restarted: adopt its numbering rather
            # than measuring it against the previous run's counter.
            if sender != self._last_sender:
                self._last_sender = sender
                self._last_sequence = -1
            # Within one sender, UDP can still reorder; never let an older
            # frame overwrite a newer one.
            if sequence <= self._last_sequence:
                self.dropped += 1
                continue
            if self._last_sequence >= 0:
                self.dropped += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            self._last_action = action
            self._last_time = time.time()
            self.received += 1

        if self._last_action is None:
            return None
        if time.time() - self._last_time > self.stale_after:
            return None
        return self._last_action

    @property
    def is_live(self) -> bool:
        return self._last_action is not None and (time.time() - self._last_time) <= self.stale_after

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "ActionReceiver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
