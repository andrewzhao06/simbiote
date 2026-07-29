"""File-backed command queue between an operator and the live hospital sim.

Why a directory of files and not a socket
-----------------------------------------
Isaac Sim's application has to be pumped from the *main* thread -- an HTTP
server would need a second thread, and every skill it dispatched would have to
marshal back to main before touching the simulation. A queue the main loop
polls between frames keeps the whole thing single-threaded, which is the only
arrangement that is obviously correct here.

It also survives the two processes being started independently, in either
order, from different terminals, with no port to agree on.

Layout under `root` (default `<stage>/control`, where `<stage>` is whatever
`demo_logger.stage_dir()` resolves to)::

    status.json          written by the sim: ready flag, robot pose, locations
    inbox/<seq>.json     written by the client: one instruction
    outbox/<seq>.json    written by the sim: the result for that instruction

`seq` is a zero-padded counter so the sim processes instructions in the order
they were sent. Files are written to a temporary name and then renamed, since
the reader is polling and would otherwise pick up a half-written file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simbiote import demo_logger


def default_control_root() -> Path:
    """`<stage>/control`, resolved the same way every other session artifact is.

    Deriving this from `__file__` (as it used to) puts the queue inside
    site-packages for an installed copy, and ignores `$SIMBIOTE_STAGE`, so the
    sim and the client could end up polling two different directories.
    """
    return demo_logger.stage_dir() / "control"


def _write_atomic(path: Path, payload: dict) -> None:
    """Write JSON so a polling reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Mid-rename or mid-write; the caller is polling and will retry.
        return None


@dataclass
class ControlQueue:
    root: Path = field(default_factory=default_control_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.inbox = self.root / "inbox"
        self.outbox = self.root / "outbox"
        self.status_path = self.root / "status.json"

    def ensure(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

    # -- sim side -----------------------------------------------------------

    def reset(self) -> None:
        """Clear stale traffic from a previous session.

        Called by the sim at startup: without it, a client that timed out
        against a dead sim leaves instructions that the next one would execute
        the moment it comes up, which is a surprising way to make a robot move.
        """
        self.ensure()
        for directory in (self.inbox, self.outbox):
            for item in directory.glob("*.json"):
                item.unlink(missing_ok=True)
        self.status_path.unlink(missing_ok=True)

    def publish_status(self, **fields: Any) -> None:
        _write_atomic(self.status_path, {"updated": time.time(), **fields})

    def next_instruction(self) -> tuple[str, str] | None:
        """Oldest unprocessed instruction as (seq, text), or None."""
        pending = sorted(self.inbox.glob("*.json"))
        for path in pending:
            payload = _read_json(path)
            if payload is None:
                continue
            path.unlink(missing_ok=True)
            return path.stem, str(payload.get("instruction", ""))
        return None

    def publish_result(self, seq: str, payload: dict) -> None:
        _write_atomic(self.outbox / f"{seq}.json", payload)

    # -- client side --------------------------------------------------------

    def status(self) -> dict | None:
        if not self.status_path.is_file():
            return None
        return _read_json(self.status_path)

    def submit(self, instruction: str) -> str:
        """Enqueue an instruction, returning its sequence id."""
        self.ensure()
        # Sequence off the highest id ever seen in either direction, so ids
        # keep increasing even after the inbox drains.
        seen = [
            int(p.stem)
            for p in list(self.inbox.glob("*.json")) + list(self.outbox.glob("*.json"))
            if p.stem.isdigit()
        ]
        seq = f"{max(seen, default=0) + 1:06d}"
        _write_atomic(self.inbox / f"{seq}.json", {"instruction": instruction, "sent": time.time()})
        return seq

    def await_result(self, seq: str, timeout_s: float = 900.0, poll_s: float = 0.5) -> dict | None:
        deadline = time.time() + timeout_s
        path = self.outbox / f"{seq}.json"
        while time.time() < deadline:
            if path.is_file():
                payload = _read_json(path)
                if payload is not None:
                    return payload
            time.sleep(poll_s)
        return None
