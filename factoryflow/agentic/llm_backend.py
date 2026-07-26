"""Pluggable LLM backends for instruction parsing — master doc 6b.3.

One interface, two implementations:

* :class:`FakeBackend` — deterministic rules over the loaded scene. No model, no
  download, no network. It makes the whole parser -> FSM -> logger chain runnable
  and testable tonight, and it is the fallback if the GB10's model server
  misbehaves on the day.
* :class:`OpenAICompatBackend` — any OpenAI-compatible ``/v1/chat/completions``
  endpoint. Ollama (Qwen3 8B / Phi-4-mini) on a laptop today, Nemotron on the
  GB10 tomorrow. Same code, different base URL.

Uses ``urllib`` rather than ``httpx``/``requests`` deliberately: zero runtime
dependencies means nothing extra to build for aarch64 or vendor onto the USB
drive (master doc Part 2, ARM caveat).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from factoryflow.agentic.scene_query import SceneGraph

__all__ = ["LLMBackend", "FakeBackend", "OpenAICompatBackend", "LLMError", "make_backend"]

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible port
DEFAULT_MODEL = "qwen3:8b"


class LLMError(RuntimeError):
    """The backend could not produce a completion."""


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeBackend:
    """Rule-based stand-in that emits the same JSON an LLM is asked for.

    Covers both acceptance instructions from 6b.5 plus the common phrasings
    around them. Anything it does not recognise returns an empty plan, which
    ``validate_calls`` rejects — an honest parse failure rather than a silently
    wrong one.
    """

    #: Verbs that mean "grasp and carry", as opposed to "just drive there".
    _FETCH = re.compile(r"\b(pick up|pickup|grab|fetch|get|collect|bring|take|carry|deliver)\b")
    _GOTO = re.compile(r"\b(go|drive|navigate|move|head|proceed)\b")
    _MOVE_PAYLOAD = re.compile(r"\b(move|push|take|bring|wheel|transport|deliver)\b")

    def __init__(self, scene: SceneGraph) -> None:
        self.scene = scene

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002 - system unused by design
        return json.dumps({"tool_calls": self._plan(user)}, indent=2)

    # ---- planning ---------------------------------------------------------

    def _plan(self, text: str) -> list[dict[str, object]]:
        lower = text.lower().strip()
        destination = self._destination(lower)
        obj_id = self.scene.resolve(lower, kind="object")
        obj = self.scene.get_object(obj_id) if obj_id else None

        # Compound / stateful: a payload that gets attached, driven, released.
        if obj and "wheelchair" in obj.label.lower() and destination and self._MOVE_PAYLOAD.search(lower):
            return [
                {"tool": "approach_wheelchair", "args": {"object_id": obj.id}},
                {"tool": "align_gripper", "args": {"object_id": obj.id}},
                {"tool": "attach_handle", "args": {"object_id": obj.id}},
                {"tool": "nav_with_payload", "args": {"location_id": destination}},
                {"tool": "detach", "args": {}},
            ]

        # Simple fetch: drive to where the object is, then grasp it.
        if obj and obj.is_graspable and self._FETCH.search(lower):
            source = self.scene.location_of(obj.id)
            calls: list[dict[str, object]] = []
            if source is not None:
                calls.append({"tool": "navigate_to", "args": {"location_id": source.id}})
            calls.append({"tool": "pick_up", "args": {"object_id": obj.id}})
            if destination and (source is None or destination != source.id):
                calls.append({"tool": "navigate_to", "args": {"location_id": destination}})
            return calls

        # Pure navigation.
        if destination and self._GOTO.search(lower):
            return [{"tool": "navigate_to", "args": {"location_id": destination}}]

        return []

    def _destination(self, lower: str) -> str | None:
        """The location named after the last ``to``, else any named location.

        Anchoring on ``to`` is what keeps "move the wheelchair to Room 2" from
        resolving to the room the wheelchair is currently sitting in.
        """
        idx = lower.rfind(" to ")
        if idx != -1:
            tail = lower[idx + 4 :]
            resolved = self.scene.resolve(tail, kind="location")
            if resolved:
                return resolved
        return self.scene.resolve(lower, kind="location")


class OpenAICompatBackend:
    """Any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        # Greedy decoding: the same instruction must produce the same plan
        # across rehearsal runs, or a stage failure is unreproducible.
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMError(
                f"could not reach the model server at {self.base_url}: {exc}. "
                "Is Ollama (or the GB10 inference server) running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"model server returned non-JSON: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape from model server: {body}") from exc


def make_backend(kind: str, scene: SceneGraph) -> LLMBackend:
    """Select a backend by name. Configuration comes from the environment so the
    same command line works on the laptop and on the GB10."""
    if kind == "fake":
        return FakeBackend(scene)
    if kind in ("openai-compat", "openai", "ollama"):
        return OpenAICompatBackend(
            base_url=os.environ.get("FACTORYFLOW_LLM_URL", DEFAULT_BASE_URL),
            model=os.environ.get("FACTORYFLOW_LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("FACTORYFLOW_LLM_KEY"),
        )
    raise ValueError(f"unknown LLM backend {kind!r}; expected 'fake' or 'openai-compat'")
