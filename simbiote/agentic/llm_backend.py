"""Pluggable LLM backends for instruction parsing — spec §6b.3.

One interface, three implementations:

* :class:`FakeBackend` — deterministic rules over the loaded scene. No model, no
  download, no network. It makes the whole parser -> FSM -> logger chain runnable
  and testable tonight, and it is the fallback if the GB10's model server
  misbehaves on the day.
* :class:`OpenAICompatBackend` — any OpenAI-compatible ``/v1/chat/completions``
  endpoint. Ollama (Qwen3 8B / Phi-4-mini) on a laptop today, Nemotron on the
  GB10 tomorrow. Same code, different base URL.
* :class:`FallbackBackend` — tries a primary, falls back to a second backend on
  transport failure. See "Surviving the swap" below.

Uses ``urllib`` rather than ``httpx``/``requests`` deliberately: zero runtime
dependencies means nothing extra to build for aarch64 or vendor onto the USB
drive (spec Part 2, ARM caveat).

Surviving the swap
------------------

Part 1's extended memory rule says Nemotron Super (~60 GB) *cannot* stay
resident alongside a fully-loaded Isaac Sim, and gets swapped back in for the
live agentic phase. So on the day, the model server behind this module is
expected to be absent or still loading at exactly the moment Step 4 is asked to
parse something. That is a normal state, not an error, and this module treats it
as one: probe with :meth:`OpenAICompatBackend.wait_ready`, retry transport
failures, and — unless explicitly disabled — degrade to :class:`FakeBackend`
rather than failing the demo beat.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from simbiote.agentic.scene_query import SceneGraph

__all__ = [
    "LLMBackend",
    "FakeBackend",
    "OpenAICompatBackend",
    "FallbackBackend",
    "LLMError",
    "ModelProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "resolve_profile",
    "backend_id",
    "primary_of",
    "describe_backend",
    "make_backend",
]

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible port
DEFAULT_MODEL = "qwen3:8b"

#: Laptop default. The GB10 profile is selected explicitly (CLI or
#: ``SIMBIOTE_LLM_PROFILE``) so nobody points at a 60 GB model by accident.
DEFAULT_PROFILE = "qwen3-8b"


class LLMError(RuntimeError):
    """The backend could not produce a completion."""


# ---------------------------------------------------------------------------
# Model profiles — spec §6b.3 / Part 1 memory budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """A named model target, sized against Part 1's memory budget.

    ``footprint_gb`` is documentation, not enforcement — the watchdog on the
    GB10 owns the 105 GB ceiling. It is here so ``--llm-profile`` prints what it
    is about to cost before anyone loads it.
    """

    name: str
    model: str
    base_url: str
    footprint_gb: float | None
    #: Per-request ceiling. A 60 GB NVFP4 checkpoint answering its first prompt
    #: after a cold load is not a 60-second operation.
    timeout_s: float
    #: How long :meth:`OpenAICompatBackend.wait_ready` will wait for the server
    #: to come up — dominated by weight load from NVMe, not by inference.
    ready_timeout_s: float
    resident: str
    notes: str = ""


#: Model ids are what the *inference server* advertises, which depends on how it
#: was launched. Confirm the two Nemotron ids against the GB10 model-download
#: notes on the day; both are overridable with ``SIMBIOTE_LLM_MODEL`` without
#: touching this table.
PROFILES: dict[str, ModelProfile] = {
    "qwen3-8b": ModelProfile(
        name="qwen3-8b",
        model="qwen3:8b",
        base_url=DEFAULT_BASE_URL,
        footprint_gb=5.0,
        timeout_s=60.0,
        ready_timeout_s=30.0,
        resident="laptop (Ollama)",
        notes="Tonight's laptop model.",
    ),
    "qwen3-8b-vllm": ModelProfile(
        name="qwen3-8b-vllm",
        model="qwen3-8b",
        base_url="http://localhost:8000/v1",
        footprint_gb=6.0,
        timeout_s=180.0,
        ready_timeout_s=600.0,
        resident="always",
        notes=(
            "The GB10's working NL-parsing model: local Qwen3-8B-NVFP4 served "
            "by the NGC vLLM container on :8000. Small enough (~6 GB) to stay "
            "resident alongside Isaac Sim, unlike the Nemotron profiles, and "
            "ample for turning one instruction into a couple of tool calls. "
            "Launch it with --enforce-eager: Inductor autotuning inside the "
            "container dies on 'CUDA driver error: operation not permitted'. "
            "The 27B Qwen3.6 in /home/dell/Qwen3.6-27B-FP8 also loads but is "
            "multimodal and stalls for minutes on encoder-cache profiling."
        ),
    ),
    "phi4-mini": ModelProfile(
        name="phi4-mini",
        model="phi4-mini",
        base_url=DEFAULT_BASE_URL,
        footprint_gb=3.0,
        timeout_s=60.0,
        ready_timeout_s=30.0,
        resident="laptop (Ollama)",
        notes="Laptop alternative to Qwen3 8B.",
    ),
    "nemotron-super": ModelProfile(
        name="nemotron-super",
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="http://localhost:8000/v1",
        footprint_gb=60.0,
        timeout_s=300.0,
        ready_timeout_s=900.0,
        resident="swapped",
        notes=(
            "Part 1: ~60 GB, roughly half the machine. Cannot be resident "
            "alongside a fully-loaded Isaac Sim — it is unloaded for training "
            "and swapped back in for the agentic/demo phase. Preflight it "
            "before the operator types anything."
        ),
    ),
    "nemotron-nano": ModelProfile(
        name="nemotron-nano",
        model="nvidia/nemotron-3-nano-30b-a3b",
        base_url="http://localhost:8000/v1",
        footprint_gb=25.0,
        timeout_s=180.0,
        ready_timeout_s=300.0,
        resident="always",
        notes=(
            "Part 1's simpler fallback: lower reasoning quality than Super, but "
            "coexists with a fully-loaded Isaac Sim with none of the swapping. "
            "Removes the cold-reload failure mode entirely."
        ),
    ),
}

#: Part 1 is unambiguous: Ultra (550B-A55B) is ~275 GB at its native NVFP4
#: checkpoint, ~2.5x the GB10's entire 128 GB of unified memory. Not a
#: quality/risk trade-off like Super vs. Nano — a hard hardware ceiling. Caught
#: here so a hopeful ``SIMBIOTE_LLM_MODEL`` fails in milliseconds rather than
#: after a long, doomed load.
_ULTRA = re.compile(r"(ultra|550b|a55b)", re.IGNORECASE)


def _reject_ultra(model: str) -> None:
    if _ULTRA.search(model):
        raise LLMError(
            f"refusing to target {model!r}: Nemotron 3 Ultra (550B-A55B) is ~275 GB "
            "at NVFP4, ~2.5x the GB10's 128 GB of unified memory (spec Part 1). "
            "It cannot load. Use 'nemotron-super' or 'nemotron-nano'."
        )


def resolve_profile(name: str | None = None) -> ModelProfile:
    """Look up a profile, applying environment overrides.

    Precedence: explicit argument, ``$SIMBIOTE_LLM_PROFILE``, then the laptop
    default. ``SIMBIOTE_LLM_URL`` / ``_MODEL`` / ``_TIMEOUT`` override the
    chosen profile's fields, so the same command line works on both machines.
    """
    key = name or os.environ.get("SIMBIOTE_LLM_PROFILE") or DEFAULT_PROFILE
    profile = PROFILES.get(key)
    if profile is None:
        raise LLMError(
            f"unknown model profile {key!r}; expected one of {', '.join(sorted(PROFILES))}"
        )

    model = os.environ.get("SIMBIOTE_LLM_MODEL", profile.model)
    _reject_ultra(model)

    timeout_raw = os.environ.get("SIMBIOTE_LLM_TIMEOUT")
    try:
        timeout_s = float(timeout_raw) if timeout_raw else profile.timeout_s
    except ValueError as exc:
        raise LLMError(f"SIMBIOTE_LLM_TIMEOUT is not a number: {timeout_raw!r}") from exc

    return ModelProfile(
        name=profile.name,
        model=model,
        base_url=os.environ.get("SIMBIOTE_LLM_URL", profile.base_url),
        footprint_gb=profile.footprint_gb,
        timeout_s=timeout_s,
        ready_timeout_s=profile.ready_timeout_s,
        resident=profile.resident,
        notes=profile.notes,
    )


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeBackend:
    """Rule-based stand-in that emits the same JSON an LLM is asked for.

    Covers both acceptance instructions from §6b.5 plus the common phrasings
    around them. Anything it does not recognise returns an empty plan, which
    ``validate_calls`` rejects — an honest parse failure rather than a silently
    wrong one.
    """

    #: Identifies which backend actually served a parse, for the CLI banner and
    #: the execution-report sidecar (Part 3's audit trail).
    backend_name = "fake"

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


#: HTTP statuses that mean "ask again shortly" rather than "you asked wrongly".
#: 503 is what a server still loading weights returns, which is precisely the
#: Super-swap case from Part 1.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class OpenAICompatBackend:
    """Any OpenAI-compatible chat-completions endpoint."""

    backend_name = "openai-compat"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        sleep: "object | None" = None,
    ) -> None:
        _reject_ultra(model)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        # Greedy decoding: the same instruction must produce the same plan
        # across rehearsal runs, or a stage failure is unreproducible.
        self.temperature = temperature
        # Transport-level retry, distinct from the one schema retry in
        # parse_instruction. That one re-asks a model that answered badly; this
        # one re-asks a server that was not there yet.
        self.max_attempts = max(1, max_attempts)
        self.backoff_s = backoff_s
        self._sleep = sleep or time.sleep

    @classmethod
    def from_profile(
        cls, profile: ModelProfile, *, api_key: str | None = None, **kwargs: object
    ) -> "OpenAICompatBackend":
        return cls(
            base_url=profile.base_url,
            model=profile.model,
            api_key=api_key if api_key is not None else os.environ.get("SIMBIOTE_LLM_KEY"),
            timeout_s=profile.timeout_s,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def backend_id(self) -> str:
        return f"openai-compat:{self.model}"

    # ---- transport --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        }

    def _post(self, path: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        """POST JSON, retrying only failures that a later attempt could fix."""
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        last: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # A 4xx that is not rate-limiting is our fault; retrying it just
                # burns stage time.
                if exc.code not in _RETRYABLE_STATUS:
                    raise LLMError(
                        f"model server at {self.base_url} returned HTTP {exc.code}: {exc.reason}"
                    ) from exc
                last = exc
            except urllib.error.URLError as exc:
                # Connection refused — the expected shape of "Super is unloaded
                # or still coming up".
                last = exc
            except json.JSONDecodeError as exc:
                raise LLMError(f"model server returned non-JSON: {exc}") from exc

            if attempt < self.max_attempts:
                self._sleep(self.backoff_s * (2 ** (attempt - 1)))

        raise LLMError(
            f"could not reach the model server at {self.base_url} after "
            f"{self.max_attempts} attempts: {last}. Is the inference server running? "
            "If this is Nemotron Super mid-swap, it may still be loading weights — "
            "preflight it before the demo beat (spec Part 1)."
        )

    # ---- readiness --------------------------------------------------------

    def ping(self, timeout_s: float = 5.0) -> bool:
        """One cheap probe: is the server answering yet? Never raises."""
        request = urllib.request.Request(
            f"{self.base_url}/models", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def wait_ready(self, timeout_s: float, *, poll_s: float = 5.0, now=time.monotonic) -> bool:
        """Poll until the server answers or the deadline passes.

        Loading ~60 GB of NVFP4 weights off NVMe is minutes, not seconds, so the
        agentic phase warms the model as soon as training ends rather than
        discovering the wait when an operator has already typed an instruction.
        """
        deadline = now() + timeout_s
        while True:
            if self.ping(timeout_s=min(5.0, poll_s)):
                return True
            if now() >= deadline:
                return False
            self._sleep(poll_s)

    # ---- completion -------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        body = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "stream": False,
            },
            timeout_s=self.timeout_s,
        )
        try:
            return body["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from model server: {body}") from exc


class FallbackBackend:
    """Try the primary; degrade to the fallback on transport failure.

    :class:`FakeBackend` has always been the fallback if the GB10's model
    server misbehaves. Part 1's swap rule makes that likely rather than
    hypothetical — the model is deliberately unloaded for training and
    reloaded for the demo — so it is now wired up instead of assumed.

    Degradation is one-way per call and always announced. It is not silent: a
    plan produced by rules rather than by Nemotron is a materially different
    claim to make on stage, and :attr:`served_by` is written into the execution
    report so the audit trail says which one ran.
    """

    backend_name = "fallback"

    def __init__(
        self,
        primary: LLMBackend,
        fallback: LLMBackend,
        *,
        on_degrade: "object | None" = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.on_degrade = on_degrade
        self.served_by: str | None = None
        self.degrade_reason: str | None = None

    def complete(self, system: str, user: str) -> str:
        try:
            result = self.primary.complete(system, user)
        except LLMError as exc:
            self.degrade_reason = str(exc)
            self.served_by = backend_id(self.fallback)
            if self.on_degrade is not None:
                self.on_degrade(exc)  # type: ignore[operator]
            return self.fallback.complete(system, user)
        self.served_by = backend_id(self.primary)
        return result


def backend_id(backend: LLMBackend) -> str:
    """A short, loggable name for whichever backend served a parse."""
    explicit = getattr(backend, "backend_id", None)
    if isinstance(explicit, str):
        return explicit
    return str(getattr(backend, "backend_name", type(backend).__name__))


def primary_of(backend: LLMBackend) -> LLMBackend:
    """Unwrap a :class:`FallbackBackend` to the backend that owns the connection.

    Preflight talks to the real server, not to the rule-based understudy.
    """
    return backend.primary if isinstance(backend, FallbackBackend) else backend


def describe_backend(backend: LLMBackend) -> dict[str, object]:
    """A loggable record of what was configured and what actually served.

    Goes into the execution-report sidecar. Part 3 wants every agentic session
    in the audit trail, and "Nemotron planned this" versus "the rule-based
    fallback planned this" is the single most load-bearing fact in that record.
    """
    primary = primary_of(backend)
    record: dict[str, object] = {
        "configured": backend_id(primary),
        "model": getattr(primary, "model", None),
        "served_by": backend_id(backend),
        "degraded": False,
        "degrade_reason": None,
    }
    if isinstance(backend, FallbackBackend):
        record["fallback"] = backend_id(backend.fallback)
        record["served_by"] = backend.served_by
        record["degraded"] = backend.degrade_reason is not None
        record["degrade_reason"] = backend.degrade_reason
    return record


def make_backend(
    kind: str,
    scene: SceneGraph,
    *,
    profile: str | None = None,
    fallback: bool = True,
    on_degrade: "object | None" = None,
) -> LLMBackend:
    """Select a backend by name.

    Configuration comes from the profile table and the environment, so the same
    command line works on the laptop and on the GB10. A real backend is wrapped
    in :class:`FallbackBackend` unless ``fallback=False`` — rehearsals want the
    hard failure so a broken model server is discovered at 4 PM, not on stage.
    """
    if kind == "fake":
        return FakeBackend(scene)
    if kind in ("openai-compat", "openai", "ollama"):
        primary = OpenAICompatBackend.from_profile(resolve_profile(profile))
        if not fallback:
            return primary
        return FallbackBackend(primary, FakeBackend(scene), on_degrade=on_degrade)
    raise ValueError(f"unknown LLM backend {kind!r}; expected 'fake' or 'openai-compat'")
