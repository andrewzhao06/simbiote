"""Verification 6 — model profiles, and surviving the Nemotron Super swap.

Part 1's extended memory rule means the model server behind Step 4 is expected
to be absent or mid-load exactly when the agentic beat runs. These tests pin the
behaviour that makes that survivable: retry what a later attempt could fix,
degrade loudly rather than failing the demo, and refuse a checkpoint that cannot
physically load.

No test here touches the network — ``urlopen`` is replaced throughout.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from factoryflow.agentic.command_parser import ParseError, parse_instruction
from factoryflow.agentic.llm_backend import (
    PROFILES,
    FakeBackend,
    FallbackBackend,
    LLMError,
    OpenAICompatBackend,
    describe_backend,
    make_backend,
    primary_of,
    resolve_profile,
)

_ENV_VARS = (
    "FACTORYFLOW_LLM_PROFILE",
    "FACTORYFLOW_LLM_URL",
    "FACTORYFLOW_LLM_MODEL",
    "FACTORYFLOW_LLM_TIMEOUT",
    "FACTORYFLOW_LLM_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """A developer's own FACTORYFLOW_LLM_* settings must not steer these tests."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---- test doubles --------------------------------------------------------


class _Response:
    def __init__(self, body: object, status: int = 200) -> None:
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _completion(text: str) -> _Response:
    return _Response({"choices": [{"message": {"content": text}}]})


def _refused() -> urllib.error.URLError:
    return urllib.error.URLError("[Errno 111] Connection refused")


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1", code, "boom", {}, None)  # type: ignore[arg-type]


class _Urlopen:
    """Replays a scripted sequence of responses/exceptions, counting calls."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request: object, timeout: float | None = None) -> object:
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _backend(monkeypatch, urlopen: _Urlopen, **kwargs: object) -> OpenAICompatBackend:
    monkeypatch.setattr("factoryflow.agentic.llm_backend.urllib.request.urlopen", urlopen)
    kwargs.setdefault("backoff_s", 0.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return OpenAICompatBackend("http://x/v1", "test-model", **kwargs)  # type: ignore[arg-type]


PLAN = json.dumps({"tool_calls": [{"tool": "navigate_to", "args": {"location_id": "room_2"}}]})


# ---- 1. model profiles ---------------------------------------------------


def test_both_gb10_options_are_available():
    """Part 1 names exactly two viable GB10 targets. Both must be selectable."""
    assert PROFILES["nemotron-super"].footprint_gb == 60.0
    assert PROFILES["nemotron-nano"].footprint_gb == 25.0


def test_super_is_marked_swapped_and_nano_always_resident():
    """The distinction that drives every other behaviour in this module."""
    assert PROFILES["nemotron-super"].resident == "swapped"
    assert PROFILES["nemotron-nano"].resident == "always"


def test_nemotron_timeouts_allow_for_a_cold_load():
    """60 s was fine for an 8B on a laptop; a 60 GB NVFP4 cold start is not."""
    for name in ("nemotron-super", "nemotron-nano"):
        assert PROFILES[name].timeout_s > 60.0
        assert PROFILES[name].ready_timeout_s >= PROFILES[name].timeout_s


def test_laptop_profile_is_the_default():
    assert resolve_profile().name == "qwen3-8b"


def test_unknown_profile_is_rejected_by_name():
    with pytest.raises(LLMError, match="unknown model profile"):
        resolve_profile("nemotron-medium")


def test_env_selects_the_profile(monkeypatch):
    monkeypatch.setenv("FACTORYFLOW_LLM_PROFILE", "nemotron-nano")
    assert resolve_profile().name == "nemotron-nano"


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv("FACTORYFLOW_LLM_PROFILE", "nemotron-nano")
    assert resolve_profile("nemotron-super").name == "nemotron-super"


def test_env_overrides_url_model_and_timeout(monkeypatch):
    """The served model id depends on how the inference server was launched."""
    monkeypatch.setenv("FACTORYFLOW_LLM_URL", "http://gb10:8000/v1")
    monkeypatch.setenv("FACTORYFLOW_LLM_MODEL", "nemotron-3-super")
    monkeypatch.setenv("FACTORYFLOW_LLM_TIMEOUT", "42")
    profile = resolve_profile("nemotron-super")
    assert (profile.base_url, profile.model, profile.timeout_s) == (
        "http://gb10:8000/v1", "nemotron-3-super", 42.0
    )


def test_nonnumeric_timeout_env_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("FACTORYFLOW_LLM_TIMEOUT", "soon")
    with pytest.raises(LLMError, match="not a number"):
        resolve_profile()


# ---- 2. Ultra is a hard ceiling, not a trade-off -------------------------


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "Nemotron-3-Ultra",
        "some-550B-thing",
        "mystery-a55b",
    ],
)
def test_ultra_is_refused_wherever_it_is_named(monkeypatch, model):
    """~275 GB against 128 GB of unified memory. Fail in milliseconds, not
    after a long doomed load."""
    monkeypatch.setenv("FACTORYFLOW_LLM_MODEL", model)
    with pytest.raises(LLMError, match="275 GB"):
        resolve_profile("nemotron-super")


def test_ultra_is_refused_at_construction_too():
    with pytest.raises(LLMError, match="cannot load"):
        OpenAICompatBackend("http://x/v1", "nemotron-3-ultra-550b-a55b")


def test_super_and_nano_are_not_caught_by_the_ultra_guard():
    for name in ("nemotron-super", "nemotron-nano"):
        assert resolve_profile(name).name == name


# ---- 3. transport retry --------------------------------------------------


def test_connection_refused_is_retried_then_succeeds(monkeypatch):
    """The shape of 'Super is unloaded or still coming up'."""
    urlopen = _Urlopen(_refused(), _refused(), _completion("ok"))
    assert _backend(monkeypatch, urlopen).complete("s", "u") == "ok"
    assert urlopen.calls == 3


def test_retries_are_bounded(monkeypatch):
    urlopen = _Urlopen(_refused())
    with pytest.raises(LLMError, match="after 3 attempts"):
        _backend(monkeypatch, urlopen).complete("s", "u")
    assert urlopen.calls == 3


def test_503_is_retried_because_a_loading_server_returns_it(monkeypatch):
    urlopen = _Urlopen(_http(503), _completion("ok"))
    assert _backend(monkeypatch, urlopen).complete("s", "u") == "ok"
    assert urlopen.calls == 2


def test_400_is_not_retried_because_retrying_cannot_fix_it(monkeypatch):
    urlopen = _Urlopen(_http(400))
    with pytest.raises(LLMError, match="HTTP 400"):
        _backend(monkeypatch, urlopen).complete("s", "u")
    assert urlopen.calls == 1


def test_backoff_grows_between_attempts(monkeypatch):
    slept: list[float] = []
    backend = _backend(
        monkeypatch, _Urlopen(_refused()), backoff_s=1.0, sleep=slept.append
    )
    with pytest.raises(LLMError):
        backend.complete("s", "u")
    assert slept == [1.0, 2.0]  # no sleep after the final attempt


def test_malformed_response_shape_is_not_retried(monkeypatch):
    urlopen = _Urlopen(_Response({"unexpected": True}))
    with pytest.raises(LLMError, match="unexpected response shape"):
        _backend(monkeypatch, urlopen).complete("s", "u")
    assert urlopen.calls == 1


# ---- 4. readiness --------------------------------------------------------


def test_ping_is_false_when_the_server_is_absent(monkeypatch):
    assert _backend(monkeypatch, _Urlopen(_refused())).ping() is False


def test_ping_is_true_when_the_server_answers(monkeypatch):
    assert _backend(monkeypatch, _Urlopen(_Response({"data": []}))).ping() is True


def test_wait_ready_polls_until_the_server_comes_up(monkeypatch):
    urlopen = _Urlopen(_refused(), _refused(), _Response({"data": []}))
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    backend = _backend(monkeypatch, urlopen)
    assert backend.wait_ready(60.0, poll_s=0.0, now=lambda: next(clock)) is True


def test_wait_ready_gives_up_at_the_deadline(monkeypatch):
    clock = iter([0.0, 10.0, 20.0])
    backend = _backend(monkeypatch, _Urlopen(_refused()))
    assert backend.wait_ready(5.0, poll_s=0.0, now=lambda: next(clock)) is False


# ---- 5. fallback ---------------------------------------------------------


def test_unreachable_model_still_produces_a_plan(monkeypatch, scene):
    """The demo beat survives a model server that is still loading."""
    llm = FallbackBackend(_backend(monkeypatch, _Urlopen(_refused())), FakeBackend(scene))
    calls = parse_instruction("pick up the tray in the supply room", scene, llm)
    assert [c.tool for c in calls] == ["navigate_to", "pick_up"]


def test_degradation_is_recorded_not_silent(monkeypatch, scene):
    """A plan from rules rather than from Nemotron is a different claim to make
    on stage, so the audit trail has to say which one ran (Part 3)."""
    llm = FallbackBackend(_backend(monkeypatch, _Urlopen(_refused())), FakeBackend(scene))
    parse_instruction("go to the nurse station", scene, llm)
    record = describe_backend(llm)
    assert record["degraded"] is True
    assert record["served_by"] == "fake"
    assert "Connection refused" in str(record["degrade_reason"])


def test_degradation_is_announced(monkeypatch, scene):
    seen: list[Exception] = []
    llm = FallbackBackend(
        _backend(monkeypatch, _Urlopen(_refused())), FakeBackend(scene), on_degrade=seen.append
    )
    parse_instruction("go to the nurse station", scene, llm)
    assert len(seen) == 1


def test_working_primary_is_not_degraded(monkeypatch, scene):
    llm = FallbackBackend(_backend(monkeypatch, _Urlopen(_completion(PLAN))), FakeBackend(scene))
    calls = parse_instruction("go to Room 2", scene, llm)
    assert [c.tool for c in calls] == ["navigate_to"]
    record = describe_backend(llm)
    assert record["degraded"] is False
    assert record["served_by"] == "openai-compat:test-model"


def test_a_bad_model_reply_does_not_trigger_fallback(monkeypatch, scene):
    """Fallback is for an absent server. A server that answers badly is the
    schema retry's job, and must not be masked by rule-based output."""
    llm = FallbackBackend(
        _backend(monkeypatch, _Urlopen(_completion("not json"))), FakeBackend(scene)
    )
    with pytest.raises(ParseError):
        parse_instruction("go to Room 2", scene, llm)


# ---- 6. make_backend wiring ---------------------------------------------


def test_make_backend_wraps_in_fallback_by_default(scene):
    llm = make_backend("openai-compat", scene, profile="nemotron-nano")
    assert isinstance(llm, FallbackBackend)
    assert isinstance(llm.fallback, FakeBackend)


def test_no_fallback_leaves_the_failure_hard(monkeypatch, scene):
    """Rehearsals want to discover a broken model server at 4 PM, not on stage."""
    llm = make_backend("openai-compat", scene, profile="nemotron-nano", fallback=False)
    monkeypatch.setattr(
        "factoryflow.agentic.llm_backend.urllib.request.urlopen", _Urlopen(_refused())
    )
    llm.backoff_s = 0.0  # type: ignore[attr-defined]
    with pytest.raises(ParseError, match="model backend failed"):
        parse_instruction("go to Room 2", scene, llm)


def test_make_backend_applies_the_profile(scene):
    llm = make_backend("openai-compat", scene, profile="nemotron-super")
    assert primary_of(llm).model == PROFILES["nemotron-super"].model  # type: ignore[attr-defined]
    assert primary_of(llm).timeout_s == PROFILES["nemotron-super"].timeout_s  # type: ignore[attr-defined]


def test_fake_backend_needs_no_profile(scene):
    assert isinstance(make_backend("fake", scene), FakeBackend)


def test_unknown_kind_is_rejected(scene):
    with pytest.raises(ValueError, match="unknown LLM backend"):
        make_backend("telepathy", scene)
