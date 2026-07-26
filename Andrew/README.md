# Andrew — agentic / LLM control

Owns `simbiote/agentic/` and the `simbiote/fixtures/hospital_scene_graph.json`
fixture. Tests live in `Andrew/tests/` and run as part of the root `pytest`
suite. Entry point: `simbiote-agentic` (see `pyproject.toml`).

## What's here

- `scene_query.py` — `SceneGraph` loader/query layer over the hospital scene
  fixture (locations, objects, `is_graspable`/`grasp_type`/`mass_kg`,
  alias-based `resolve()` for natural-language phrases).
- `tool_schema.py` — the tool-call contract between the LLM and the executor
  (`ToolCall`, `TOOL_SPECS`, `parse_tool_calls()`, `validate_calls()`).
  Tolerant of markdown-fenced JSON, bare arrays, and
  `name`/`arguments`-style keys, since small local models don't always
  follow the requested shape exactly.
- `command_parser.py` — `parse_instruction()`, the single LLM call per
  instruction. One corrective retry on a schema/validation failure, then a
  clean `ParseError` — never a retry loop that could hang on stage.
- `llm_backend.py` — pluggable backends:
  - `FakeBackend` — deterministic rule-based planner, no model or network.
  - `OpenAICompatBackend` — any OpenAI-compatible `/v1/chat/completions`
    endpoint (Ollama on the laptop today, Nemotron on the GB10). Retries
    transport failures (connection refused, 5xx) with backoff; a 4xx fails
    immediately.
  - `FallbackBackend` — tries a real backend, degrades to `FakeBackend` on
    transport failure, and records which one actually served the parse.
  - Model profiles (`PROFILES`) for `qwen3-8b` (laptop default),
    `nemotron-super`, and `nemotron-nano`, sized against the GB10's memory
    budget. Nemotron Ultra (550B-A55B, ~275 GB) is refused outright — it
    cannot fit in 128 GB of unified memory.
- `robot_tools.py` — the atomic skills (`navigate_to`, `pick_up`, and the
  wheelchair sequence `approach_wheelchair` -> `align_gripper` ->
  `attach_handle` -> `nav_with_payload` -> `detach`) plus the attachment
  state the executor gates on. `StubBackend` is a deterministic stand-in;
  `CheckpointBackend` wires `navigate_to`/`pick_up` to Suraj's real
  `simbiote.robot_iface.skills`, with the wheelchair skills left as an
  explicit `NotImplementedError` pending a persistent env handle.
- `task_executor.py` — the FSM that runs a validated plan one skill at a
  time: checks preconditions before each step, aborts and compensates
  (auto-`detach`) on failure, and times out a hung skill instead of hanging
  the demo.
- `agentic_session.py` — `run_session()`: parse -> execute -> log, with an
  execution-report sidecar (`demo_logger.write_report()`) recording which
  LLM backend actually served the plan. CLI: `simbiote-agentic`.

## Tests

`Andrew/tests/` — scene graph parsing/resolution, tool-call schema
tolerance and scene validation, instruction parsing (including the two
acceptance instructions from the build spec), LLM backend profiles/retry/
fallback behavior, the task-executor FSM (ordering, failure, compensation,
timeout), and end-to-end `run_session()` runs against a real temp staging
directory.

## GB10 next steps

- Point `SIMBIOTE_LLM_PROFILE=nemotron-super` (or `nemotron-nano`) and
  `SIMBIOTE_LLM_URL` at the local inference server; confirm the served
  model id against `SIMBIOTE_LLM_MODEL` if it differs from `PROFILES`.
- Run `simbiote-agentic --preflight` as soon as training/Isaac Sim releases
  memory, so Nemotron Super's swap-back-in load finishes before an operator
  types an instruction.
- Swap `CheckpointBackend`'s wheelchair `NotImplementedError` for a real
  implementation once Suraj's `skills.py` exposes a persistent robot/env
  handle across a compound plan.
