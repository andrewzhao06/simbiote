# Step 4 — Robot Prompting (Agentic Control)

Master doc Part 6b. Owner: Teammate 4.

A typed natural-language instruction becomes a sequence of high-level skills that
execute autonomously against Step 2's trained policies. Every session is logged
as a demonstration, exactly like a Step 3 teleop session.

## Running it

Nothing to install beyond the package itself — no model, no GPU, no Isaac Sim.

```bash
export FACTORYFLOW_STAGE=./stage          # writable log dir (GB10: /var/factoryflow/stage)

# the two acceptance cases from 6b.5
python -m factoryflow.agentic.agentic_session "pick up the tray in the supply room"
python -m factoryflow.agentic.agentic_session "move the wheelchair to Room 2"

# exercise the failure path
python -m factoryflow.agentic.agentic_session "move the wheelchair to Room 2" \
    --fail-skill nav_with_payload
```

Tests: `uv run --with pytest --python 3.11 pytest -q`

## The chain

```
instruction
   │  command_parser.parse_instruction   ← the only LLM call
   ▼
[ToolCall, ToolCall, ...]                 validated against tool_schema AND the scene graph
   │  task_executor.execute               ← plain Python FSM, no LLM per step
   ▼
robot_tools.RobotTools                    six atomic skills over a RobotBackend
   │
   ▼
demo_logger                               JSONL trajectory + execution report sidecar
```

The FSM is the point (§6b.4). The model plans once; from there each skill has a
precondition, runs to completion or failure, and only then does the executor
decide whether to advance. Failures abort the run and release any held
constraint rather than leaving the arm welded to a payload.

## Swapping in the real thing

Two backends, both selected at the CLI, neither requiring a code change:

| | Today (laptop) | Tomorrow (GB10) |
| :---- | :---- | :---- |
| `--llm` | `fake` — deterministic rules, no model | `openai-compat` → Nemotron |
| `--robot` | `stub` — synthesised actions | `checkpoint` → Step 2's exports |

For a real model tonight, install Ollama, `ollama pull qwen3:8b`, then:

```bash
export FACTORYFLOW_LLM_URL=http://localhost:11434/v1
export FACTORYFLOW_LLM_MODEL=qwen3:8b
python -m factoryflow.agentic.agentic_session "pick up the tray in the supply room" --llm openai-compat
```

`FakeBackend` is not only a test double — it is the fallback if the GB10's model
server misbehaves on the day.

## What is still stubbed

- **`CheckpointBackend`** in `robot_tools.py` — raises `NotImplementedError`.
  Needs Step 2's exported ONNX/TorchScript and the call signatures agreed with
  Teammate 2 (§6b.7). This is the one piece that genuinely could not be built
  tonight.
- **OpenClaw registration** of `parse_instruction` as a gateway tool (Part 1) —
  needs the orchestrator running on the GB10.
- **Three shared schemas** — see `SCHEMAS_PROPOSAL.md` at the repo root. All
  three are proposals behind single adapter functions.
