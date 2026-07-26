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

Against a real model on the GB10, warm it first — see [Model profiles](#model-profiles).

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
| `--llm` | `fake` — deterministic rules, no model | `openai-compat` + `--llm-profile` |
| `--robot` | `stub` — synthesised actions | `checkpoint` → Step 2's exports |

For a real model tonight, install Ollama, `ollama pull qwen3:8b`, then:

```bash
python -m factoryflow.agentic.agentic_session "pick up the tray in the supply room" \
    --llm openai-compat --llm-profile qwen3-8b
```

## Model profiles

`--llm-profile` (or `$FACTORYFLOW_LLM_PROFILE`) picks the target. Sizes are from
master doc Part 1's memory budget.

| Profile | Footprint | Resident | Notes |
| :---- | :---- | :---- | :---- |
| `qwen3-8b` | ~5 GB | laptop | default |
| `phi4-mini` | ~3 GB | laptop | laptop alternative |
| `nemotron-super` | ~60 GB | **swapped** | best reasoning; cannot coexist with a loaded Isaac Sim |
| `nemotron-nano` | ~25 GB | always | lower quality; no juggling |

`FACTORYFLOW_LLM_URL` / `_MODEL` / `_TIMEOUT` / `_KEY` override any profile field.
The two Nemotron model ids are placeholders — the id is whatever your inference
server advertises, so confirm them against `factoryflow_gb10_model_downloads.md`
and override with `FACTORYFLOW_LLM_MODEL` if they differ. No code change needed.

**Nemotron 3 Ultra is refused outright.** At ~275 GB it is ~2.5x the GB10's
entire 128 GB — a hard ceiling, not a trade-off (Part 1). Naming it in a profile
or in `FACTORYFLOW_LLM_MODEL` fails immediately rather than after a doomed load.

### Super vs Nano — decide before 9 AM

Part 1 says Super (~60 GB) cannot stay resident alongside a fully-loaded Isaac
Sim: it gets unloaded for training and **swapped back in for the agentic phase**.
That puts a multi-minute weight load directly in front of Step 4's demo beat.

Step 4 handles either choice, but they are not equally risky:

- **Nano** removes the failure mode. Always resident, nothing to sequence.
- **Super** buys reasoning quality. Whether Step 4 needs it is arguable — every
  emitted tool call is already validated against the scene graph, so a
  hallucinated `room_7` is rejected under either model.

With Super, run preflight the moment training releases memory:

```bash
python -m factoryflow.agentic.agentic_session --llm openai-compat \
    --llm-profile nemotron-super --preflight
```

It polls until the server answers, exits 0 (ready) or 1 (timed out), and takes an
optional instruction to warm-then-run in one command.

## When the model server is not there

`FakeBackend` is not only a test double — it is the fallback if the GB10's model
server misbehaves, and Part 1's swap rule makes that likely rather than
hypothetical. So it is wired up rather than assumed:

- Transport failures (connection refused, 503 from a server still loading) are
  retried with backoff. A 4xx is not — retrying cannot fix it.
- If the server is still unreachable, the run **degrades to `fake`**, prints a
  loud warning to stderr, and records `llm.degraded: true` plus the reason in the
  `.report.json` sidecar. A plan produced by rules rather than by Nemotron is a
  materially different claim to make on stage; the audit trail says which ran.
- `--no-fallback` disables that. **Use it during rehearsals** — you want a broken
  model server discovered at 4 PM, not covered up until stage.

This is distinct from the one corrective retry in `parse_instruction`. That
re-asks a model that answered badly; this re-asks a server that was not there.

## What is still stubbed

- **`CheckpointBackend`** in `robot_tools.py` — raises `NotImplementedError`.
  Needs Step 2's exported ONNX/TorchScript and the call signatures agreed with
  Teammate 2 (§6b.7). This is the one piece that genuinely could not be built
  tonight.
- **OpenClaw registration** of `parse_instruction` as a gateway tool (Part 1) —
  needs the orchestrator running on the GB10.
- **Three shared schemas** — see `SCHEMAS_PROPOSAL.md` at the repo root. All
  three are proposals behind single adapter functions.
