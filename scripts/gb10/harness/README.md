# GB10 verification harness

Ground truth for the long-horizon build-out. Every part of the work owns a check
that actually exercises the capability; the harness runs them, records machine-
readable reports, detects regressions, and regenerates `STATUS.md` so the
human-readable tracker cannot drift from reality.

```
run_checks.sh      run every check, print a table, write a report      (agents A-D own the checks)
checks/            one executable per part                             (add yours here)
trend.py           latest report vs. previous: what moved, what broke
watch.sh           run the suite on an interval, append durable history, shout on regression
render_status.py   regenerate the Parts table in STATUS.md from reports/latest.json
parts.json         part -> check-prefix map that render_status.py reads
reports/           reports, per-check logs, history.jsonl, watch.log
```

## The exit-code contract

A check is **any executable file** in `checks/`. It prints human-readable detail
on stdout and exits:

| Code | Status | Meaning |
| :-- | :-- | :-- |
| 0 | `PASS` | the capability works, verified by exercising it — not by checking a file exists |
| 1 | `FAIL` | it is meant to work and does not. A regression. |
| 2 | `BLOCKED` | a prerequisite is missing; this part has not been built yet |
| 3 | `SKIP` | deliberately not applicable in this environment |

`124`/`137` (timeout kill) are recorded as `TIMEOUT` and counted as failures.

**The last non-empty line of stdout is the check's one-line summary** and is what
lands in the report and the results table. Everything else you print goes to the
per-check log.

`BLOCKED` is how long-horizon progress is tracked. A part is done when its check
flips `BLOCKED → PASS` **and stays there** — which is exactly what `trend.py`
watches for in both directions.

## Adding a check

1. Create `checks/NN_short_name.sh`, `chmod +x` it. The `NN_` prefix orders the
   run and is how `parts.json` groups checks into parts.
2. Source the shared helpers and use them to exit — they enforce the contract:

   ```bash
   #!/usr/bin/env bash
   # Part N: one sentence on what capability this proves.
   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

   require_dgrut_python                       # -> BLOCKED if the venv is missing
   ...
   pass "tracer compiled, forward pass ok"    # or fail / blocked / skip
   ```

   `common.sh` also exports the paths (`DGRUT_ROOT`, `ISAAC_RELEASE`, …) and
   `TORCH_CUDA_ARCH_LIST=12.1`. It is sourced, not executed, so it must stay
   non-executable or `run_checks.sh` will try to run it as a check.
3. Distinguish **BLOCKED** (not built yet) from **FAIL** (built and broken)
   carefully — that distinction is the whole progress signal.
4. Keep it under the timeout (`CHECK_TIMEOUT`, default 1800s) and remember checks
   that boot Isaac Sim or compile CUDA consume the same RAM the OS needs.
5. Add or extend an entry in `parts.json` so `STATUS.md` picks it up.

## Running

```bash
./run_checks.sh                    # everything
./run_checks.sh 20 30              # only checks whose name starts with 20 or 30
FAIL_ON_BLOCKED=1 ./run_checks.sh  # gate a release: BLOCKED counts as failure
CHECK_TIMEOUT=600 ./run_checks.sh  # tighter per-check timeout
```

`run_checks.sh` **refuses to start below 20 GB available RAM** (`MIN_AVAILABLE_GB`)
and exits 3. Do not lower it. See "Memory" below.

## Trend / regression detection

```bash
./trend.py                # latest.json vs the report before it
./trend.py --quiet        # only checks that moved
./trend.py --strict       # also gate on NEW FAIL (was BLOCKED, now broken)
./trend.py --json         # machine-readable (what watch.sh consumes)
./trend.py --current a.json --previous b.json
```

Categories: `REGRESSION` (PASS → anything else), `NEW FAIL` (BLOCKED/SKIP →
FAIL), `NEW PASS` (progress), `NEW CHECK`, `REMOVED`, `STILL BLOCKED`,
`STILL FAILING`, `UNCHANGED`.

Exit codes: `0` no regressions · `1` regression (so it gates automation) ·
`2` cannot compare (fewer than two reports).

## Watch mode

```bash
./watch.sh                       # every 15 minutes, forever
./watch.sh --interval 300
./watch.sh --once                # one cycle; exit 1 if it regressed
./watch.sh --iterations 5 --keep 10
```

Each cycle: memory pre-check → `run_checks.sh` (one at a time, never concurrent)
→ `trend.py --json` → one line appended to `reports/history.jsonl` →
`render_status.py`. Regressions are printed in a banner and, if
`WATCH_ON_REGRESSION` is set, that command is run as a notification hook.

Safe to leave running for hours:

- An `flock` on `reports/.watch.lock` means **one watcher, one run at a time**.
- It **raises** `run_checks.sh`'s memory floor to `--min-gb` and never lowers it;
  below the floor it skips the cycle and backs off exponentially (up to 8×).
- Report dirs are pruned to `--keep` (default 24, never deleting `latest.json`'s
  target); `history.jsonl` rotates at 2000 lines; `watch.log` rotates at 5 MB.
- `SIGTERM`/`SIGINT` terminate an in-flight `run_checks.sh` and exit cleanly.

Env: `WATCH_INTERVAL WATCH_KEEP WATCH_MIN_GB WATCH_MAX_HISTORY
WATCH_LOG_MAX_BYTES WATCH_ON_REGRESSION`.

## STATUS.md

```bash
./render_status.py            # rewrite the generated region
./render_status.py --check    # exit 1 if STATUS.md is stale
./render_status.py --stdout   # preview, write nothing
```

Only the region between the marker comments is rewritten:

```
<!-- BEGIN GENERATED: parts (render_status.py) -->
<!-- END GENERATED: parts -->
```

Prose outside the markers is preserved byte for byte. **Do not hand-edit the
table** — edit `parts.json` (part name, owner, check prefixes) and re-run.
A part with no check of its own can pin a `"state"` there.

## Where reports land

```
reports/<stamp>.json          one report per run  (stamp = 20260726T190005Z)
reports/<stamp>/<check>.log   full stdout+stderr of each check
reports/latest.json           symlink to the newest report
reports/history.jsonl         one line per watch cycle: counts + what changed
reports/history.jsonl.1       rotated archive
reports/watch.log[.1]         watch loop output, size capped
reports/.watch.lock           flock held by the running watcher
```

A `history.jsonl` line:

```json
{"available_gb":117,"changed":[{"category":"NEW PASS","check":"30_dgrut_native_ext.sh","from":"BLOCKED","to":"PASS"}],
 "duration_s":11,"regressions":[],"report":"20260726T193012Z","run_exit":0,
 "status":"ok","summary":{"blocked":5,"fail":0,"pass":2,"skip":0},"ts":"2026-07-26T19:30:23Z"}
```

`status` is one of `ok`, `failing`, `regression`, `skipped-low-memory`, `aborted`.

## Memory — read this before adding a check

This box is a DGX Spark GB10: **128 GB unified memory, no discrete VRAM.** GPU
allocations come out of the same pool as the OS. A resident vLLM server plus a
GUI Isaac Sim hard-locked the machine on 2026-07-26.

- Never lower `MIN_AVAILABLE_GB`.
- Never run two checks concurrently, and don't background work inside a check.
- Don't start an LLM/vLLM container while the harness or a build is running:
  `docker stop $(docker ps -q --filter name=vllm)`.
- Cap build parallelism inside checks (`MAX_JOBS`/`-j`), and prefer headless.
- `sudo` needs a password and is unavailable — build against what is on the box.
