#!/usr/bin/env python3
"""Regenerate the "Parts" table in STATUS.md from reports/latest.json.

The human-readable tracker must not be able to drift from what the checks
actually report, so the table is generated, never hand-edited. Everything
outside the marker comments is hand-written prose and is preserved byte for
byte.

  <!-- BEGIN GENERATED: parts (render_status.py) -->
  ... table ...
  <!-- END GENERATED: parts -->

The part -> check-prefix map lives in parts.json. Add a part there, not here,
and not in the table.

Usage:
  ./render_status.py              # rewrite the generated region in STATUS.md
  ./render_status.py --check      # exit 1 if STATUS.md is stale (CI gate)
  ./render_status.py --stdout     # print the generated block, touch nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = HARNESS_DIR / "reports"
STATUS_MD = HARNESS_DIR / "STATUS.md"
PARTS_JSON = HARNESS_DIR / "parts.json"

BEGIN = "<!-- BEGIN GENERATED: parts (render_status.py) -->"
END = "<!-- END GENERATED: parts -->"

BROKEN = {"FAIL", "TIMEOUT"}


def load_json(path: Path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"render_status: missing {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"render_status: malformed {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def part_state(part: dict, results: dict) -> tuple:
    """Return (state_token, note) derived from the checks a part owns."""
    prefixes = part.get("prefixes") or []
    if not prefixes:
        return part.get("state", "—"), part.get("note", "")

    matched = {
        name: row
        for name, row in results.items()
        if any(name.startswith(p) for p in prefixes)
    }
    missing = [p for p in prefixes if not any(n.startswith(p) for n in results)]

    if not matched:
        return "PENDING", f"no check written yet ({', '.join(prefixes)})"

    statuses = [row["status"] for row in matched.values()]
    n_pass = sum(1 for s in statuses if s == "PASS")
    note = f"{n_pass}/{len(statuses)} pass"
    if missing:
        note += f", {len(missing)} check(s) not written"

    if any(s in BROKEN for s in statuses):
        return "FAIL", note
    if any(s == "BLOCKED" for s in statuses):
        return "BLOCKED", note
    if missing:
        return "BLOCKED", note
    if all(s in ("PASS", "SKIP") for s in statuses) and n_pass:
        return "PASS", note
    if all(s == "SKIP" for s in statuses):
        return "SKIP", note
    return "BLOCKED", note


def format_state(state: str) -> str:
    if state in ("PASS", "FAIL"):
        return f"**{state}**"
    return state


def build_block(report: dict, parts: list) -> str:
    results = {r["check"]: r for r in report.get("results", [])}
    stamp = report.get("timestamp", "unknown")
    summary = report.get("summary", {})

    lines = [
        BEGIN,
        f"<!-- generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"from reports/{stamp}.json -- do not edit by hand, run ./render_status.py -->",
        "",
        "| # | Part | Checks | Owner | State |",
        "| :-- | :---- | :---- | :---- | :---- |",
    ]
    for part in parts:
        state, note = part_state(part, results)
        prefixes = ", ".join(f"`{p}`" for p in part.get("prefixes") or []) or "—"
        cell = format_state(state)
        if note:
            cell = f"{cell} <br><sub>{note}</sub>"
        lines.append(
            f"| {part['id']} | {part['name']} | {prefixes} | {part.get('owner', '—')} | {cell} |"
        )

    lines += [
        "",
        f"Source: `reports/{stamp}.json` — "
        f"PASS {summary.get('pass', 0)} · FAIL {summary.get('fail', 0)} · "
        f"BLOCKED {summary.get('blocked', 0)} · SKIP {summary.get('skip', 0)}. "
        f"Regenerate with `./render_status.py`; trend with `./trend.py`.",
        END,
    ]
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    """Replace the marked region, or install markers around the current Parts table."""
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head + block + tail

    # First run: markers not installed yet. Replace the body of the "## Parts"
    # section (heading kept, prose after the next H2 kept).
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## parts"):
            start = i
            break
    if start is None:
        raise SystemExit(
            "render_status: STATUS.md has neither generated markers nor a '## Parts' "
            "heading; cannot decide what to replace. Add the markers by hand."
        )
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    new = lines[: start + 1] + ["", block, ""] + lines[end:]
    return "\n".join(new) + ("\n" if text.endswith("\n") else "")


def atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".status.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="regenerate the Parts table in STATUS.md")
    ap.add_argument("--report", help="report JSON (default reports/latest.json)")
    ap.add_argument("--status", help=f"markdown file to update (default {STATUS_MD})")
    ap.add_argument("--parts", help=f"part map (default {PARTS_JSON})")
    ap.add_argument("--stdout", action="store_true", help="print the block, write nothing")
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = ap.parse_args()

    report_path = Path(args.report).resolve() if args.report else (REPORTS_DIR / "latest.json")
    status_path = Path(args.status).resolve() if args.status else STATUS_MD
    parts_path = Path(args.parts).resolve() if args.parts else PARTS_JSON

    report = load_json(report_path)
    parts = load_json(parts_path).get("parts", [])
    block = build_block(report, parts)

    if args.stdout:
        print(block)
        return 0

    original = status_path.read_text() if status_path.exists() else "# GB10 build-out status\n\n## Parts\n"
    updated = splice(original, block)

    if args.check:
        # Ignore the generation timestamp line, which always differs.
        def strip_ts(s):
            return "\n".join(l for l in s.splitlines() if not l.startswith("<!-- generated "))

        if strip_ts(original) != strip_ts(updated):
            print(f"render_status: {status_path} is STALE -- run ./render_status.py", file=sys.stderr)
            return 1
        print(f"render_status: {status_path} is up to date")
        return 0

    if updated != original:
        atomic_write(status_path, updated)
        print(f"render_status: rewrote Parts table in {status_path} from {report_path}")
    else:
        print(f"render_status: {status_path} already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
