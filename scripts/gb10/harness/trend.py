#!/usr/bin/env python3
"""Compare two harness reports and classify every check's movement.

Reads reports/latest.json (or --current) against the previous report in
reports/ (or --previous) and classifies each check:

  REGRESSION    PASS -> anything else.  The one that matters.  Exits non-zero.
  NEW FAIL      was not PASS, is now FAIL/TIMEOUT (built, but broken).
  NEW PASS      is now PASS and was not.  This is long-horizon progress.
  NEW CHECK     did not exist in the previous report, is not PASS.
  REMOVED       existed in the previous report, gone now.
  STILL BLOCKED BLOCKED in both.  Part not built yet.
  STILL FAILING FAIL/TIMEOUT in both.
  UNCHANGED     same status in both (PASS->PASS, SKIP->SKIP).

Exit codes:
  0  no regressions
  1  at least one regression (or, with --strict, a NEW FAIL as well)
  2  could not compare (no reports, or only one report exists)

Usage:
  ./trend.py                                  # latest vs the one before it
  ./trend.py --json                           # machine-readable, for watch.sh
  ./trend.py --current a.json --previous b.json
  ./trend.py --strict                         # also gate on NEW FAIL
  ./trend.py --quiet                          # only print movement, not UNCHANGED
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = HARNESS_DIR / "reports"

STAMP_RE = re.compile(r"^\d{8}T\d{6}Z\.json$")

PASSING = {"PASS"}
BROKEN = {"FAIL", "TIMEOUT"}
NOT_BUILT = {"BLOCKED"}

# Ordering used when printing sections: loudest first.
CATEGORY_ORDER = [
    "REGRESSION",
    "NEW FAIL",
    "NEW PASS",
    "REMOVED",
    "NEW CHECK",
    "STILL FAILING",
    "STILL BLOCKED",
    "UNCHANGED",
]

MOVEMENT = {"REGRESSION", "NEW FAIL", "NEW PASS", "REMOVED", "NEW CHECK"}


def die(msg: str, code: int = 2) -> None:
    print(f"trend: {msg}", file=sys.stderr)
    sys.exit(code)


def load_report(path: Path) -> dict:
    try:
        with path.open() as fh:
            data = json.load(fh)
    except FileNotFoundError:
        die(f"no such report: {path}")
    except json.JSONDecodeError as exc:
        die(f"malformed report {path}: {exc}")
    if not isinstance(data, dict) or "results" not in data:
        die(f"report {path} has no 'results' array")
    return data


def report_files(reports_dir: Path) -> list:
    """All timestamped reports, oldest first. Excludes the latest.json symlink."""
    if not reports_dir.is_dir():
        return []
    return sorted(p for p in reports_dir.iterdir() if STAMP_RE.match(p.name))


def resolve_pair(args) -> tuple:
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else REPORTS_DIR

    if args.current:
        current = Path(args.current).resolve()
    else:
        link = reports_dir / "latest.json"
        if link.exists():
            current = link.resolve()
        else:
            files = report_files(reports_dir)
            if not files:
                die(f"no reports in {reports_dir} -- run ./run_checks.sh first")
            current = files[-1]

    if args.previous:
        previous = Path(args.previous).resolve()
        return current, previous

    files = report_files(reports_dir)
    earlier = [p for p in files if p.name < current.name]
    if not earlier:
        die(
            f"only one report ({current.name}) -- nothing to compare against yet. "
            "Run ./run_checks.sh again to establish a trend."
        )
    return current, earlier[-1]


def index(report: dict) -> dict:
    out = {}
    for row in report.get("results", []):
        name = row.get("check")
        if name:
            out[name] = row
    return out


def classify(prev: str|None, cur: str|None) -> str:
    if prev is None:
        return "NEW PASS" if cur in PASSING else "NEW CHECK"
    if cur is None:
        return "REMOVED"
    if prev in PASSING and cur not in PASSING:
        return "REGRESSION"
    if cur in PASSING and prev not in PASSING:
        return "NEW PASS"
    if cur in BROKEN and prev in BROKEN:
        return "STILL FAILING"
    if cur in BROKEN:
        # prev was BLOCKED or SKIP: something got built and is now broken.
        return "NEW FAIL"
    if cur in NOT_BUILT and prev in NOT_BUILT:
        return "STILL BLOCKED"
    if cur == prev:
        return "UNCHANGED"
    return "UNCHANGED"


def build_diff(cur_report: dict, prev_report: dict) -> list:
    cur, prev = index(cur_report), index(prev_report)
    rows = []
    for name in sorted(set(cur) | set(prev)):
        c = cur.get(name)
        p = prev.get(name)
        c_status = c["status"] if c else None
        p_status = p["status"] if p else None
        rows.append(
            {
                "check": name,
                "previous": p_status,
                "current": c_status,
                "category": classify(p_status, c_status),
                "detail": (c or p or {}).get("detail", ""),
                "seconds": (c or {}).get("seconds"),
                "log": (c or {}).get("log"),
            }
        )
    return rows


def summarize(rows: list) -> dict:
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def render(cur_path, prev_path, cur_report, prev_report, rows, counts, quiet):
    def stamp(report, path):
        return report.get("timestamp") or path.stem

    cs, ps = stamp(cur_report, cur_path), stamp(prev_report, prev_path)
    print(f"trend: {ps}  ->  {cs}")

    def summary_line(rep):
        s = rep.get("summary", {})
        return (
            f"PASS {s.get('pass', 0)}  FAIL {s.get('fail', 0)}  "
            f"BLOCKED {s.get('blocked', 0)}  SKIP {s.get('skip', 0)}"
        )

    print(f"  previous: {summary_line(prev_report)}")
    print(f"  current:  {summary_line(cur_report)}")
    print("-" * 78)

    printed = False
    for category in CATEGORY_ORDER:
        group = [r for r in rows if r["category"] == category]
        if not group:
            continue
        if quiet and category not in MOVEMENT:
            continue
        printed = True
        print(f"{category}  ({len(group)})")
        for r in group:
            transition = f"{r['previous'] or '-'} -> {r['current'] or '-'}"
            print(f"    {r['check']:<32} {transition:<20} {r['detail'][:60]}")
        print()

    if not printed:
        print("no movement.\n")

    regressions = counts.get("REGRESSION", 0)
    new_fails = counts.get("NEW FAIL", 0)
    print("-" * 78)
    if regressions:
        print(f"!! {regressions} REGRESSION(S) -- a check that used to PASS no longer does.")
        for r in rows:
            if r["category"] == "REGRESSION":
                print(f"!!   {r['check']}: {r['previous']} -> {r['current']}")
                if r.get("log"):
                    print(f"!!     log: {r['log']}")
    if new_fails:
        print(f" * {new_fails} NEW FAIL(S) -- newly built and already broken.")
    prog = counts.get("NEW PASS", 0)
    if prog:
        print(f" + {prog} NEW PASS -- progress.")
    if not regressions and not new_fails:
        print("no regressions.")


def main() -> int:
    ap = argparse.ArgumentParser(description="classify movement between two harness reports")
    ap.add_argument(
        "--current", help="report JSON to treat as current (default reports/latest.json)"
    )
    ap.add_argument(
        "--previous", help="report JSON to compare against (default: the one before current)"
    )
    ap.add_argument("--reports-dir", help=f"reports directory (default {REPORTS_DIR})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--quiet", action="store_true", help="only print checks that moved")
    ap.add_argument("--strict", action="store_true", help="also exit non-zero on NEW FAIL")
    args = ap.parse_args()

    cur_path, prev_path = resolve_pair(args)
    cur_report, prev_report = load_report(cur_path), load_report(prev_path)
    rows = build_diff(cur_report, prev_report)
    counts = summarize(rows)

    if args.json:
        print(
            json.dumps(
                {
                    "current": cur_report.get("timestamp") or cur_path.stem,
                    "previous": prev_report.get("timestamp") or prev_path.stem,
                    "current_path": str(cur_path),
                    "previous_path": str(prev_path),
                    "current_summary": cur_report.get("summary", {}),
                    "counts": counts,
                    "changes": [r for r in rows if r["category"] in MOVEMENT],
                    "regressions": [r for r in rows if r["category"] == "REGRESSION"],
                },
                indent=None,
            )
        )
    else:
        render(cur_path, prev_path, cur_report, prev_report, rows, counts, args.quiet)

    bad = counts.get("REGRESSION", 0)
    if args.strict:
        bad += counts.get("NEW FAIL", 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
