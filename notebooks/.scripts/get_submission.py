#!/usr/bin/env python3
"""
get_submission.py — Extract recommended next points from executed week notebooks
                    and generate a cross-week results summary.

Usage:
    python get_submission.py <week>              # print all 8 function suggestions
    python get_submission.py <week> --write      # also update submissions/Week_<N>.md
    python get_submission.py <week> --fn 3       # single function only
    python get_submission.py --results           # print results summary to stdout
    python get_submission.py --results --write   # write to RESULTS.md

How it works:
    Parses the stored cell outputs of notebooks/week_<N>_function_<M>.ipynb and
    looks for the machine-readable tag inserted at the end of each notebook:

        SUBMISSION: x1-x2-...-xn

    If a notebook has not been executed (no outputs), the function is skipped
    with a warning.

    --results reads data/week_N/function_M/outputs.npy for all available weeks
    and writes results/RESULTS.md with best-known y per function per week,
    the current best point, and links to each week's submission file.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
DATA_DIR = NOTEBOOKS_DIR / "data"
RESULTS_DIR = REPO_ROOT

FUNC_DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


def _parse_notebooks_readme() -> tuple[dict, dict]:
    """Parse Surrogate and Active dims from the strategy table in notebooks/README.md.

    Returns (FUNC_SURROGATE, FUNC_ACTIVE_DIMS) dicts keyed by function number.
    Falls back to empty dicts (callers render '—') if the file or table is missing.
    """
    readme = NOTEBOOKS_DIR / "README.md"
    if not readme.exists():
        return {}, {}

    surrogate, active = {}, {}
    in_table = False
    for line in readme.read_text().splitlines():
        # Detect the strategy table header
        if re.search(r"\|\s*Fn\s*\|.*Surrogate.*Active dims", line):
            in_table = True
            continue
        if not in_table:
            continue
        # Stop at a blank line or a new heading
        if not line.strip() or line.startswith("#"):
            break
        # Skip separator rows (---|---...)
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        try:
            fn = int(parts[0])
        except ValueError:
            continue
        surrogate[fn] = parts[3]   # "Surrogate" column (0-indexed after Fn, Dims, Best y)
        active[fn] = parts[4]      # "Active dims" column
    return surrogate, active


FUNC_SURROGATE, FUNC_ACTIVE_DIMS = _parse_notebooks_readme()


# ---------------------------------------------------------------------------
# Submission extraction
# ---------------------------------------------------------------------------

def extract_submission(nb_path: Path) -> Optional[str]:
    """Return the SUBMISSION: tag value from executed notebook outputs, or None."""
    with open(nb_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") in ("stream", "display_data", "execute_result"):
                raw = output.get("text", output.get("data", {}).get("text/plain", ""))
                text = "".join(raw) if isinstance(raw, list) else raw
            match = re.search(r"SUBMISSION:\s*([\d.\-]+)", text)
            if match:
                return match.group(1).strip()
    return None


def format_submission(values_str: str, dims: int) -> str:
    """Validate and reformat submission string to 6dp, dash-separated."""
    parts = values_str.split("-")
    if len(parts) != dims:
        raise ValueError(f"Expected {dims} values, got {len(parts)}: {values_str}")
    return "-".join(f"{float(p):.6f}" for p in parts)


def get_week_submissions(week: int, functions: list) -> dict:
    results = {}
    for fn in functions:
        nb_path = NOTEBOOKS_DIR / f"week_{week}_function_{fn}.ipynb"
        if not nb_path.exists():
            print(f"  [WARN] Fn{fn}: notebook not found at {nb_path}", file=sys.stderr)
            results[fn] = None
            continue
        raw = extract_submission(nb_path)
        if raw is None:
            print(f"  [WARN] Fn{fn}: no SUBMISSION tag found — has the notebook been executed?",
                  file=sys.stderr)
            results[fn] = None
        else:
            try:
                results[fn] = format_submission(raw, FUNC_DIMS[fn])
            except ValueError as e:
                print(f"  [ERROR] Fn{fn}: {e}", file=sys.stderr)
                results[fn] = None
    return results


def update_submission_file(week: int, results: dict) -> None:
    submission_path = SUBMISSIONS_DIR / f"Week_{week}.md"
    if not submission_path.exists():
        lines = [f"# Week {week} Submissions:\n", "\n"]
        for fn in range(1, 9):
            dims = FUNC_DIMS[fn]
            lines.append(f"- Function {fn} ({dims} - D): \n")
        submission_path.write_text("".join(lines))

    content = submission_path.read_text()
    for fn, value in results.items():
        if value is None:
            continue
        dims = FUNC_DIMS[fn]
        pattern = rf"(- Function {fn} \({dims} - D\):).*"
        replacement = rf"\1 {value}"
        content, n = re.subn(pattern, replacement, content)
        if n == 0:
            print(f"  [WARN] Fn{fn}: could not find entry in {submission_path.name}", file=sys.stderr)

    submission_path.write_text(content)
    print(f"\nUpdated {submission_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def load_best_y(week: int, fn: int) -> Optional[float]:
    """Return best known y for a function at a given week, or None if data missing."""
    path = DATA_DIR / f"week_{week}" / f"function_{fn}" / "outputs.npy"
    if not path.exists():
        return None
    return float(np.load(path).max())


def load_best_point(week: int, fn: int) -> Optional[np.ndarray]:
    """Return the input point with the highest y for a function at a given week."""
    y_path = DATA_DIR / f"week_{week}" / f"function_{fn}" / "outputs.npy"
    x_path = DATA_DIR / f"week_{week}" / f"function_{fn}" / "inputs.npy"
    if not y_path.exists() or not x_path.exists():
        return None
    y = np.load(y_path)
    X = np.load(x_path)
    return X[y.argmax()]


def available_weeks() -> list:
    weeks = []
    for w in range(1, 14):
        if (DATA_DIR / f"week_{w}").exists():
            weeks.append(w)
    return weeks


def parse_submission_point(week: int, fn: int) -> Optional[str]:
    """Read the submitted point for fn in week from submissions/Week_N.md."""
    path = SUBMISSIONS_DIR / f"Week_{week}.md"
    if not path.exists():
        return None
    dims = FUNC_DIMS[fn]
    match = re.search(
        rf"- Function {fn} \({dims} - D\):\s*([\d.\-]+)",
        path.read_text()
    )
    return match.group(1).strip() if match else None


def generate_results(write: bool = False) -> None:

    weeks = available_weeks()
    if not weeks:
        print("No week data found.", file=sys.stderr)
        return

    latest_week = max(weeks)
    lines = []

    lines.append("# BBO Capstone — Results Summary\n\n")
    lines.append(f"_Last updated: {date.today().isoformat()} · Latest data: Week {latest_week}_\n\n")
    lines.append("> Auto-generated by `notebooks/.scripts/get_submission.py --results --write`\n\n")

    # --- Best y progress table ---
    lines.append("## Best known y per function per week\n\n")
    header = "| Fn | Dims |" + "".join(f" W{w} |" for w in weeks)
    separator = "|----|------|" + "".join("---------|" for _ in weeks)
    lines.append(header + "\n")
    lines.append(separator + "\n")

    for fn in range(1, 9):
        dims = FUNC_DIMS[fn]
        row = f"| {fn}  | {dims}D   |"
        for w in weeks:
            best = load_best_y(w, fn)
            row += f" {best:.3e} |" if best is not None else " —       |"
        lines.append(row + "\n")

    lines.append("\n")

    # --- Current best point per function ---
    lines.append(f"## Current best known points (Week {latest_week})\n\n")
    lines.append("| Fn | Dims | Surrogate | Active dims | Best y | Best x |\n")
    lines.append("|----|------|-----------|-------------|--------|--------|\n")

    for fn in range(1, 9):
        dims = FUNC_DIMS[fn]
        best_y = load_best_y(latest_week, fn)
        best_x = load_best_point(latest_week, fn)
        y_str = f"{best_y:.4e}" if best_y is not None else "—"
        x_str = "-".join(f"{v:.4f}" for v in best_x) if best_x is not None else "—"
        lines.append(
            f"| {fn}  | {dims}D   | {FUNC_SURROGATE.get(fn, '—')} "
            f"| {FUNC_ACTIVE_DIMS.get(fn, '—')} | {y_str} | `{x_str}` |\n"
        )

    lines.append("\n")

    # --- Submission audit trail ---
    lines.append("## Submission audit trail\n\n")
    lines.append("Links each submitted point to the week it was generated.\n\n")
    lines.append("| Week | Fn | Dims | Submitted point | Best y at submission |\n")
    lines.append("|------|----|------|-----------------|---------------------|\n")

    for w in weeks:
        for fn in range(1, 9):
            dims = FUNC_DIMS[fn]
            point = parse_submission_point(w, fn)
            best_y = load_best_y(w, fn)
            point_str = f"`{point}`" if point else "—"
            y_str = f"{best_y:.4e}" if best_y is not None else "—"
            submission_link = f"[Week_{w}.md](../submissions/Week_{w}.md)"
            lines.append(f"| {submission_link} | {fn} | {dims}D | {point_str} | {y_str} |\n")

    lines.append("\n")

    content = "".join(lines)
    if write:
        out_path = RESULTS_DIR / "RESULTS.md"
        out_path.write_text(content)
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract notebook submission points and generate results summary.")
    parser.add_argument("week", type=int, nargs="?", help="Week number (e.g. 6)")
    parser.add_argument("--write", action="store_true",
                        help="Write results to submissions/Week_<N>.md")
    parser.add_argument("--fn", type=int, default=None,
                        help="Single function number (1-8). Defaults to all.")
    parser.add_argument("--results", action="store_true",
                        help="Print results summary (add --write to save to RESULTS.md)")
    args = parser.parse_args()

    if args.results:
        generate_results(write=args.write)
        return

    if args.week is None:
        parser.error("week is required unless --results is used")

    functions = [args.fn] if args.fn else list(range(1, 9))
    print(f"# Week {args.week} Submissions:\n")

    submissions = get_week_submissions(args.week, functions)

    for fn in functions:
        val = submissions[fn]
        dims = FUNC_DIMS[fn]
        status = val if val else "— not available"
        print(f"- Function {fn} ({dims} - D): {status}")

    if args.write:
        update_submission_file(args.week, submissions)


if __name__ == "__main__":
    main()
