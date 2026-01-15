#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# AI Assistance: Yes (GPT-5.2)
# Is_Check: Yes
# Description: group-aware evaluator (unchanged scoring) with acc-style reporting + saving
#              + sub-dimension (TP/VP/SP/IA/PU/GU/SR/FP) micro scores
# ================================================================
"""
Group-aware evaluator for RTV-Bench style JSON results.

Scoring protocol (DO NOT CHANGE, preserved from your original test2.py):
- Group items by questionID: q-group-<groupid>(-group)?-<qtype>  where qtype in {0,1,2}
- basic_pass = (all q0 correct if exists) AND (all q1 correct if exists)
  - Missing q0 or q1 is treated as pass (True)
- Group contributes q2 correctness ONLY IF basic_pass; otherwise q2 score is 0.
- For each group, use group['q2'][0]['type'] as the group type key (full_type).
- Aggregate by full_type:
  - total_score += sum(q2.correct) if basic_pass else 0
  - total_q2 += len(q2)
  - type_stats[full_type].total_score += ...
  - type_stats[full_type].total_q2 += ...

Acc-style reporting:
- Evaluate one or multiple JSON files (supports glob patterns).
- Print per-file report and a final batch summary table.
- Save per-file reports and batch summaries under a relative output directory.

Added reporting (NO scoring change):
- Per-dimension (Reasoning/Understanding/Perception) micro scores
- Per-subdimension (TP/VP/SP/IA/PU/GU/SR/FP) micro scores
- Save subdimension columns into batch_summary.tsv/txt

Outputs:
- Per-file report:  <outdir>/<basename>.txt
- Batch summary:    <outdir>/batch_summary.tsv
- Batch summary:    <outdir>/batch_summary.txt

Default outdir:
- ./eval_statics/scores
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VALID_QTYPES = ("0", "1", "2")
REQUIRED_FIELDS = ("questionID", "type", "pred")

MAIN_CATEGORIES = ("Object", "Action", "Event")
DIM_CATEGORIES = ("Reasoning", "Understanding", "Perception")

# NEW: sub-dim categories for reporting
SUBDIM_CATEGORIES = ("TP", "VP", "SP", "IA", "PU", "GU", "SR", "FP")

DEFAULT_OUTDIR = os.path.join(".", "eval_statics/scores")

# NOTE: keep your original group-id parsing behavior (pattern + logic),
# but compiled for speed and cleanliness.
GROUP_ID_RE = re.compile(
    r"^q-group-([a-z0-9]+)-([012])(?:-[^-]+)*$",
    re.IGNORECASE,
)

# NEW: robust token match for sub-dimension tags
_SUBDIM_PATTERNS = {k: re.compile(rf"(^|[-_]){k}($|[-_])") for k in SUBDIM_CATEGORIES}


# -----------------------------------------------------------------------------
# Utilities (generic)
# -----------------------------------------------------------------------------

def expand_inputs(inputs: Iterable[str]) -> List[str]:
    """Expand file paths and glob patterns into a sorted unique list."""
    paths: List[str] = []
    for x in inputs:
        if any(ch in x for ch in ("*", "?", "[")):
            paths.extend(glob.glob(x))
        else:
            paths.append(x)
    return sorted(dict.fromkeys(paths))


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return normalized relative path."""
    os.makedirs(path, exist_ok=True)
    return os.path.normpath(path)


def write_text(path: str, text: str) -> None:
    """Write UTF-8 text to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_rate(correct: int, total: int) -> Optional[float]:
    """Return accuracy in percent; None if total is zero."""
    return (correct / total * 100.0) if total else None


def macro_avg(values: Iterable[Optional[float]]) -> Optional[float]:
    """Macro average over non-None values."""
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt_stat_line(title: str, correct: int, total: int) -> str:
    """Format a single stats line (micro accuracy)."""
    if total == 0:
        return f"{title.ljust(55)}: No valid data"
    rate = correct / total * 100
    return f"{title.ljust(55)}: {correct:>4}/{total:<4} | {rate:>6.2f}%"


def fmt_rate(x: Optional[float], width: int = 6) -> str:
    """Format an accuracy percentage for summary tables."""
    return ("N/A".rjust(width)) if x is None else f"{x:>{width}.2f}"


# -----------------------------------------------------------------------------
# Parsing helpers (dataset-specific)
# -----------------------------------------------------------------------------

def parse_group_id(qid: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse (group_id, qtype) from questionID.

    Supported formats:
      - q-group-<groupid>-<qtype>
      - q-group-<groupid>-<qtype>-<any-suffix...>

    Example:
      q-group-9eq4fqfqg-2-option2  -> ("9eq4fqfqg", "2")
    """
    if not isinstance(qid, str):
        return None, None

    m = GROUP_ID_RE.match(qid.strip())
    if not m:
        return None, None

    group_id, qtype = m.group(1), m.group(2)
    return group_id, qtype


def parse_main_category(type_str: Any) -> Optional[str]:
    """Parse main category (Object/Action/Event) from `type` prefix."""
    if not isinstance(type_str, str):
        return None
    for cat in MAIN_CATEGORIES:
        if type_str.startswith(f"{cat}-"):
            return cat
    return None


def parse_dimension(type_str: Any) -> Optional[str]:
    """
    Parse dimension group for your pivot table.

    IMPORTANT: This does NOT change scoring. It is only for reporting:
    - If type contains SR/FP -> Reasoning
    - IA/PU/GU -> Understanding
    - TP/VP/SP -> Perception
    """
    if not isinstance(type_str, str):
        return None
    if ("SR" in type_str) or ("FP" in type_str):
        return "Reasoning"
    if ("IA" in type_str) or ("PU" in type_str) or ("GU" in type_str):
        return "Understanding"
    if ("TP" in type_str) or ("VP" in type_str) or ("SP" in type_str):
        return "Perception"
    return None


def parse_subdimension(type_str: Any) -> Optional[str]:
    """
    Parse sub-dimension tag for reporting only.
    Expected keys: TP/VP/SP/IA/PU/GU/SR/FP

    NOTE: This does NOT change scoring. It only aggregates on top of type_stats.
    """
    if not isinstance(type_str, str):
        return None

    s = type_str.strip()

    # Prefer token match if type strings are like "Object-TP-xxx" or "Object_TP_xxx"
    for k in SUBDIM_CATEGORIES:
        if _SUBDIM_PATTERNS[k].search(s):
            return k

    # Fallback: substring containment (disabled by default to avoid false positives).
    # Uncomment if your type strings do not use '-'/'_' separators.
    # for k in SUBDIM_CATEGORIES:
    #     if k in s:
    #         return k

    return None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Counters:
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool, n: int = 1) -> None:
        self.total += n
        self.correct += int(is_correct) * n


def make_counters() -> defaultdict:
    return defaultdict(Counters)


def make_nested_counters() -> defaultdict:
    return defaultdict(lambda: defaultdict(Counters))


# -----------------------------------------------------------------------------
# Core: unchanged scoring logic, acc-style output
# -----------------------------------------------------------------------------

def get_is_correct(item: Dict[str, Any]) -> Optional[bool]:
    """
    Keep the exact semantics you used:
    - Use item['correct'] if present
    - If missing and pred == 'ERROR' -> correct=False
    - Else invalid for scoring
    """
    if "correct" in item:
        return bool(item["correct"])
    if item.get("pred") == "ERROR":
        return False
    return None


def evaluate_one_file(path: str) -> Dict[str, Any]:
    """
    Evaluate one JSON file with your original group-aware scoring.

    Returns a state dict for reporting:
    - totals: total_score, total_q2, valid_groups
    - type_stats: full_type -> {total_groups,total_score,total_q2}
    - pivot (dimension x main) computed from type_stats (no scoring change)
    - subdim totals (TP/VP/SP/...) computed from type_stats (no scoring change)
    - invalid logs
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON must be a list, got {type(data)}: {path}")

    # Original containers (preserve semantics)
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    error_log: List[str] = []

    type_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total_groups": 0,
        "total_score": 0,
        "total_q2": 0,
    })

    # -------------------------
    # Data grouping (preserve)
    # -------------------------
    for item in data:
        try:
            if any(k not in item for k in REQUIRED_FIELDS):
                error_log.append(f"Missing required fields: {json.dumps(item, ensure_ascii=False)[:120]}...")
                continue

            is_correct = get_is_correct(item)
            if is_correct is None:
                error_log.append(
                    f"Missing `correct` and pred != ERROR: {json.dumps(item, ensure_ascii=False)[:120]}..."
                )
                continue

            gid, qtype = parse_group_id(item["questionID"])
            if not gid or qtype not in VALID_QTYPES:
                error_log.append(f"Invalid group questionID: {item['questionID']}")
                continue

            # Store (do not modify scoring)
            item["correct"] = bool(is_correct)
            groups[gid][f"q{qtype}"].append(item)

        except Exception as e:
            error_log.append(f"Item processing error: {e} | item={json.dumps(item, ensure_ascii=False)[:120]}...")

    # -------------------------
    # Scoring (preserve)
    # -------------------------
    total_score = 0
    total_q2 = 0
    valid_groups = 0

    for gid, g in groups.items():
        try:
            q0_valid = all(x["correct"] for x in g["q0"]) if len(g["q0"]) > 0 else True
            q1_valid = all(x["correct"] for x in g["q1"]) if len(g["q1"]) > 0 else True
            basic_pass = q0_valid and q1_valid

            if ("q2" not in g) or (len(g["q2"]) == 0):
                error_log.append(f"Group {gid} missing q2")
                continue

            full_type = g["q2"][0]["type"]
            stat = type_stats[full_type]

            valid_groups += 1
            current_q2_count = len(g["q2"])
            current_score = sum(x["correct"] for x in g["q2"]) if basic_pass else 0

            total_score += current_score
            total_q2 += current_q2_count

            stat["total_groups"] += 1
            stat["total_score"] += current_score
            stat["total_q2"] += current_q2_count

        except KeyError as e:
            error_log.append(f"Group {gid} missing key: {e}")
        except Exception as e:
            error_log.append(f"Group {gid} scoring error: {e}")

    # -------------------------
    # Build pivot tables from type_stats (reporting only; no scoring change)
    # -------------------------
    pivot = make_nested_counters()  # dim -> main -> Counters
    dim_totals = make_counters()    # dim -> Counters
    main_totals = make_counters()   # main -> Counters
    subdim_totals = make_counters() # subdim -> Counters (TP/VP/SP/...)
    grand_total = Counters()

    for full_type, stat in type_stats.items():
        main = parse_main_category(full_type) or "Unknown"
        dim = parse_dimension(full_type) or "Unknown"
        sub = parse_subdimension(full_type) or "Unknown"

        c = stat["total_score"]
        t = stat["total_q2"]

        pivot[dim][main].correct += c
        pivot[dim][main].total += t

        dim_totals[dim].correct += c
        dim_totals[dim].total += t

        main_totals[main].correct += c
        main_totals[main].total += t

        subdim_totals[sub].correct += c
        subdim_totals[sub].total += t

        grand_total.correct += c
        grand_total.total += t

    return {
        "path": path,
        "totals": {
            "valid_groups": valid_groups,
            "total_score": total_score,
            "total_q2": total_q2,
        },
        "type_stats": type_stats,
        "pivot": {
            "cell": pivot,
            "dim_totals": dim_totals,
            "main_totals": main_totals,
            "subdim_totals": subdim_totals,  # NEW
            "grand_total": grand_total,
        },
        "errors": error_log,
    }


# -----------------------------------------------------------------------------
# Reporting (acc-style)
# -----------------------------------------------------------------------------

def build_pivot_table_text(pivot_state: Dict[str, Any]) -> str:
    """Render the dimension x main-category pivot table."""
    cell = pivot_state["cell"]
    dim_totals = pivot_state["dim_totals"]
    main_totals = pivot_state["main_totals"]
    grand_total: Counters = pivot_state["grand_total"]

    # Prefer these dims, then fall back to unknown if present
    dims_order = list(DIM_CATEGORIES)
    if "Unknown" in cell and "Unknown" not in dims_order:
        dims_order.append("Unknown")

    # Prefer these mains, then fall back to unknown if present
    mains_order = list(MAIN_CATEGORIES)
    if "Unknown" in main_totals and "Unknown" not in mains_order:
        mains_order.append("Unknown")

    lines: List[str] = []
    lines.append("\n[Group-aware Q2 Accuracy Pivot]")
    header = f"{'Dimension':<20} | " + " | ".join(f"{m:<16}" for m in mains_order) + " | " + f"{'Total':<16}"
    lines.append(header)
    lines.append("-" * len(header))

    for dim in dims_order:
        row = [f"{dim:<20}"]
        for m in mains_order:
            c = cell[dim].get(m, Counters())
            if c.total:
                row.append(f"{c.correct:3}/{c.total:3} ({(c.correct / c.total * 100):5.1f}%)".ljust(16))
            else:
                row.append("N/A".ljust(16))
        dt = dim_totals.get(dim, Counters())
        if dt.total:
            row.append(f"{dt.correct:3}/{dt.total:3} ({(dt.correct / dt.total * 100):5.1f}%)".ljust(16))
        else:
            row.append("N/A".ljust(16))
        lines.append(" | ".join(row))

    lines.append("-" * len(header))
    total_row = [f"{'Total':<20}"]
    for m in mains_order:
        mt = main_totals.get(m, Counters())
        if mt.total:
            total_row.append(f"{mt.correct:3}/{mt.total:3} ({(mt.correct / mt.total * 100):5.1f}%)".ljust(16))
        else:
            total_row.append("N/A".ljust(16))
    if grand_total.total:
        total_row.append(
            f"{grand_total.correct:3}/{grand_total.total:3} ({(grand_total.correct / grand_total.total * 100):5.1f}%)".ljust(16)
        )
    else:
        total_row.append("N/A".ljust(16))
    lines.append(" | ".join(total_row))

    return "\n".join(lines) + "\n"


def build_subdim_table_text(subdim_totals: Dict[str, Counters]) -> str:
    """Render sub-dimension micro accuracies table."""
    lines: List[str] = []
    lines.append("\n[Sub-dimension Micro Accuracy (from group-aware q2)]")

    # show fixed order first
    for sd in SUBDIM_CATEGORIES:
        c = subdim_totals[sd].correct if sd in subdim_totals else 0
        t = subdim_totals[sd].total if sd in subdim_totals else 0
        lines.append(fmt_stat_line(f"Sub-dim {sd} (micro)", c, t))

    # show Unknown if exists with data
    if "Unknown" in subdim_totals and subdim_totals["Unknown"].total > 0:
        u = subdim_totals["Unknown"]
        lines.append(fmt_stat_line("Sub-dim Unknown (micro)", u.correct, u.total))

    return "\n".join(lines) + "\n"


def build_type_lines(type_stats: Dict[str, Dict[str, int]]) -> List[str]:
    """
    Render per-type lines in the same format your original code produced:
    <type_name.ljust(55)> <score>/<total> (<ratio:.1%>)
    """
    lines: List[str] = []
    for type_name, stat in sorted(type_stats.items(), key=lambda x: x[0]):
        if stat["total_q2"] > 0:
            ratio = stat["total_score"] / stat["total_q2"]
            lines.append(f"{type_name.ljust(55)} {stat['total_score']}/{stat['total_q2']} ({ratio:.1%})")
        else:
            lines.append(f"{type_name.ljust(55)} No valid data")
    return lines


def build_report_text(state: Dict[str, Any]) -> str:
    """Build the full per-file report text."""
    path = state["path"]
    totals = state["totals"]
    type_stats = state["type_stats"]
    errors = state["errors"]

    total_score = totals["total_score"]
    total_q2 = totals["total_q2"]
    valid_groups = totals["valid_groups"]

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"File: {path}")
    lines.append("-" * 80)

    lines.append("\n[Global (Group-aware Q2 Scoring)]")
    lines.append(f"Valid groups: {valid_groups}")
    lines.append(fmt_stat_line("Group-aware Q2 accuracy (micro)", total_score, total_q2))

    # Macro averages computed on top of unchanged type_stats totals
    pivot = state["pivot"]
    main_totals = pivot["main_totals"]
    main_macro = macro_avg(safe_rate(main_totals[m].correct, main_totals[m].total) for m in MAIN_CATEGORIES)
    if main_macro is not None:
        lines.append(f"{'Main Macro Avg (from group-aware q2)'.ljust(55)}: {main_macro:>6.2f}%")

    dim_totals = pivot["dim_totals"]
    dim_macro = macro_avg(safe_rate(dim_totals[d].correct, dim_totals[d].total) for d in DIM_CATEGORIES)
    if dim_macro is not None:
        lines.append(f"{'Dimension Macro Avg (from group-aware q2)'.ljust(55)}: {dim_macro:>6.2f}%")

    # NEW: sub-dimension micro accuracies
    subdim_totals = pivot.get("subdim_totals", make_counters())
    lines.append(build_subdim_table_text(subdim_totals).rstrip("\n"))

    # Per-type breakdown (same format as your original)
    lines.append("\n[Per-type breakdown (unchanged)]")
    lines.extend(build_type_lines(type_stats))

    # Pivot table
    lines.append(build_pivot_table_text(pivot).rstrip("\n"))

    # Data quality
    lines.append("\n" + "-" * 80)
    lines.append("[Data quality]")
    lines.append(f"Errors logged: {len(errors)}")
    for e in errors[:10]:
        lines.append(f"- {e}")
    if len(errors) > 10:
        lines.append(f"... ({len(errors) - 10} more)")

    return "\n".join(lines) + "\n"


def summarize_for_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a batch summary row (acc-style)."""
    totals = state["totals"]
    pivot = state["pivot"]

    overall = safe_rate(totals["total_score"], totals["total_q2"])

    main_totals = pivot["main_totals"]
    dim_totals = pivot["dim_totals"]

    main_macro = macro_avg(safe_rate(main_totals[m].correct, main_totals[m].total) for m in MAIN_CATEGORIES)
    dim_macro = macro_avg(safe_rate(dim_totals[d].correct, dim_totals[d].total) for d in DIM_CATEGORIES)

    # NEW: sub-dimension micro accuracies
    subdim_totals = pivot.get("subdim_totals", {})
    subdim_rates: Dict[str, Optional[float]] = {
        sd: safe_rate(subdim_totals[sd].correct, subdim_totals[sd].total) if sd in subdim_totals else None
        for sd in SUBDIM_CATEGORIES
    }

    row: Dict[str, Any] = {
        "file": os.path.basename(state["path"]),
        "overall": overall,
        "main_macro": main_macro,
        "dim_macro": dim_macro,
        "valid_groups": totals["valid_groups"],
        "total_q2": totals["total_q2"],
        "errors": len(state["errors"]),
        "path": os.path.relpath(state["path"]),
    }
    row.update({f"sub_{k}": v for k, v in subdim_rates.items()})
    return row


def write_batch_summaries(outdir: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Write batch summary to TSV and TXT files."""
    out_tsv = os.path.join(outdir, "batch_summary.tsv")
    out_txt = os.path.join(outdir, "batch_summary.txt")

    headers = [
        "file",
        "overall",
        "main_macro",
        "dim_macro",
        # NEW: sub-dimension columns
        "sub_TP", "sub_VP", "sub_SP",
        "sub_IA", "sub_PU", "sub_GU",
        "sub_SR", "sub_FP",
        "valid_groups",
        "total_q2",
        "errors",
        "path",
    ]

    def rfmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join([
                r["file"],
                rfmt(r.get("overall")),
                rfmt(r.get("main_macro")),
                rfmt(r.get("dim_macro")),
                rfmt(r.get("sub_TP")),
                rfmt(r.get("sub_VP")),
                rfmt(r.get("sub_SP")),
                rfmt(r.get("sub_IA")),
                rfmt(r.get("sub_PU")),
                rfmt(r.get("sub_GU")),
                rfmt(r.get("sub_SR")),
                rfmt(r.get("sub_FP")),
                str(r.get("valid_groups", 0)),
                str(r.get("total_q2", 0)),
                str(r.get("errors", 0)),
                r.get("path", ""),
            ]) + "\n")

    lines = [
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 80,
        "Batch Summary (Accuracy in %)",
        "-" * 80,
        "\t".join(headers),
    ]
    for r in rows:
        lines.append("\t".join([
            r["file"],
            rfmt(r.get("overall")),
            rfmt(r.get("main_macro")),
            rfmt(r.get("dim_macro")),
            rfmt(r.get("sub_TP")),
            rfmt(r.get("sub_VP")),
            rfmt(r.get("sub_SP")),
            rfmt(r.get("sub_IA")),
            rfmt(r.get("sub_PU")),
            rfmt(r.get("sub_GU")),
            rfmt(r.get("sub_SR")),
            rfmt(r.get("sub_FP")),
            str(r.get("valid_groups", 0)),
            str(r.get("total_q2", 0)),
            str(r.get("errors", 0)),
            r.get("path", ""),
        ]))
    write_text(out_txt, "\n".join(lines) + "\n")
    return out_tsv, out_txt


def print_batch_summary(rows: List[Dict[str, Any]]) -> None:
    """Print batch summary table to stdout."""
    if not rows:
        return

    # Keep console concise: include TP/VP/SP only (adjust if you want all sub-dims)
    headers = [
        "file", "overall", "main_macro", "dim_macro",
        "sub_TP", "sub_VP", "sub_SP",
        "valid_groups", "total_q2", "errors"
    ]

    colw = {h: len(h) for h in headers}
    for r in rows:
        colw["file"] = max(colw["file"], len(str(r.get("file", ""))))
        for h in headers[1:]:
            val = r.get(h)
            if isinstance(val, float) or val is None:
                s = fmt_rate(val, width=6)
            else:
                s = str(val)
            colw[h] = max(colw[h], len(s))

    print("\n" + "=" * 80)
    print("Batch Summary (Accuracy in %):")
    print("-" * 80)
    line = " | ".join(h.ljust(colw[h]) for h in headers)
    print(line)
    print("-" * len(line))

    for r in rows:
        print(
            f"{str(r.get('file','')).ljust(colw['file'])} | "
            f"{fmt_rate(r.get('overall')).rjust(colw['overall'])} | "
            f"{fmt_rate(r.get('main_macro')).rjust(colw['main_macro'])} | "
            f"{fmt_rate(r.get('dim_macro')).rjust(colw['dim_macro'])} | "
            f"{fmt_rate(r.get('sub_TP')).rjust(colw['sub_TP'])} | "
            f"{fmt_rate(r.get('sub_VP')).rjust(colw['sub_VP'])} | "
            f"{fmt_rate(r.get('sub_SP')).rjust(colw['sub_SP'])} | "
            f"{str(r.get('valid_groups',0)).rjust(colw['valid_groups'])} | "
            f"{str(r.get('total_q2',0)).rjust(colw['total_q2'])} | "
            f"{str(r.get('errors',0)).rjust(colw['errors'])}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-aware RTV-Bench evaluator (unchanged scoring) with acc-style reports + saving."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="JSON file path(s) or glob pattern(s), e.g. eval_results/qwen2.5-VL-*.json",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for reports (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Skip per-file stdout printing (still saves per-file reports).",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        print("No input files matched.")
        return

    outdir = ensure_dir(args.outdir)

    rows: List[Dict[str, Any]] = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            state = evaluate_one_file(p)

            report_text = build_report_text(state)
            report_path = os.path.join(outdir, f"{base}.txt")
            write_text(report_path, report_text)

            if not args.no_verbose:
                print(report_text, end="")

            rows.append(summarize_for_batch(state))

        except Exception as e:
            err_text = f"[ERROR] Failed on: {p}\nReason: {e}\n"
            write_text(os.path.join(outdir, f"{base}.error.txt"), err_text)
            print(err_text)

    write_batch_summaries(outdir, rows)
    print_batch_summary(rows)

    print("\n" + "=" * 80)
    print(f"Reports saved to: {outdir}")


if __name__ == "__main__":
    main()
