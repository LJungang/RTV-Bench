#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# Created by: Li-Jungang
# Email: ljungang.02@gmail.com
# AI Assistance: Yes (GPT-5.2)
# Is_Check: Yes
# Description: batch evaluator for RTV-Bench JSON results
# ================================================================
"""
Batch evaluator for RTV-Bench style JSON results.

Features:
- Evaluate one or multiple JSON files (supports glob patterns).
- Report overall accuracy (micro) and macro-average accuracies for:
  - Question types (qtype: 0/1/2)
  - Main categories (Object/Action/Event)
  - Dimension categories (Reasoning/Understanding/Perception)
- Print per-file detailed reports, write them to disk, and output a final batch summary table.

Outputs:
- Per-file report:  <outdir>/<basename>.txt
- Batch summary:    <outdir>/batch_summary.tsv
- Batch summary:    <outdir>/batch_summary.txt

Default outdir:
- ./RTV-Bench/eval_statics   (relative to current working directory)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VALID_QTYPES = ("0", "1", "2")
REQUIRED_FIELDS = ("questionID", "type")

MAIN_CATEGORIES = ("Object", "Action", "Event")
DIM_CATEGORIES = ("Reasoning", "Understanding", "Perception")

# Default output directory (requested): ./eval_statics
DEFAULT_OUTDIR = os.path.join(".", "eval_statics/acc")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def expand_inputs(inputs: Iterable[str]) -> List[str]:
    """Expand input paths and glob patterns into a sorted unique file list."""
    paths: List[str] = []
    for x in inputs:
        if any(ch in x for ch in ("*", "?", "[")):
            paths.extend(glob.glob(x))
        else:
            paths.append(x)
    return sorted(dict.fromkeys(paths))


def safe_rate(correct: int, total: int) -> Optional[float]:
    """Return accuracy in percent; None if total is zero."""
    return (correct / total * 100.0) if total else None


def macro_avg(rates: Iterable[Optional[float]]) -> Optional[float]:
    """Macro-average of accuracies (in percent), ignoring None."""
    vals = [r for r in rates if r is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt_stat_line(title: str, correct: int, total: int) -> str:
    """Format a single statistics line."""
    if total == 0:
        return f"{title.ljust(55)}: No valid data"
    rate = correct / total * 100
    return f"{title.ljust(55)}: {correct:>4}/{total:<4} | {rate:>6.2f}%"


def parse_qtype(question_id: Any) -> str:
    """Parse qtype (0/1/2) from questionID using RTV-Bench naming conventions."""
    if not isinstance(question_id, str):
        return "invalid_format"

    parts = question_id.split("-")
    if len(parts) >= 5 and parts[-2] in VALID_QTYPES:
        return parts[-2]
    if len(parts) >= 4 and parts[-1] in VALID_QTYPES:
        return parts[-1]
    return "unknown"


def parse_main_category(type_str: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse main category (Object/Action/Event) from the `type` string."""
    if not isinstance(type_str, str):
        return None, None
    for cat in MAIN_CATEGORIES:
        prefix = f"{cat}-"
        if type_str.startswith(prefix):
            return cat, type_str[len(prefix):]
    return None, None


def parse_dimension(type_str: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse dimension group (Reasoning/Understanding/Perception) and subtype from `type`.

    Keeps the original subtype extraction logic: subtype = type_str.split("-")[1].
    """
    if not isinstance(type_str, str):
        return None, None

    parts = type_str.split("-")
    sub = parts[1] if len(parts) > 1 else None

    if ("SR" in type_str) or ("FP" in type_str):
        return "Reasoning", sub
    if ("IA" in type_str) or ("PU" in type_str) or ("GU" in type_str):
        return "Understanding", sub
    if ("TP" in type_str) or ("VP" in type_str) or ("SP" in type_str):
        return "Perception", sub
    return None, None


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return normalized relative path."""
    os.makedirs(path, exist_ok=True)
    return os.path.normpath(path)



# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Counters:
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        self.correct += int(is_correct)


def make_counters() -> defaultdict:
    """key -> Counters"""
    return defaultdict(Counters)


def make_nested_counters() -> defaultdict:
    """group -> subgroup -> Counters"""
    return defaultdict(lambda: defaultdict(Counters))


def init_stats() -> Dict[str, Any]:
    """Initialize a stats container."""
    return {
        "overall": Counters(),
        "qtype": make_counters(),
        "main": make_counters(),
        "dim_group": make_counters(),
        "dim_detail": make_nested_counters(),
        "invalid": {
            "missing_fields": [],
            "invalid_qtype": [],
            "missing_correct": [],
        },
    }


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

def get_is_correct(item: Dict[str, Any]) -> Optional[bool]:
    """
    Determine correctness.
    - If `correct` exists, use it.
    - Else if pred == "ERROR", treat as incorrect.
    - Else return None (invalid for scoring).
    """
    if "correct" in item:
        return bool(item["correct"])
    if item.get("pred") == "ERROR":
        return False
    return None


def evaluate_file(path: str) -> Dict[str, Any]:
    """Evaluate a single JSON file and return stats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON must be a list, got {type(data)}: {path}")

    stats = init_stats()

    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            stats["invalid"]["missing_fields"].append(
                {"line": idx, "missing": list(REQUIRED_FIELDS), "id": "non-dict item"}
            )
            continue

        missing = [k for k in REQUIRED_FIELDS if k not in item]
        if missing:
            stats["invalid"]["missing_fields"].append(
                {"line": idx, "missing": missing, "id": item.get("questionID", "unknown")}
            )
            continue

        is_correct = get_is_correct(item)
        if is_correct is None:
            stats["invalid"]["missing_correct"].append(
                {"line": idx, "id": item.get("questionID", "unknown"), "pred": item.get("pred", "no_pred")}
            )
            continue

        qtype = parse_qtype(item["questionID"])
        if qtype not in VALID_QTYPES:
            stats["invalid"]["invalid_qtype"].append(
                {"line": idx, "id": item.get("questionID", "unknown"), "actual": qtype}
            )
            continue

        stats["overall"].add(is_correct)
        stats["qtype"][qtype].add(is_correct)

        main_cat, _ = parse_main_category(item["type"])
        if main_cat in MAIN_CATEGORIES:
            stats["main"][main_cat].add(is_correct)

        dim, sub = parse_dimension(item["type"])
        if dim in DIM_CATEGORIES and sub:
            stats["dim_group"][dim].add(is_correct)
            stats["dim_detail"][dim][sub].add(is_correct)

    return stats


# -----------------------------------------------------------------------------
# Reporting + saving
# -----------------------------------------------------------------------------

def compute_macro(counter_map: Dict[str, Counters], keys: Iterable[str]) -> Optional[float]:
    """Macro-average accuracy over a fixed key set (ignores missing/empty keys)."""
    return macro_avg(safe_rate(counter_map[k].correct, counter_map[k].total) for k in keys)


def build_report_text(path: str, stats: Dict[str, Any]) -> str:
    """Build a per-file report as a string."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"File: {path}")
    lines.append("-" * 80)

    lines.append("\nOverall:")
    lines.append(fmt_stat_line("Accuracy", stats["overall"].correct, stats["overall"].total))

    lines.append("\nMain categories:")
    for cat in MAIN_CATEGORIES:
        c = stats["main"][cat]
        lines.append(fmt_stat_line(cat, c.correct, c.total))
    main_macro = compute_macro(stats["main"], MAIN_CATEGORIES)
    if main_macro is not None:
        lines.append(f"{'Main Macro Avg'.ljust(55)}: {main_macro:>6.2f}%")

    lines.append("\nQuestion types:")
    for qt in VALID_QTYPES:
        c = stats["qtype"][qt]
        lines.append(fmt_stat_line(f"QType {qt}", c.correct, c.total))
    q_macro = compute_macro(stats["qtype"], VALID_QTYPES)
    if q_macro is not None:
        lines.append(f"{'QType Macro Avg'.ljust(55)}: {q_macro:>6.2f}%")

    lines.append("\nDimensions:")
    for dim in DIM_CATEGORIES:
        c = stats["dim_group"][dim]
        lines.append(fmt_stat_line(dim, c.correct, c.total))

        details = stats["dim_detail"].get(dim, {})
        for sub, cc in details.items():
            lines.append(fmt_stat_line(f"  - {dim}/{sub}", cc.correct, cc.total))

    dim_macro = compute_macro(stats["dim_group"], DIM_CATEGORIES)
    if dim_macro is not None:
        lines.append(f"{'Dimension Macro Avg'.ljust(55)}: {dim_macro:>6.2f}%")

    inv = stats["invalid"]
    lines.append("\n" + "-" * 80)
    lines.append("Data quality:")
    if inv["missing_fields"]:
        lines.append(f"Missing required fields: {len(inv['missing_fields'])}")
        for e in inv["missing_fields"][:3]:
            lines.append(f"  line {e['line']} | id={e['id']} | missing={','.join(e['missing'])}")
    if inv["missing_correct"]:
        lines.append(f"Missing `correct` (and pred != ERROR): {len(inv['missing_correct'])}")
        for e in inv["missing_correct"][:3]:
            lines.append(f"  line {e['line']} | id={e['id']} | pred={e['pred']}")
    if inv["invalid_qtype"]:
        lines.append(f"Invalid qtype: {len(inv['invalid_qtype'])}")
        for e in inv["invalid_qtype"][:3]:
            lines.append(f"  line {e['line']} | id={e['id']} | parsed={e['actual']}")

    return "\n".join(lines) + "\n"


def write_text(path: str, text: str) -> None:
    """Write text to file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def summarize_for_batch(path: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary row for batch reporting."""
    inv = stats["invalid"]
    return {
        "file": os.path.basename(path),
        "overall": safe_rate(stats["overall"].correct, stats["overall"].total),
        "qtype_macro": compute_macro(stats["qtype"], VALID_QTYPES),
        "main_macro": compute_macro(stats["main"], MAIN_CATEGORIES),
        "dim_macro": compute_macro(stats["dim_group"], DIM_CATEGORIES),
        "missing_fields": len(inv["missing_fields"]),
        "missing_correct": len(inv["missing_correct"]),
        "invalid_qtype": len(inv["invalid_qtype"]),
        "path": path,
    }


def write_batch_summaries(outdir: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Write batch summary in both TSV and human-readable TXT formats."""
    out_tsv = os.path.join(outdir, "batch_summary.tsv")
    out_txt = os.path.join(outdir, "batch_summary.txt")

    headers = [
        "file", "overall", "qtype_macro", "main_macro", "dim_macro",
        "missing_fields", "missing_correct", "invalid_qtype", "path",
    ]

    def rfmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    # TSV (machine-friendly)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join([
                r["file"],
                rfmt(r["overall"]),
                rfmt(r["qtype_macro"]),
                rfmt(r["main_macro"]),
                rfmt(r["dim_macro"]),
                str(r["missing_fields"]),
                str(r["missing_correct"]),
                str(r["invalid_qtype"]),
                r["path"],
            ]) + "\n")

    # TXT (human-friendly)
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
            rfmt(r["overall"]),
            rfmt(r["qtype_macro"]),
            rfmt(r["main_macro"]),
            rfmt(r["dim_macro"]),
            str(r["missing_fields"]),
            str(r["missing_correct"]),
            str(r["invalid_qtype"]),
            r["path"],
        ]))
    write_text(out_txt, "\n".join(lines) + "\n")

    return out_tsv, out_txt


def print_batch_summary(rows: List[Dict[str, Any]]) -> None:
    """Print a batch summary table to stdout."""
    if not rows:
        return

    headers = ["file", "overall", "qtype_macro", "main_macro", "dim_macro",
               "missing_fields", "missing_correct", "invalid_qtype"]

    def fmt_rate(x: Optional[float]) -> str:
        return "  N/A " if x is None else f"{x:6.2f}"

    colw = {h: len(h) for h in headers}
    for r in rows:
        colw["file"] = max(colw["file"], len(str(r["file"])))
        for h in headers[1:]:
            colw[h] = max(colw[h], len(str(r.get(h, ""))))

    print("\n" + "=" * 80)
    print("Batch Summary (Accuracy in %):")
    print("-" * 80)
    line = " | ".join(h.ljust(colw[h]) for h in headers)
    print(line)
    print("-" * len(line))

    for r in rows:
        print(
            f"{str(r['file']).ljust(colw['file'])} | "
            f"{fmt_rate(r['overall']).rjust(colw['overall'])} | "
            f"{fmt_rate(r['qtype_macro']).rjust(colw['qtype_macro'])} | "
            f"{fmt_rate(r['main_macro']).rjust(colw['main_macro'])} | "
            f"{fmt_rate(r['dim_macro']).rjust(colw['dim_macro'])} | "
            f"{str(r['missing_fields']).rjust(colw['missing_fields'])} | "
            f"{str(r['missing_correct']).rjust(colw['missing_correct'])} | "
            f"{str(r['invalid_qtype']).rjust(colw['invalid_qtype'])}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one or multiple RTV-Bench eval_results JSON files (supports glob)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="JSON file path(s) or glob pattern(s), e.g. eval_results/qwen2.5-VL-*.json",
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
        try:
            stats = evaluate_file(p)

            # Save per-file report
            base = os.path.splitext(os.path.basename(p))[0]
            report_path = os.path.join(outdir, f"{base}.txt")
            report_text = build_report_text(p, stats)
            write_text(report_path, report_text)

            if not args.no_verbose:
                print(report_text, end="")

            rows.append(summarize_for_batch(p, stats))

        except Exception as e:
            err_text = f"[ERROR] Failed on: {p}\nReason: {e}\n"
            base = os.path.splitext(os.path.basename(p))[0]
            write_text(os.path.join(outdir, f"{base}.error.txt"), err_text)
            print(err_text)

    write_batch_summaries(outdir, rows)
    print_batch_summary(rows)

    print("\n" + "=" * 80)
    print(f"Reports saved to: {outdir}")


if __name__ == "__main__":
    main()
