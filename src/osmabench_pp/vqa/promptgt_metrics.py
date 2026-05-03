#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STANDARD_CATEGORY_MAP = {
    "Measurement": "Measurements",
    "Object Relations - Spatial": "Relations",
    "Object Relations - Functional": "Relations",
}

PROMPTGT_CATEGORY_MAP = {
    "PromptGT-Measurement": "Measurements",
    "PromptGT-Relations": "Relations",
    "PromptGT-Existence": "Relations",
}

LABEL_MAP = {
    "evaluated_baseline": "baseline",
    "evaluated_camera": "camera",
    "evaluated_camera_lights": "camera",
    "evaluated_dynamic_lights": "dynamic_lights",
    "evaluated_no_lights": "no_lights",
    "baseline": "baseline",
    "camera": "camera",
    "camera_lights": "camera",
    "dynamic_lights": "dynamic_lights",
    "no_lights": "no_lights",
    "nominal": "no_lights",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_approach(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "bbq" in parts:
        return "BBQ"
    if "cg" in parts or "conceptgraphs" in parts:
        return "CG"

    text = str(path).lower()
    if "bbq" in text:
        return "BBQ"
    if "cg" in text or "conceptgraph" in text:
        return "CG"
    return "unknown"


def normalize_label(path: Path) -> str:
    for part in path.parts:
        part_l = part.lower()
        if part_l in LABEL_MAP:
            return LABEL_MAP[part_l]
    return "unknown"


def normalize_subset(path: Path) -> str:
    name = path.name.lower()
    text = str(path).lower()

    if "furniture_stage" in name or "furniture_stage" in text:
        return "furniture"
    if "manipuland_stage" in name or "manipuland_stage" in text:
        return "manipuland"
    if "furniture" in text:
        return "furniture"
    if "manipuland" in text:
        return "manipuland"
    return "unknown"


def is_yes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "1", "correct", "similar"}


def get_status(entry: Dict[str, Any]) -> str:
    for key in (
        "similar",
        "is_correct",
        "correct",
        "match",
        "prompt_gt_similar",
        "promptgt_similar",
        "prompt_gt_correct",
        "promptgt_correct",
    ):
        if key in entry:
            return "correct" if is_yes(entry[key]) else "wrong"
    return "unknown"


def summarize_entries(
    root: Path,
    source: str,
    category_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "unknown": 0,
        }
    )

    for path in sorted(root.rglob("*answered*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        approach = normalize_approach(path)
        label = normalize_label(path)
        subset = normalize_subset(path)

        for entry in data:
            if not isinstance(entry, dict):
                continue

            raw_category = str(entry.get("category", ""))
            if raw_category not in category_map:
                continue

            category = category_map[raw_category]
            key = (approach, subset, label, category)
            status = get_status(entry)

            grouped[key]["total"] += 1
            if status == "correct":
                grouped[key]["correct"] += 1
            elif status == "wrong":
                grouped[key]["wrong"] += 1
            else:
                grouped[key]["unknown"] += 1

    rows: List[Dict[str, Any]] = []
    for (approach, subset, label, category), stats in sorted(grouped.items()):
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total else None

        rows.append(
            {
                "approach": approach,
                "subset": subset,
                "label": label,
                "source": source,
                "category": category,
                "total": total,
                "correct": correct,
                "wrong": stats["wrong"],
                "unknown": stats["unknown"],
                "accuracy": accuracy,
            }
        )

    return rows


def build_final_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for row in rows:
        key = (row["approach"], row["subset"], row["label"])
        out = by_key.setdefault(
            key,
            {
                "approach": row["approach"],
                "subset": row["subset"],
                "label": row["label"],
                "Standard VQA | Relations": None,
                "PromptGT | Relations": None,
                "Standard VQA | Measurements": None,
                "PromptGT | Measurements": None,
            },
        )

        col = f'{row["source"]} | {row["category"]}'
        if col in out:
            out[col] = row["accuracy"]

    order = {
        "BBQ": 0,
        "CG": 1,
        "unknown": 9,
    }
    label_order = {
        "baseline": 0,
        "camera": 1,
        "dynamic_lights": 2,
        "no_lights": 3,
        "unknown": 9,
    }

    return sorted(
        by_key.values(),
        key=lambda r: (
            order.get(r["approach"], 9),
            r["subset"],
            label_order.get(r["label"], 9),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Standard VQA and PromptGT results into comparable summary tables."
    )
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--promptgt-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    standard_root = args.standard_root.resolve()
    promptgt_root = args.promptgt_root.resolve()
    out_dir = args.out_dir.resolve()

    if not standard_root.exists():
        raise FileNotFoundError(f"Missing standard VQA root: {standard_root}")
    if not promptgt_root.exists():
        raise FileNotFoundError(f"Missing PromptGT root: {promptgt_root}")

    standard_rows = summarize_entries(
        root=standard_root,
        source="Standard VQA",
        category_map=STANDARD_CATEGORY_MAP,
    )
    promptgt_rows = summarize_entries(
        root=promptgt_root,
        source="PromptGT",
        category_map=PROMPTGT_CATEGORY_MAP,
    )

    all_rows = standard_rows + promptgt_rows
    final_table = build_final_table(all_rows)

    write_csv(out_dir / "standard_vqa_summary.csv", standard_rows)
    write_csv(out_dir / "promptgt_summary.csv", promptgt_rows)
    write_csv(out_dir / "vqa_vs_promptgt_aggregated_long.csv", all_rows)
    write_csv(out_dir / "vqa_vs_promptgt_final_table.csv", final_table)

    print("[OK] wrote PromptGT/VQA summaries")
    print(f"standard rows : {len(standard_rows)}")
    print(f"promptgt rows : {len(promptgt_rows)}")
    print(f"final rows    : {len(final_table)}")
    print(f"out dir       : {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
