from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUBSETS = ("furniture_stage", "manipuland_stage")


HOUSE_STATE_CANDIDATES = (
    "combined_house/house_state.json",
    "combined_house_after_ceiling/house_state.json",
    "combined_house_after_furniture/house_state.json",
    "combined_house_after_wall_objects/house_state.json",
)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON safely. Return None if the file cannot be parsed."""

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to parse {path}: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[WARN] JSON root is not an object: {path}")
        return None

    return data


def extract_prompt_fields(data: dict[str, Any]) -> dict[str, str]:
    """Extract prompt-like fields from SceneSmith house_state.json."""

    result = {
        "house_prompt": "",
        "room_prompts": "",
        "room_text_descriptions": "",
    }

    layout = data.get("layout") or {}
    rooms_top = data.get("rooms") or {}

    if not isinstance(layout, dict):
        layout = {}
    if not isinstance(rooms_top, dict):
        rooms_top = {}

    house_prompt = layout.get("house_prompt")
    if isinstance(house_prompt, str) and house_prompt.strip():
        result["house_prompt"] = house_prompt.strip()

    room_prompts: list[str] = []
    for room in layout.get("rooms", []) or []:
        if not isinstance(room, dict):
            continue

        room_id = str(room.get("id", "")).strip()
        prompt = room.get("prompt", "")

        if isinstance(prompt, str) and prompt.strip():
            if room_id:
                room_prompts.append(f"{room_id}: {prompt.strip()}")
            else:
                room_prompts.append(prompt.strip())

    result["room_prompts"] = " || ".join(room_prompts)

    room_descriptions: list[str] = []
    for room_id, room_data in rooms_top.items():
        if not isinstance(room_data, dict):
            continue

        text_description = room_data.get("text_description", "")
        if isinstance(text_description, str) and text_description.strip():
            room_descriptions.append(f"{room_id}: {text_description.strip()}")

    result["room_text_descriptions"] = " || ".join(room_descriptions)

    return result


def find_best_house_state(scene_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Find the first SceneSmith house_state.json that contains usable prompt fields."""

    for relative_path in HOUSE_STATE_CANDIDATES:
        candidate = scene_dir / relative_path

        if not candidate.exists():
            continue

        data = load_json(candidate)
        if data is None:
            continue

        extracted = extract_prompt_fields(data)
        if any(extracted.values()):
            return candidate, data

    return None, None


def collect_scene_rows(root: Path, subsets: tuple[str, ...]) -> list[dict[str, str]]:
    """Collect prompt metadata rows from SceneSmith generated scene folders."""

    rows: list[dict[str, str]] = []

    for subset_name in subsets:
        subset_dir = root / subset_name

        if not subset_dir.exists():
            print(f"[WARN] subset directory does not exist: {subset_dir}")
            continue

        for scene_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
            source_path, data = find_best_house_state(scene_dir)

            if data is not None:
                extracted = extract_prompt_fields(data)
                found = any(extracted.values())
            else:
                extracted = {
                    "house_prompt": "",
                    "room_prompts": "",
                    "room_text_descriptions": "",
                }
                found = False

            rows.append(
                {
                    "subset": subset_name,
                    "scene_name": scene_dir.name,
                    "scene_path": str(scene_dir.resolve()),
                    "prompt_source": str(source_path.resolve()) if source_path else "",
                    "house_prompt": extracted["house_prompt"],
                    "room_prompts": extracted["room_prompts"],
                    "room_text_descriptions": extracted["room_text_descriptions"],
                }
            )

            tag = "FOUND" if found else "MISS "
            print(f"[{tag}] {subset_name}/{scene_dir.name}")

    return rows


def save_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Save collected SceneSmith prompt metadata to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subset",
        "scene_name",
        "scene_path",
        "prompt_source",
        "house_prompt",
        "room_prompts",
        "room_text_descriptions",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_subsets(value: str) -> tuple[str, ...]:
    """Parse comma-separated subset names."""

    subsets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not subsets:
        raise argparse.ArgumentTypeError("At least one subset must be provided.")
    return subsets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect prompt metadata from generated SceneSmith scenes."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/Scenes_SceneSmith").expanduser(),
        help="Root directory containing SceneSmith subset folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <root>/scene_prompts.csv.",
    )
    parser.add_argument(
        "--subsets",
        type=parse_subsets,
        default=",".join(DEFAULT_SUBSETS),
        help="Comma-separated subset folders to scan.",
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    root = args.root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else root / "scene_prompts.csv"

    if not root.exists():
        raise FileNotFoundError(f"SceneSmith root does not exist: {root}")

    rows = collect_scene_rows(root=root, subsets=args.subsets)
    save_metadata_csv(output_csv, rows)

    print()
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
