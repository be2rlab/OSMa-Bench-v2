from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .schemas import PromptItem


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses recursively into JSON-serializable objects."""

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def save_json(path: Path, obj: Any) -> None:
    """Save JSON with UTF-8 and stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def save_prompts_csv(path: Path, items: list[PromptItem]) -> None:
    """Save prompts in SceneSmith-compatible CSV format."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_index", "prompt"])

        for index, item in enumerate(items, start=1):
            writer.writerow([index, item.prompt])
