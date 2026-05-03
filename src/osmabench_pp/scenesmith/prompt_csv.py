from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenePrompt:
    """One prompt row used for SceneSmith generation."""

    scene_id: str
    prompt: str
    row_index: int


def _looks_like_header(row: list[str]) -> bool:
    """Detect whether the first CSV row is a header."""

    normalized = [cell.strip().lower() for cell in row]
    return "prompt" in normalized


def read_prompt_csv(csv_path: Path) -> list[ScenePrompt]:
    """Read SceneSmith prompts from a CSV file.

    Supported formats:
    - with header: scene_index,prompt
    - without header: scene_id,prompt
    """

    rows: list[ScenePrompt] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_rows = [row for row in reader if row]

    if not raw_rows:
        return rows

    has_header = _looks_like_header(raw_rows[0])

    if has_header:
        header = [cell.strip() for cell in raw_rows[0]]
        prompt_idx = header.index("prompt")

        if "scene_index" in header:
            scene_idx = header.index("scene_index")
        elif "scene_id" in header:
            scene_idx = header.index("scene_id")
        else:
            scene_idx = None

        data_rows = raw_rows[1:]

        for row_index, row in enumerate(data_rows, start=1):
            if len(row) <= prompt_idx:
                raise ValueError(f"CSV row {row_index} has no prompt column: {row}")

            scene_id = str(row[scene_idx]).strip() if scene_idx is not None and len(row) > scene_idx else str(row_index)
            prompt = str(row[prompt_idx]).strip()

            if scene_id and prompt:
                rows.append(ScenePrompt(scene_id=scene_id, prompt=prompt, row_index=row_index))

        return rows

    for row_index, row in enumerate(raw_rows, start=1):
        if len(row) < 2:
            raise ValueError(f"CSV row {row_index} has fewer than 2 columns: {row}")

        scene_id = str(row[0]).strip()
        prompt = str(row[1]).strip()

        if scene_id and prompt:
            rows.append(ScenePrompt(scene_id=scene_id, prompt=prompt, row_index=row_index))

    return rows


def select_prompt_range(
    prompts: list[ScenePrompt],
    start_index: int,
    end_index: int | None,
) -> list[ScenePrompt]:
    """Select prompts by 1-based row index."""

    if start_index < 1:
        raise ValueError("start_index must be >= 1.")

    if end_index is None:
        end_index = len(prompts)

    if end_index < start_index:
        raise ValueError(f"Invalid range: start={start_index}, end={end_index}")

    return [
        item for item in prompts
        if start_index <= item.row_index <= end_index
    ]
