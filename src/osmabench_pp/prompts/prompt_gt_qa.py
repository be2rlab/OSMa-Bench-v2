from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .io_utils import save_json
from .openai_client import call_json_response, make_openai_client


MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 2200
FRAME_NAME = "prompt_ground_truth"

ALLOWED_CATEGORIES = {
    "PromptGT-Measurement",
    "PromptGT-Relations",
    "PromptGT-Existence",
}


SYSTEM_PROMPT = """
You generate scene-level question-answer pairs strictly from a given scene prompt.

Rules:
- Use ONLY the information explicitly stated in the prompt.
- Do NOT infer hidden objects, missing objects, or geometry not explicitly described.
- Do NOT add questions about objects that are not explicitly mentioned.
- Do NOT create negative-existence questions.
- Do NOT create ambiguous questions.
- Prefer exact count questions and exact spatial relation questions.
- Return JSON only.

Question policy:
- Generate count questions for explicitly counted core objects.
- Generate relation questions for explicitly stated relations.
- Generate optional existence questions only for clearly stated objects or object groups.
- Avoid duplicate questions.
- Answers must be short.
- Count answers must be Arabic numerals.
- Boolean answers must be exactly "Yes" or "No".

Categories allowed:
- PromptGT-Measurement
- PromptGT-Relations
- PromptGT-Existence

Return format:
{
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "category": "PromptGT-Measurement"
    }
  ]
}
""".strip()


def build_user_prompt(scene_name: str, scene_prompt: str) -> str:
    """Build the user prompt used to generate PromptGT QA."""

    return f"""
Scene name:
{scene_name}

Scene prompt:
{scene_prompt}

Task:
Generate 8 to 14 scene-level QA pairs grounded strictly in the scene prompt.

Requirements:
- At least 3 PromptGT-Measurement questions.
- At least 3 PromptGT-Relations questions.
- 1 to 4 PromptGT-Existence questions.
- Use only explicit information from the prompt.
- Prefer questions about counts of core objects, exact stated spatial relations, and whether a stated object is present.
- For relation questions, ask only about relations explicitly described in the prompt.
- Do not ask about non-mentioned objects.
- Do not ask about unspecified viewpoints.
- Do not ask open-ended questions with multiple possible answers.
""".strip()


def choose_prompt_source(row: dict[str, str], prefer: str) -> str:
    """Choose the best prompt text from one metadata CSV row."""

    candidates = [prefer]

    for key in ("house_prompt", "room_prompts", "room_text_descriptions"):
        if key not in candidates:
            candidates.append(key)

    for key in candidates:
        value = row.get(key, "")
        if value and value.strip():
            return value.strip()

    subset = row.get("subset", "")
    scene_name = row.get("scene_name", "")
    raise ValueError(f"No usable prompt found for subset={subset}, scene_name={scene_name}")


def make_full_scene_name(subset: str, scene_name: str) -> str:
    """Build a stable scene name used in PromptGT output files."""

    subset = subset.strip()
    scene_name = scene_name.strip()

    if not subset:
        return scene_name

    return f"{subset}__{scene_name}"


def load_scenesmith_metadata_csv(csv_path: Path, prefer_prompt_field: str) -> list[dict[str, str]]:
    """Load SceneSmith prompt metadata CSV produced by scenesmith_metadata.py."""

    rows: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "subset",
            "scene_name",
            "house_prompt",
            "room_prompts",
            "room_text_descriptions",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            subset = (row.get("subset") or "").strip()
            scene_name = (row.get("scene_name") or "").strip()

            if not scene_name:
                continue

            prompt_text = choose_prompt_source(row, prefer=prefer_prompt_field)

            rows.append(
                {
                    "subset": subset,
                    "scene_name": scene_name,
                    "full_scene_name": make_full_scene_name(subset, scene_name),
                    "scene_path": (row.get("scene_path") or "").strip(),
                    "prompt_source": (row.get("prompt_source") or "").strip(),
                    "prompt_text": prompt_text,
                }
            )

    if not rows:
        raise RuntimeError(f"No valid rows loaded from CSV: {csv_path}")

    return rows


def normalize_answer(answer: str, category: str) -> str:
    """Normalize answers for OSMa-Bench-style matching."""

    answer = str(answer).strip()

    if category == "PromptGT-Measurement":
        if answer.isdigit():
            return answer

        digits = "".join(ch for ch in answer if ch.isdigit())
        if digits:
            return digits

    low = answer.lower()
    if low in {"yes", "true"}:
        return "Yes"
    if low in {"no", "false"}:
        return "No"

    return answer


def clean_qa_list(raw_qa: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Validate, normalize, and deduplicate generated QA items."""

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in raw_qa:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        category = str(item.get("category", "")).strip()

        if not question or not answer:
            continue

        if category not in ALLOWED_CATEGORIES:
            continue

        answer = normalize_answer(answer, category)

        key = (question.lower(), answer.lower(), category)
        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
            }
        )

    return cleaned


def to_osmabench_scene_json(scene_name: str, qa_items: list[dict[str, str]]) -> dict[str, Any]:
    """Convert QA items into the OSMa-Bench-compatible scene JSON structure."""

    return {
        "scene_name": scene_name,
        "parameters": [
            {
                "frame": FRAME_NAME,
                "qa": qa_items,
            }
        ],
    }


def generate_scene_questions(client, scene_name: str, scene_prompt: str) -> dict[str, Any]:
    """Generate PromptGT QA for one scene."""

    data = call_json_response(
        client=client,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(scene_name, scene_prompt),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    raw_qa = data.get("qa", [])
    if not isinstance(raw_qa, list):
        raw_qa = []

    qa_items = clean_qa_list(raw_qa)
    return to_osmabench_scene_json(scene_name, qa_items)


def generate_prompt_gt_dataset(
    csv_path: Path,
    output_dir: Path,
    prefer_prompt_field: str,
    limit: int | None,
    osma_vqa_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate PromptGT QA files for all scenes in a metadata CSV."""

    client = make_openai_client()

    rows = load_scenesmith_metadata_csv(
        csv_path=csv_path,
        prefer_prompt_field=prefer_prompt_field,
    )

    if limit is not None:
        rows = rows[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        scene_name = row["full_scene_name"]
        prompt_text = row["prompt_text"]

        scene_json = generate_scene_questions(
            client=client,
            scene_name=scene_name,
            scene_prompt=prompt_text,
        )

        save_json(output_dir / f"{scene_name}_questions.json", scene_json)

        if osma_vqa_root is not None:
            osma_questions_path = osma_vqa_root / scene_name / "vqa" / f"{scene_name}_questions.json"
            save_json(osma_questions_path, scene_json)

        merged.append(scene_json)

        qa_count = len(scene_json["parameters"][0]["qa"])
        print(f"[OK] {index}/{len(rows)} {scene_name}: {qa_count} QA")

    save_json(output_dir / "merged_prompt_gt_questions.json", merged)

    print()
    print(f"[DONE] saved {len(merged)} scene files to {output_dir}")

    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate prompt-grounded QA from SceneSmith prompt metadata."
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to SceneSmith prompt metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save generated PromptGT JSON files.",
    )
    parser.add_argument(
        "--prefer-prompt-field",
        default="house_prompt",
        choices=["house_prompt", "room_prompts", "room_text_descriptions"],
        help="CSV prompt field to prefer as GT source.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing on the first N scenes.",
    )
    parser.add_argument(
        "--osma-vqa-root",
        type=Path,
        default=None,
        help=(
            "Optional OSMa-Bench base_scenes_dir-style root. "
            "If set, each scene QA file is also written to "
            "<root>/<scene_name>/vqa/<scene_name>_questions.json."
        ),
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    generate_prompt_gt_dataset(
        csv_path=args.csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        prefer_prompt_field=args.prefer_prompt_field,
        limit=args.limit,
        osma_vqa_root=args.osma_vqa_root.expanduser().resolve() if args.osma_vqa_root else None,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
