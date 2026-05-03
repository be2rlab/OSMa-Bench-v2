from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .diversity import farthest_first_selection, remove_near_duplicates
from .filters import analyze_prompt, normalize_text_for_dedup, validate_prompt
from .io_utils import save_json, save_prompts_csv
from .openai_client import call_json_response, embed_texts, make_openai_client
from .prompt_specs import build_generation_prompt, get_prompt_spec
from .schemas import PromptItem, PromptRunSummary


GENERATION_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_OUTPUT_TOKENS = 3000
NEAR_DUP_THRESHOLD = 0.90


def parse_prompt_batch_response(data: dict) -> list[str]:
    """Extract prompt strings from model JSON."""

    prompts: list[str] = []

    for item in data.get("prompts", []):
        if isinstance(item, dict) and "prompt" in item:
            prompt = str(item["prompt"]).strip()
        elif isinstance(item, str):
            prompt = item.strip()
        else:
            continue

        if prompt:
            prompts.append(prompt)

    return prompts


def generate_prompt_batch(
    client,
    scene_type: str,
    batch_size: int,
    room_hint: str,
    scenario_hint: str,
) -> list[str]:
    """Generate one batch of prompts for a selected scene type."""

    spec = get_prompt_spec(scene_type)

    data = call_json_response(
        client=client,
        model=GENERATION_MODEL,
        system_prompt=spec.system_prompt,
        user_prompt=build_generation_prompt(
            spec=spec,
            batch_size=batch_size,
            room_hint=room_hint,
            scenario_hint=scenario_hint,
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return parse_prompt_batch_response(data)


def generate_raw_prompt_pool(
    client,
    scene_type: str,
    raw_target: int,
    batch_size: int,
    max_attempts: int,
    seed: int,
) -> tuple[list[PromptItem], list]:
    """Generate and soft-filter a raw prompt pool."""

    spec = get_prompt_spec(scene_type)
    rng = random.Random(seed)

    collected: list[PromptItem] = []
    rejected_analysis = []
    seen_exact: set[str] = set()

    attempt = 0
    next_id = 1

    while len(collected) < raw_target and attempt < max_attempts:
        attempt += 1

        room_hint = rng.choice(spec.room_hints)
        scenario_hint = rng.choice(spec.scenario_hints)

        try:
            batch = generate_prompt_batch(
                client=client,
                scene_type=scene_type,
                batch_size=batch_size,
                room_hint=room_hint,
                scenario_hint=scenario_hint,
            )
        except Exception as exc:
            print(f"[GEN ERROR] attempt={attempt}: {type(exc).__name__}: {exc}")
            continue

        for prompt in batch:
            reasons = validate_prompt(prompt, spec)

            if reasons:
                rejected_analysis.append(
                    analyze_prompt(
                        prompt_id=f"rejected_{len(rejected_analysis) + 1:04d}",
                        prompt=prompt,
                        spec=spec,
                        rejected_reasons=reasons,
                    )
                )
                continue

            norm = normalize_text_for_dedup(prompt)
            if norm in seen_exact:
                continue

            seen_exact.add(norm)

            collected.append(
                PromptItem(
                    prompt_id=f"{scene_type}_{next_id:04d}",
                    prompt=prompt,
                    scene_type=scene_type,
                    room_hint=room_hint,
                    scenario_hint=scenario_hint,
                )
            )
            next_id += 1

            if len(collected) >= raw_target:
                break

        print(f"[POOL] scene_type={scene_type} attempt={attempt} collected={len(collected)}/{raw_target}")

    if not collected:
        raise RuntimeError("No valid prompts were collected.")

    return collected, rejected_analysis


def run_generation(
    scene_type: str,
    raw_target: int | None,
    final_target: int | None,
    batch_size: int | None,
    max_attempts: int,
    output_dir: Path | None,
    seed: int,
) -> PromptRunSummary:
    """Run the full prompt pipeline."""

    spec = get_prompt_spec(scene_type)
    client = make_openai_client()

    raw_target = raw_target or spec.default_raw_target
    final_target = final_target or spec.default_final_target
    batch_size = batch_size or spec.default_batch_size
    output_dir = output_dir or Path(spec.output_dir_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_items, rejected_analysis = generate_raw_prompt_pool(
        client=client,
        scene_type=scene_type,
        raw_target=raw_target,
        batch_size=batch_size,
        max_attempts=max_attempts,
        seed=seed,
    )

    save_json(output_dir / "raw_prompts.json", raw_items)
    save_json(output_dir / "rejected_prompts_analysis.json", rejected_analysis)

    raw_analysis = [
        analyze_prompt(item.prompt_id, item.prompt, spec)
        for item in raw_items
    ]
    save_json(output_dir / "raw_prompts_analysis.json", raw_analysis)

    raw_embeddings = embed_texts(
        client=client,
        texts=[item.prompt for item in raw_items],
        model=EMBEDDING_MODEL,
    )

    dedup_items, keep_indices = remove_near_duplicates(
        items=raw_items,
        embeddings=raw_embeddings,
        threshold=NEAR_DUP_THRESHOLD,
    )
    save_json(output_dir / "dedup_prompts.json", dedup_items)

    dedup_embeddings = raw_embeddings[keep_indices]

    final_items = farthest_first_selection(
        items=dedup_items,
        embeddings=dedup_embeddings,
        k=min(final_target, len(dedup_items)),
        seed_idx=0,
    )

    final_analysis = [
        analyze_prompt(item.prompt_id, item.prompt, spec)
        for item in final_items
    ]

    save_json(output_dir / "final_prompts.json", final_items)
    save_json(output_dir / "final_prompts_analysis.json", final_analysis)
    save_prompts_csv(output_dir / "final_prompts.csv", final_items)

    room_counts: dict[str, int] = {}
    for item in final_analysis:
        room_counts[item.room_guess] = room_counts.get(item.room_guess, 0) + 1

    summary = PromptRunSummary(
        scene_type=scene_type,
        raw_count=len(raw_items),
        dedup_count=len(dedup_items),
        final_count=len(final_items),
        rooms_in_final=room_counts,
        output_dir=str(output_dir.resolve()),
    )

    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SceneSmith prompts for OSMa-Bench++.")

    parser.add_argument(
        "--scene-type",
        required=True,
        choices=["furniture", "manipuland"],
        help="Prompt regime to use.",
    )
    parser.add_argument("--raw-target", type=int, default=None)
    parser.add_argument("--final-target", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    run_generation(
        scene_type=args.scene_type,
        raw_target=args.raw_target,
        final_target=args.final_target,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
