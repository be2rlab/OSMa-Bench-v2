from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .prompt_csv import read_prompt_csv, select_prompt_range
from .runner import SceneRunResult, run_one_prompt


def save_summary(path: Path, results: list[SceneRunResult]) -> None:
    """Save batch run summary to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "total": len(results),
        "success": sum(1 for result in results if result.success),
        "failed": sum(1 for result in results if not result.success),
        "results": [asdict(result) for result in results],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_hydra_overrides(raw: list[str]) -> list[str]:
    """Remove optional '--' separator before Hydra overrides."""

    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected CSV prompts through SceneSmith one by one."
    )

    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--scenesmith-root", required=True, type=Path)
    parser.add_argument("--main", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=Path("runner_logs"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Optional stable root where generated scenes are copied, e.g. ~/Scenes_SceneSmith.",
    )
    parser.add_argument(
        "--archive-subset",
        type=str,
        default=None,
        help="Subset folder inside archive root, e.g. furniture_stage or manipuland_stage.",
    )
    parser.add_argument(
        "--overwrite-archive",
        action="store_true",
        help="Overwrite archived scene directory if it already exists.",
    )

    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides passed to SceneSmith after optional '--'.",
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    csv_path = args.csv.expanduser().resolve()
    scenesmith_root = args.scenesmith_root.expanduser().resolve()
    main_py = args.main.expanduser().resolve() if args.main else scenesmith_root / "main.py"
    logs_dir = args.logs_dir.expanduser().resolve()
    summary_json = args.summary_json.expanduser().resolve() if args.summary_json else logs_dir / "summary.json"
    hydra_overrides = split_hydra_overrides(args.overrides)

    archive_root = args.archive_root.expanduser().resolve() if args.archive_root else None
    archive_subset = args.archive_subset

    if archive_root is not None and not archive_subset:
        raise ValueError("--archive-subset is required when --archive-root is used.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt CSV does not exist: {csv_path}")
    if not scenesmith_root.exists():
        raise FileNotFoundError(f"SceneSmith root does not exist: {scenesmith_root}")
    if not main_py.exists():
        raise FileNotFoundError(f"SceneSmith main.py does not exist: {main_py}")

    all_prompts = read_prompt_csv(csv_path)
    selected_prompts = select_prompt_range(
        prompts=all_prompts,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    print(f"Loaded prompts : {len(all_prompts)}")
    print(f"Selected range : {args.start_index}..{args.end_index or len(all_prompts)}")
    print(f"Selected count : {len(selected_prompts)}")
    print(f"SceneSmith root: {scenesmith_root}")
    print(f"SceneSmith main: {main_py}")
    print(f"Logs dir       : {logs_dir}")
    print(f"Summary JSON   : {summary_json}")
    print(f"Archive root   : {archive_root}")
    print(f"Archive subset : {archive_subset}")
    print(f"Overrides      : {hydra_overrides}")
    print()

    results: list[SceneRunResult] = []

    for item in selected_prompts:
        result = run_one_prompt(
            item=item,
            main_py=main_py,
            workdir=scenesmith_root,
            logs_dir=logs_dir,
            hydra_overrides=hydra_overrides,
            python_executable=args.python,
            archive_root=archive_root,
            archive_subset=archive_subset,
            overwrite_archive=args.overwrite_archive,
        )

        results.append(result)
        save_summary(summary_json, results)

        if not result.success and not args.continue_on_error:
            print()
            print("[STOP] failure detected; use --continue-on-error to keep running")
            break

    failed = [result for result in results if not result.success]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total completed: {len(results)}")
    print(f"Success        : {len(results) - len(failed)}")
    print(f"Failed         : {len(failed)}")
    print(f"Summary JSON   : {summary_json}")

    if failed:
        print()
        print("Failures:")
        for result in failed:
            print(f"- row={result.row_index} scene_id={result.scene_id}: {result.reason}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
