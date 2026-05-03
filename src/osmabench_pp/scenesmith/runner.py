from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .process_control import process_alive, terminate_process_group
from .prompt_csv import ScenePrompt


SUCCESS_PATTERNS = [
    re.compile(r"Scene generation completed successfully"),
    re.compile(r"ALL SCENES COMPLETED!"),
]

FAIL_PATTERNS = [
    re.compile(r"Scene generation failed"),
    re.compile(r"Loaded 0 prompts from CSV"),
    re.compile(r"Processing 0 scenes"),
    re.compile(r"RuntimeError:"),
    re.compile(r"Fatal Python error:"),
    re.compile(r"Process crashed"),
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"Error executing job with overrides:"),
]

RESOLVED_CONFIG_RE = re.compile(r"Saved resolved config to:\s*(.+?/resolved_config\.yaml)")


@dataclass
class SceneRunResult:
    """Result of running one prompt through SceneSmith."""

    scene_id: str
    row_index: int
    success: bool
    reason: str
    returncode: int | None
    log_path: str | None
    source_scene_path: str | None = None
    archived_scene_path: str | None = None


def scene_dir_name(scene_id: str) -> str:
    """Return stable SceneSmith-style scene directory name."""

    scene_id = str(scene_id).strip()

    if scene_id.isdigit():
        return f"scene_{int(scene_id):03d}"

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id)
    if safe.startswith("scene_"):
        return safe
    return f"scene_{safe}"


def write_single_prompt_csv(scene_id: str, prompt: str, out_path: Path) -> None:
    """Write a temporary one-row CSV consumed by SceneSmith.

    SceneSmith expects a header and two columns:
    scene_index,prompt
    """

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_index", "prompt"])
        writer.writerow([scene_id, prompt])


def build_scenesmith_command(
    python_executable: str,
    main_py: Path,
    run_name: str,
    prompt_csv: Path,
    hydra_overrides: list[str],
) -> list[str]:
    """Build the SceneSmith command line."""

    return [
        python_executable,
        str(main_py),
        f"+name={run_name}",
        f"experiment.csv_path={prompt_csv}",
        *hydra_overrides,
    ]


def find_generated_scene_dir(run_output_dir: Path, scene_id: str) -> Path | None:
    """Find generated scene directory inside one SceneSmith Hydra output directory."""

    expected = run_output_dir / scene_dir_name(scene_id)
    if expected.exists():
        return expected

    candidates = sorted(path for path in run_output_dir.glob("scene_*") if path.is_dir())

    if len(candidates) == 1:
        return candidates[0]

    numeric_candidates = [
        path for path in candidates
        if path.name.endswith(str(scene_id).zfill(3)) or path.name.endswith(str(scene_id))
    ]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    return None


def archive_generated_scene(
    source_scene_dir: Path,
    archive_root: Path,
    archive_subset: str,
    scene_id: str,
    overwrite: bool,
) -> Path:
    """Copy generated SceneSmith scene to a stable archive root."""

    destination = archive_root.expanduser().resolve() / archive_subset / scene_dir_name(scene_id)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Archive destination already exists: {destination}. "
                f"Use --overwrite-archive to replace it."
            )
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_scene_dir, destination)

    return destination


def run_one_prompt(
    item: ScenePrompt,
    main_py: Path,
    workdir: Path,
    logs_dir: Path,
    hydra_overrides: list[str],
    python_executable: str = sys.executable,
    success_grace_seconds: float = 3.0,
    archive_root: Path | None = None,
    archive_subset: str | None = None,
    overwrite_archive: bool = False,
) -> SceneRunResult:
    """Run one SceneSmith prompt, stream logs, and optionally archive the generated scene."""

    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"scene_{item.scene_id}"
    log_path = logs_dir / f"{run_name}.log"

    with tempfile.TemporaryDirectory(prefix=f"scenesmith_{run_name}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        one_prompt_csv = tmpdir_path / "single_prompt.csv"
        write_single_prompt_csv(item.scene_id, item.prompt, one_prompt_csv)

        cmd = build_scenesmith_command(
            python_executable=python_executable,
            main_py=main_py,
            run_name=run_name,
            prompt_csv=one_prompt_csv,
            hydra_overrides=hydra_overrides,
        )

        print("=" * 100)
        print(f"[START] row={item.row_index} scene_id={item.scene_id}")
        print(f"Prompt : {item.prompt}")
        print(f"Command: {' '.join(map(str, cmd))}")
        print(f"Log    : {log_path}")
        print("=" * 100)
        sys.stdout.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )

        saw_success = False
        saw_failure = False
        failure_reason = ""
        success_time: float | None = None
        returncode: int | None = None
        run_output_dir: Path | None = None

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                assert proc.stdout is not None

                for line in proc.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()

                    config_match = RESOLVED_CONFIG_RE.search(line)
                    if config_match:
                        run_output_dir = Path(config_match.group(1)).expanduser().resolve().parent

                    if any(pattern.search(line) for pattern in FAIL_PATTERNS):
                        saw_failure = True
                        failure_reason = line.strip()

                    if any(pattern.search(line) for pattern in SUCCESS_PATTERNS):
                        saw_success = True
                        success_time = time.time()

                    if saw_success and success_time is not None:
                        if time.time() - success_time >= success_grace_seconds:
                            print()
                            print(f"[SUCCESS MARKER] scene_id={item.scene_id}; stopping process group")
                            terminate_process_group(proc)
                            break

            if process_alive(proc) and saw_success:
                terminate_process_group(proc)

            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_process_group(proc)
                returncode = proc.wait(timeout=10)

        except KeyboardInterrupt:
            print()
            print(f"[INTERRUPTED] stopping SceneSmith process group for scene_id={item.scene_id}")
            terminate_process_group(proc)
            raise

    source_scene_dir: Path | None = None
    archived_scene_dir: Path | None = None

    if run_output_dir is not None:
        source_scene_dir = find_generated_scene_dir(run_output_dir, item.scene_id)

    if source_scene_dir is not None and archive_root is not None and archive_subset is not None:
        archived_scene_dir = archive_generated_scene(
            source_scene_dir=source_scene_dir,
            archive_root=archive_root,
            archive_subset=archive_subset,
            scene_id=item.scene_id,
            overwrite=overwrite_archive,
        )
        print(f"[ARCHIVED] {source_scene_dir} -> {archived_scene_dir}")

    if saw_failure:
        print(f"[DONE] scene_id={item.scene_id}: failure")
        return SceneRunResult(
            scene_id=item.scene_id,
            row_index=item.row_index,
            success=False,
            reason=failure_reason or f"returncode={returncode}",
            returncode=returncode,
            log_path=str(log_path),
            source_scene_path=str(source_scene_dir) if source_scene_dir else None,
            archived_scene_path=str(archived_scene_dir) if archived_scene_dir else None,
        )

    if saw_success:
        print(f"[DONE] scene_id={item.scene_id}: success")
        return SceneRunResult(
            scene_id=item.scene_id,
            row_index=item.row_index,
            success=True,
            reason="success marker detected",
            returncode=returncode,
            log_path=str(log_path),
            source_scene_path=str(source_scene_dir) if source_scene_dir else None,
            archived_scene_path=str(archived_scene_dir) if archived_scene_dir else None,
        )

    success = returncode == 0
    reason = f"returncode={returncode}"

    print(f"[DONE] scene_id={item.scene_id}: {reason}")

    return SceneRunResult(
        scene_id=item.scene_id,
        row_index=item.row_index,
        success=success,
        reason=reason,
        returncode=returncode,
        log_path=str(log_path),
        source_scene_path=str(source_scene_dir) if source_scene_dir else None,
        archived_scene_path=str(archived_scene_dir) if archived_scene_dir else None,
    )
