from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SUBSETS = ("furniture_stage", "manipuland_stage")



COMBINED_HOUSE_CANDIDATES = (
    "combined_house",
    "combined_house_after_manipuland",
    "combined_house_after_manipulands",
    "combined_house_after_furniture",
    "combined_house_after_wall_objects",
    "combined_house_after_ceiling",
)


def find_combined_house_source(scene_root: Path) -> Path:
    """Find the best available SceneSmith combined_house directory.

    SceneSmith stage-limited runs may stop before creating combined_house/.
    For example, furniture-only runs usually create combined_house_after_furniture/.
    The original SceneSmith MuJoCo exporter expects scene_root/combined_house,
    so the adapter creates an alias when needed.
    """

    for name in COMBINED_HOUSE_CANDIDATES:
        candidate = scene_root / name
        if (candidate / "house_state.json").exists() or (candidate / "house.dmd.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "No combined house metadata found. Expected one of: "
        + ", ".join(str(scene_root / name) for name in COMBINED_HOUSE_CANDIDATES)
    )


def ensure_combined_house_alias(scene_root: Path, overwrite_alias: bool = False) -> Path:
    """Ensure scene_root/combined_house exists for the original SceneSmith exporter."""

    target = scene_root / "combined_house"
    source = find_combined_house_source(scene_root)

    if target.exists():
        return target

    if target.is_symlink():
        if overwrite_alias:
            target.unlink()
        else:
            return target

    try:
        target.symlink_to(source.name, target_is_directory=True)
        print(f"[ALIAS] {target} -> {source.name}")
    except OSError:
        import shutil

        shutil.copytree(source, target)
        print(f"[ALIAS COPY] {source} -> {target}")

    return target


@dataclass
class MuJoCoExportResult:
    scene_root: str
    output_dir: str
    success: bool
    reason: str


def discover_scene_roots(input_root: Path, subsets: tuple[str, ...]) -> list[Path]:
    """Find archived SceneSmith scenes under subset folders."""

    input_root = input_root.expanduser().resolve()
    scene_roots: list[Path] = []

    for subset in subsets:
        subset_dir = input_root / subset

        if not subset_dir.exists():
            print(f"[WARN] missing subset directory: {subset_dir}")
            continue

        for child in sorted(subset_dir.iterdir()):
            if child.is_dir():
                scene_roots.append(child)

    return scene_roots


def export_one_scene(
    scene_root: Path,
    exporter_script: Path,
    python_executable: str,
    overwrite: bool = False,
    write_usd: bool = True,
) -> MuJoCoExportResult:
    """Export one SceneSmith scene to MuJoCo format."""

    scene_root = scene_root.expanduser().resolve()
    exporter_script = exporter_script.expanduser().resolve()
    output_dir = scene_root / "mujoco"

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")

    if not exporter_script.exists():
        raise FileNotFoundError(f"Exporter script does not exist: {exporter_script}")

    ensure_combined_house_alias(scene_root=scene_root, overwrite_alias=overwrite)

    if output_dir.exists() and not overwrite:
        return MuJoCoExportResult(
            scene_root=str(scene_root),
            output_dir=str(output_dir),
            success=True,
            reason="already_exists",
        )

    cmd = [
        python_executable,
        str(exporter_script),
        str(scene_root),
        "-o",
        str(output_dir),
    ]

    if write_usd:
        cmd.append("--usd")

    print("=" * 100)
    print(f"[EXPORT MUJOCO] {scene_root}")
    print("Command:", " ".join(cmd))
    print("=" * 100)

    proc = subprocess.run(
        cmd,
        cwd=str(exporter_script.parent.parent),
        text=True,
    )

    if proc.returncode != 0:
        return MuJoCoExportResult(
            scene_root=str(scene_root),
            output_dir=str(output_dir),
            success=False,
            reason=f"returncode={proc.returncode}",
        )

    expected_scene_xml = output_dir / "scene.xml"
    expected_meshes_dir = output_dir / "meshes"

    if not expected_scene_xml.exists():
        return MuJoCoExportResult(
            scene_root=str(scene_root),
            output_dir=str(output_dir),
            success=False,
            reason="missing_scene_xml",
        )

    if not expected_meshes_dir.exists():
        return MuJoCoExportResult(
            scene_root=str(scene_root),
            output_dir=str(output_dir),
            success=False,
            reason="missing_meshes_dir",
        )

    return MuJoCoExportResult(
        scene_root=str(scene_root),
        output_dir=str(output_dir),
        success=True,
        reason="exported",
    )


def export_dataset(
    input_root: Path,
    exporter_script: Path,
    python_executable: str,
    subsets: tuple[str, ...],
    overwrite: bool,
    write_usd: bool,
    continue_on_error: bool,
) -> list[MuJoCoExportResult]:
    """Export all archived SceneSmith scenes under an input root."""

    scene_roots = discover_scene_roots(input_root=input_root, subsets=subsets)

    if not scene_roots:
        raise RuntimeError(f"No scene directories found under: {input_root}")

    results: list[MuJoCoExportResult] = []

    for index, scene_root in enumerate(scene_roots, start=1):
        print()
        print(f"[{index}/{len(scene_roots)}] {scene_root}")

        try:
            result = export_one_scene(
                scene_root=scene_root,
                exporter_script=exporter_script,
                python_executable=python_executable,
                overwrite=overwrite,
                write_usd=write_usd,
            )
        except Exception as exc:
            result = MuJoCoExportResult(
                scene_root=str(scene_root),
                output_dir=str(scene_root / "mujoco"),
                success=False,
                reason=f"{type(exc).__name__}: {exc}",
            )

            if not continue_on_error:
                raise

        results.append(result)
        status = "OK" if result.success else "FAIL"
        print(f"[{status}] {result.scene_root}: {result.reason}")

    return results


def parse_subsets(value: str) -> tuple[str, ...]:
    subsets = tuple(item.strip() for item in value.split(",") if item.strip())

    if not subsets:
        raise argparse.ArgumentTypeError("At least one subset must be provided.")

    return subsets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export archived SceneSmith scenes to MuJoCo format."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene-root", type=Path)
    mode.add_argument("--input-root", type=Path)

    parser.add_argument(
        "--exporter-script",
        required=True,
        type=Path,
        help="Path to SceneSmith scripts/export_scene_to_mujoco.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run SceneSmith exporter.",
    )
    parser.add_argument(
        "--subsets",
        type=parse_subsets,
        default=",".join(DEFAULT_SUBSETS),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    write_usd = not args.no_usd

    if args.scene_root is not None:
        result = export_one_scene(
            scene_root=args.scene_root,
            exporter_script=args.exporter_script,
            python_executable=args.python,
            overwrite=args.overwrite,
            write_usd=write_usd,
        )

        print(result)
        return 0 if result.success else 2

    results = export_dataset(
        input_root=args.input_root,
        exporter_script=args.exporter_script,
        python_executable=args.python,
        subsets=args.subsets,
        overwrite=args.overwrite,
        write_usd=write_usd,
        continue_on_error=args.continue_on_error,
    )

    failed = [result for result in results if not result.success]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total : {len(results)}")
    print(f"OK    : {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        for result in failed:
            print(f"- {result.scene_root}: {result.reason}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
