from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextureFixResult:
    """Result of processing one OBJ file."""

    obj_path: str
    status: str
    texture: str | None = None


@dataclass
class SceneTextureFixSummary:
    """Summary of texture fixing for one SceneSmith scene."""

    scene_root: str
    fixed: int
    skipped: int
    no_texture: int
    no_uv: int
    total_obj: int


class TextureResolver:
    """Resolve SceneSmith OBJ textures and remember texture choice per object family.

    SceneSmith may export multipart objects where several OBJ files share one
    texture. The family cache prevents assigning different fallback textures to
    different parts of the same generated object.
    """

    def __init__(self) -> None:
        self.family_texture_cache: dict[str, str] = {}

    @staticmethod
    def should_skip_obj(obj_path: Path) -> bool:
        """Skip collision-only or helper OBJ files."""

        name = obj_path.stem.lower()

        skip_tokens = [
            "_collision_",
            "_vhacd_",
            "convex_piece",
        ]

        if any(token in name for token in skip_tokens):
            return True

        if name == "material":
            return True

        return False

    @staticmethod
    def extract_first_usemtl(obj_text: str) -> str:
        """Read the first usemtl statement from an OBJ file."""

        match = re.search(r"^usemtl\s+(.+?)\s*$", obj_text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()

        return "default_mat"

    @staticmethod
    def replace_or_insert_mtllib(obj_text: str, mtl_filename: str) -> str:
        """Ensure OBJ references the generated MTL file."""

        if re.search(r"^mtllib\s+.+$", obj_text, flags=re.MULTILINE):
            return re.sub(
                r"^mtllib\s+.+$",
                f"mtllib {mtl_filename}",
                obj_text,
                flags=re.MULTILINE,
                count=1,
            )

        return f"mtllib {mtl_filename}\n{obj_text}"

    @staticmethod
    def has_uvs(obj_text: str) -> bool:
        """Check whether OBJ has UV coordinates."""

        return re.search(r"^vt\s+", obj_text, flags=re.MULTILINE) is not None

    @staticmethod
    def normalize_material_candidates(material_name: str) -> list[str]:
        """Generate possible texture filename stems from a material name."""

        material = material_name.strip()
        candidates = [material]

        if material.lower().startswith("mat_"):
            suffix = material[4:]
            candidates.append(f"material_{suffix}")
            candidates.append(f"Material_{suffix}")

        if material.lower().startswith("material_"):
            suffix = material.split("_", 1)[1]
            candidates.append(f"material_{suffix}")
            candidates.append(f"Material_{suffix}")
            candidates.append(f"mat_{suffix}")

        candidates.append(material.lower())
        candidates.append(material.upper())

        seen: set[str] = set()
        unique: list[str] = []

        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        return unique

    @staticmethod
    def get_family_key(obj_path: Path) -> str:
        """Return shared object-family key for multipart SceneSmith assets."""

        stem = obj_path.stem

        if "_meshes_" in stem:
            return stem.split("_meshes_", 1)[0]

        return stem

    @staticmethod
    def pick_best_material_texture(mesh_dir: Path) -> Path | None:
        """Pick the largest Material*.png fallback texture."""

        hits = sorted(list(mesh_dir.glob("Material*.png")) + list(mesh_dir.glob("material*.png")))

        if not hits:
            return None

        hits = sorted(hits, key=lambda path: path.stat().st_size, reverse=True)

        for path in hits:
            if path.stat().st_size > 1024:
                return path

        return hits[0]

    @staticmethod
    def family_texture_candidates(mesh_dir: Path, family_key: str) -> list[Path]:
        """Find textures that look related to a multipart object family."""

        patterns = [
            f"{family_key}*texture*.png",
            f"{family_key}*.png",
        ]

        hits: list[Path] = []

        for pattern in patterns:
            hits.extend(sorted(mesh_dir.glob(pattern)))

        seen: set[str] = set()
        unique: list[Path] = []

        for path in hits:
            if path.name not in seen:
                seen.add(path.name)
                unique.append(path)

        return unique

    def guess_texture_path(self, obj_path: Path, material_name: str) -> Path | None:
        """Find the most likely texture image for one OBJ."""

        mesh_dir = obj_path.parent
        family_key = self.get_family_key(obj_path)

        cached = self.family_texture_cache.get(family_key)
        if cached is not None:
            cached_path = mesh_dir / cached
            if cached_path.exists():
                return cached_path

        direct_candidates = [
            mesh_dir / f"{obj_path.stem}_texture.png",
            mesh_dir / f"{obj_path.stem}.png",
        ]

        for candidate in direct_candidates:
            if candidate.exists():
                self.family_texture_cache[family_key] = candidate.name
                return candidate

        for base in self.normalize_material_candidates(material_name):
            patterns = [
                f"{base}.png",
                f"{base}.jpg",
                f"{base}.jpeg",
                f"{base}.*.png",
                f"{base}.*.jpg",
                f"{base}.*.jpeg",
            ]

            for pattern in patterns:
                hits = sorted(mesh_dir.glob(pattern))
                if hits:
                    self.family_texture_cache[family_key] = hits[0].name
                    return hits[0]

        family_hits = self.family_texture_candidates(mesh_dir, family_key)
        if family_hits:
            family_hits = sorted(family_hits, key=lambda path: path.stat().st_size, reverse=True)
            self.family_texture_cache[family_key] = family_hits[0].name
            return family_hits[0]

        if "_meshes_" in obj_path.stem:
            best = self.pick_best_material_texture(mesh_dir)
            if best is not None:
                self.family_texture_cache[family_key] = best.name
                return best

        generic_hits = sorted(mesh_dir.glob("*.png"))
        generic_hits = [path for path in generic_hits if path.stat().st_size > 1024]

        if len(generic_hits) == 1:
            self.family_texture_cache[family_key] = generic_hits[0].name
            return generic_hits[0]

        return None


def build_mtl_text(material_name: str, texture_name: str) -> str:
    """Create simple MTL text referencing the selected texture."""

    return (
        f"newmtl {material_name}\n"
        f"Ka 1.000 1.000 1.000\n"
        f"Kd 1.000 1.000 1.000\n"
        f"Ks 0.000 0.000 0.000\n"
        f"Ns 10.000\n"
        f"d 1.0\n"
        f"Tr 0.0\n"
        f"illum 2\n"
        f"map_Ka {texture_name}\n"
        f"map_Kd {texture_name}\n"
    )


def fix_single_obj(
    obj_path: Path,
    resolver: TextureResolver,
    dry_run: bool = False,
) -> TextureFixResult:
    """Fix texture reference for one OBJ file."""

    if resolver.should_skip_obj(obj_path):
        return TextureFixResult(obj_path=str(obj_path), status="skip")

    obj_text = obj_path.read_text(encoding="utf-8", errors="ignore")
    material_name = resolver.extract_first_usemtl(obj_text)

    if not resolver.has_uvs(obj_text):
        return TextureFixResult(obj_path=str(obj_path), status="no_uv")

    texture_path = resolver.guess_texture_path(obj_path, material_name)

    if texture_path is None:
        return TextureFixResult(obj_path=str(obj_path), status="no_texture")

    mtl_filename = f"{obj_path.stem}.mtl"
    mtl_path = obj_path.with_name(mtl_filename)

    mtl_text = build_mtl_text(
        material_name=material_name,
        texture_name=texture_path.name,
    )
    new_obj_text = resolver.replace_or_insert_mtllib(
        obj_text=obj_text,
        mtl_filename=mtl_filename,
    )

    if not dry_run:
        mtl_path.write_text(mtl_text, encoding="utf-8")
        obj_path.write_text(new_obj_text, encoding="utf-8")

    return TextureFixResult(
        obj_path=str(obj_path),
        status="fixed",
        texture=texture_path.name,
    )


def fix_scene_textures(scene_root: Path, dry_run: bool = False) -> SceneTextureFixSummary:
    """Fix OBJ/MTL texture references for one SceneSmith scene."""

    scene_root = scene_root.expanduser().resolve()
    meshes_dir = scene_root / "mujoco" / "meshes"

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")

    if not meshes_dir.exists():
        raise FileNotFoundError(f"Meshes directory does not exist: {meshes_dir}")

    resolver = TextureResolver()
    obj_files = sorted(meshes_dir.glob("*.obj"))

    fixed = 0
    skipped = 0
    no_texture = 0
    no_uv = 0

    for obj_path in obj_files:
        result = fix_single_obj(
            obj_path=obj_path,
            resolver=resolver,
            dry_run=dry_run,
        )

        if result.status == "fixed":
            fixed += 1
            print(f"[FIXED] {obj_path.name} -> {result.texture}")
        elif result.status == "skip":
            skipped += 1
        elif result.status == "no_texture":
            no_texture += 1
            print(f"[NO_TEXTURE] {obj_path.name}")
        elif result.status == "no_uv":
            no_uv += 1
            print(f"[NO_UV] {obj_path.name}")

    summary = SceneTextureFixSummary(
        scene_root=str(scene_root),
        fixed=fixed,
        skipped=skipped,
        no_texture=no_texture,
        no_uv=no_uv,
        total_obj=len(obj_files),
    )

    print()
    print("Done.")
    print(f"scene_root : {summary.scene_root}")
    print(f"fixed      : {summary.fixed}")
    print(f"skipped    : {summary.skipped}")
    print(f"no_texture : {summary.no_texture}")
    print(f"no_uv      : {summary.no_uv}")
    print(f"total_obj  : {summary.total_obj}")

    return summary


def discover_scene_roots(input_root: Path, subsets: tuple[str, ...]) -> list[Path]:
    """Discover SceneSmith scene directories under selected subsets."""

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


def fix_dataset_textures(
    input_root: Path,
    subsets: tuple[str, ...],
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> list[SceneTextureFixSummary]:
    """Fix textures for all discovered SceneSmith scenes."""

    scene_roots = discover_scene_roots(
        input_root=input_root,
        subsets=subsets,
    )

    if not scene_roots:
        raise RuntimeError(f"No scene directories found under: {input_root}")

    summaries: list[SceneTextureFixSummary] = []

    for index, scene_root in enumerate(scene_roots, start=1):
        print("=" * 100)
        print(f"[{index}/{len(scene_roots)}] FIX TEXTURES: {scene_root}")
        print("=" * 100)

        try:
            summary = fix_scene_textures(
                scene_root=scene_root,
                dry_run=dry_run,
            )
        except Exception as exc:
            print(f"[ERROR] {scene_root}: {type(exc).__name__}: {exc}")
            if not continue_on_error:
                raise
            continue

        summaries.append(summary)

    return summaries


def parse_subsets(value: str) -> tuple[str, ...]:
    """Parse comma-separated subset names."""

    subsets = tuple(item.strip() for item in value.split(",") if item.strip())

    if not subsets:
        raise argparse.ArgumentTypeError("At least one subset must be provided.")

    return subsets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fix OBJ/MTL texture links for SceneSmith scenes."
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--scene-root",
        type=Path,
        help="Single SceneSmith scene root, e.g. ~/Scenes_SceneSmith/furniture_stage/scene_001.",
    )
    mode.add_argument(
        "--input-root",
        type=Path,
        help="Root containing SceneSmith subset folders.",
    )

    parser.add_argument(
        "--subsets",
        type=parse_subsets,
        default=("furniture_stage", "manipuland_stage"),
        help="Comma-separated subsets to scan when --input-root is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing OBJ/MTL files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later scenes if one scene fails.",
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.scene_root is not None:
        fix_scene_textures(
            scene_root=args.scene_root,
            dry_run=args.dry_run,
        )
        return 0

    fix_dataset_textures(
        input_root=args.input_root,
        subsets=args.subsets,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
