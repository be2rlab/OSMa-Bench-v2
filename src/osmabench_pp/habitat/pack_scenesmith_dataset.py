#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from osmabench_pp.habitat.scenesmith_to_habitat import create_habitat_scene_dataset


DATASET_CONFIG_NAME = "sceneSmith.scene_dataset_config.json"
LEXICON_REL_PATH = Path("configs") / "ssd" / "sceneSmith_semantic_lexicon.json"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def infer_split_name(scene_root: Path) -> str:
    parent = scene_root.parent.name
    if parent:
        return safe_name(parent)
    return "scenes"


def discover_scene_roots(input_roots: Iterable[Path]) -> List[Tuple[str, Path]]:
    scenes: List[Tuple[str, Path]] = []

    for input_root in input_roots:
        root = input_root.expanduser().resolve()

        if not root.exists():
            print(f"[WARN] missing input root: {root}")
            continue

        # Case 1: input root is directly furniture_stage/ or manipuland_stage/.
        direct_scenes = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("scene_")
        )

        if direct_scenes:
            split_name = safe_name(root.name)
            for scene_root in direct_scenes:
                scenes.append((split_name, scene_root.resolve()))
            continue

        # Case 2: input root contains furniture_stage/ and/or manipuland_stage/.
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            split_scenes = sorted(
                p for p in split_dir.iterdir()
                if p.is_dir() and p.name.startswith("scene_")
            )
            if not split_scenes:
                continue

            split_name = safe_name(split_dir.name)
            for scene_root in split_scenes:
                scenes.append((split_name, scene_root.resolve()))

    unique: List[Tuple[str, Path]] = []
    seen = set()
    for split_name, scene_root in scenes:
        key = str(scene_root)
        if key in seen:
            continue
        seen.add(key)
        unique.append((split_name, scene_root))

    return unique


def copy_tree_files(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def rename_object_template(template_name: str, packed_scene_name: str) -> str:
    return f"{packed_scene_name}__{template_name}"


def load_single_scene_export(scene_root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="osmapp_scene_export_") as tmp:
        tmp_root = Path(tmp)

        dataset_config_path, scene_instance_path = create_habitat_scene_dataset(
            scene_root=scene_root,
            out_root=tmp_root,
        )

        debug_summary = read_json(tmp_root / "debug_summary.json")

        persistent_tmp = Path(tempfile.mkdtemp(prefix="osmapp_scene_export_keep_"))
        shutil.copytree(tmp_root, persistent_tmp, dirs_exist_ok=True)

    return (
        persistent_tmp / dataset_config_path.relative_to(tmp_root),
        persistent_tmp / scene_instance_path.relative_to(tmp_root),
        read_json(persistent_tmp / "debug_summary.json"),
    )


def collect_categories_from_debug(debug_summary: Dict[str, Any]) -> List[str]:
    categories: List[str] = []

    templates = debug_summary.get("templates", {})
    for cfg in templates.values():
        category = cfg.get("_meta_category")
        if category:
            categories.append(str(category))

    if not categories:
        categories.append("object")

    return categories


def build_semantic_lexicon(categories: Iterable[str]) -> Dict[str, Any]:
    clean = sorted({str(c).strip().lower() for c in categories if str(c).strip()})
    if "other" in clean:
        clean.remove("other")

    classes = [{"id": 0, "name": "other"}]
    for idx, name in enumerate(clean, start=1):
        classes.append({"id": idx, "name": name})

    return {"classes": classes}


def write_hadage_sim_settings(
    out_root: Path,
    manifest_scenes: List[Dict[str, Any]],
    sim_settings_dirname: str = "hadage_sim_settings",
) -> None:
    sim_dir = out_root / sim_settings_dirname
    sim_dir.mkdir(parents=True, exist_ok=True)

    lighting_variants = [
        {
            "label": "baseline",
            "light_settings_filename": "light_settings_baseline.json",
            "override_scene_light_defaults": False,
        },
        {
            "label": "camera",
            "light_settings_filename": "light_settings_camera.json",
            "override_scene_light_defaults": True,
        },
        {
            "label": "dynamic_lights",
            "light_settings_filename": "light_settings_dynamic_lights.json",
            "override_scene_light_defaults": True,
        },
        {
            "label": "no_lights",
            "light_settings_filename": "light_settings_no_lights.json",
            "override_scene_light_defaults": True,
        },
    ]

    base_cfg = {
        "dataset_name": "scenesmith",
        "width": 1200,
        "height": 680,
        "hfov": 90,
        "zfar": 1000.0,
        "clear_color": [0.0, 0.0, 0.0, 1.0],
        "sensor_height": 1.0,
        "depth_scale": 6553.5,
        "nav_points_number": 5,
        "move_actuation_amount": 0.06,
        "turn_actuation_amount": 4.0,
        "move_freq_multiplier": 2,
        "turn_freq_multiplier": 2,
        "default_agent": 0,
        "agent_radius": 0.15,
        "scene_light_setup": "default",
        "color_sensor": True,
        "semantic_sensor": True,
        "depth_sensor": True,
        "ortho_rgba_sensor": False,
        "ortho_depth_sensor": False,
        "ortho_semantic_sensor": False,
        "fisheye_rgba_sensor": False,
        "fisheye_depth_sensor": False,
        "fisheye_semantic_sensor": False,
        "equirect_rgba_sensor": False,
        "equirect_depth_sensor": False,
        "equirect_semantic_sensor": False,
        "seed": 73,
        "enable_physics": False,
        "default_agent_navmesh": True,
        "navmesh_include_static_objects": True,
        "enable_hbao": True,
    }

    idx = 0
    for scene_info in manifest_scenes:
        packed_scene_name = scene_info["packed_scene_name"]
        split_name = scene_info["split"]

        nav_points_number = 10 if "manipuland" in split_name else 5

        for variant in lighting_variants:
            cfg = dict(base_cfg)
            cfg["scene_name"] = packed_scene_name
            cfg["label"] = variant["label"]
            cfg["light_settings_filename"] = variant["light_settings_filename"]
            cfg["override_scene_light_defaults"] = variant["override_scene_light_defaults"]
            cfg["nav_points_number"] = nav_points_number

            out_name = f"sim_settings_{idx:03d}_{packed_scene_name}__{variant['label']}.json"
            write_json(sim_dir / out_name, cfg)
            idx += 1

    print(f"[OK] HaDaGe sim settings: {sim_dir}")


def pack_dataset(
    input_roots: List[Path],
    out_root: Path,
    write_hadage: bool = False,
) -> None:
    out_root = out_root.expanduser().resolve()

    stages_dir = out_root / "stages"
    objects_dir = out_root / "objects"
    scenes_dir = out_root / "scenes"
    debug_dir = out_root / "debug"

    out_root.mkdir(parents=True, exist_ok=True)
    stages_dir.mkdir(parents=True, exist_ok=True)
    objects_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    scene_roots = discover_scene_roots(input_roots)
    if not scene_roots:
        raise RuntimeError("No scene_* folders found in input roots.")

    manifest_scenes: List[Dict[str, Any]] = []
    all_categories: List[str] = []
    used_names = set()

    for split_name, scene_root in scene_roots:
        packed_scene_name = safe_name(f"{split_name}__{scene_root.name}")

        if packed_scene_name in used_names:
            raise RuntimeError(f"Duplicate packed scene name: {packed_scene_name}")
        used_names.add(packed_scene_name)

        print(f"[PACK] {scene_root} -> {packed_scene_name}")

        dataset_config_path, scene_instance_path, debug_summary = load_single_scene_export(scene_root)

        temp_root = dataset_config_path.parent
        temp_stages = temp_root / "stages"
        temp_objects = temp_root / "objects"
        temp_scenes = temp_root / "scenes"

        temp_stage_files = sorted(temp_stages.glob("*.stage_config.json"))
        temp_scene_files = sorted(temp_scenes.glob("*.scene_instance.json"))

        if len(temp_stage_files) != 1:
            raise RuntimeError(f"Expected exactly one stage config for {scene_root}, got {len(temp_stage_files)}")
        if len(temp_scene_files) != 1:
            raise RuntimeError(f"Expected exactly one scene instance for {scene_root}, got {len(temp_scene_files)}")

        stage_cfg = read_json(temp_stage_files[0])
        scene_instance = read_json(temp_scene_files[0])

        old_stage_template = scene_instance["stage_instance"]["template_name"]
        scene_instance["stage_instance"]["template_name"] = packed_scene_name

        object_rename: Dict[str, str] = {}
        for obj_cfg_path in sorted(temp_objects.glob("*.object_config.json")):
            old_name = obj_cfg_path.name.removesuffix(".object_config.json")
            new_name = rename_object_template(old_name, packed_scene_name)
            object_rename[old_name] = new_name

            obj_cfg = read_json(obj_cfg_path)
            write_json(objects_dir / f"{new_name}.object_config.json", obj_cfg)

        for obj_inst in scene_instance.get("object_instances", []):
            old_template = obj_inst.get("template_name")
            if old_template in object_rename:
                obj_inst["template_name"] = object_rename[old_template]

        write_json(stages_dir / f"{packed_scene_name}.stage_config.json", stage_cfg)
        write_json(scenes_dir / f"{packed_scene_name}.scene_instance.json", scene_instance)
        write_json(debug_dir / f"{packed_scene_name}.debug_summary.json", debug_summary)

        all_categories.extend(collect_categories_from_debug(debug_summary))

        manifest_scenes.append(
            {
                "packed_scene_name": packed_scene_name,
                "split": split_name,
                "scene_root": str(scene_root),
                "stage_config": str(stages_dir / f"{packed_scene_name}.stage_config.json"),
                "scene_instance": str(scenes_dir / f"{packed_scene_name}.scene_instance.json"),
                "num_object_templates": len(object_rename),
                "num_object_instances": len(scene_instance.get("object_instances", [])),
            }
        )

        shutil.rmtree(temp_root, ignore_errors=True)

    dataset_config = {
        "stages": {"paths": {".stage_config.json": ["stages/"]}},
        "objects": {"paths": {".object_config.json": ["objects/"]}},
        "scene_instances": {"paths": {".scene_instance.json": ["scenes/"]}},
    }
    write_json(out_root / DATASET_CONFIG_NAME, dataset_config)

    lexicon = build_semantic_lexicon(all_categories)
    write_json(out_root / LEXICON_REL_PATH, lexicon)

    manifest = {
        "dataset_name": "sceneSmith",
        "dataset_config": str(out_root / DATASET_CONFIG_NAME),
        "semantic_lexicon": str(out_root / LEXICON_REL_PATH),
        "scenes": manifest_scenes,
    }
    write_json(out_root / "scenesmith_manifest.json", manifest)

    if write_hadage:
        write_hadage_sim_settings(out_root, manifest_scenes)

    print("[OK] packed dataset")
    print(f"Dataset config : {out_root / DATASET_CONFIG_NAME}")
    print(f"Manifest       : {out_root / 'scenesmith_manifest.json'}")
    print(f"Scenes         : {len(manifest_scenes)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack exported SceneSmith scenes into one Habitat SceneDataset-style dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Input root. Can be a split folder like ~/Scenes_SceneSmith/furniture_stage "
            "or a parent folder containing furniture_stage/manipuland_stage. Can be repeated."
        ),
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--write-hadage-sim-settings",
        action="store_true",
        help="Also write HaDaGe sim_settings JSON files for each packed scene.",
    )
    args = parser.parse_args()

    pack_dataset(
        input_roots=args.input_root,
        out_root=args.out_root,
        write_hadage=args.write_hadage_sim_settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
