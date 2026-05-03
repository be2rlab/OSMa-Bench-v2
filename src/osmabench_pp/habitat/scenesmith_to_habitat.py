#!/usr/bin/env python3
"""
Convert one exported SceneSmith scene into Habitat SceneDataset-style configs.

Input scene must already contain:
- floor_plans/
- combined_house/house.dmd.yaml
- mujoco/scene.xml
- mujoco/meshes/
- mujoco/usd/Payload/Geometry.usda

This converter does not run Habitat. It only writes JSON configs.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# SceneSmith/MuJoCo -> Habitat basis change.
# MuJoCo: x forward-ish, y lateral-ish, z up.
# Habitat: x right, y up, z backward/forward.
MUJOCO_TO_HABITAT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)

# Extra rotation used by the previous working prototype.
RX_NEG_90 = [0.70710678, -0.70710678, 0.0, 0.0]

FLOOR_Y_OFFSET = 0.002
WALL_Y_OFFSET = 0.0


class AngleAxisLoader(yaml.SafeLoader):
    """YAML loader that tolerates Drake !AngleAxis tags."""


def _angle_axis_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
    return loader.construct_mapping(node, deep=True)


AngleAxisLoader.add_constructor("!AngleAxis", _angle_axis_constructor)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def qmul(q1: List[float], q2: List[float]) -> List[float]:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_wxyz_from_axis_angle(axis_xyz: List[float], angle_deg: float) -> List[float]:
    axis = np.asarray(axis_xyz, dtype=float)
    axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(np.deg2rad(angle_deg) * axis)
    x, y, z, w = rot.as_quat()
    return [float(w), float(x), float(y), float(z)]


def mujoco_quat_to_matrix(q_wxyz: List[float]) -> np.ndarray:
    w, x, y, z = q_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quat_wxyz(matrix: np.ndarray) -> List[float]:
    x, y, z, w = R.from_matrix(matrix).as_quat()
    return [float(w), float(x), float(y), float(z)]


def convert_pose_mujoco_to_habitat(
    pos_mujoco: List[float],
    quat_mujoco: List[float],
) -> Tuple[List[float], List[float]]:
    """Convert a MuJoCo pose to Habitat coordinates."""
    t_mj = np.asarray(pos_mujoco, dtype=float)
    r_mj = mujoco_quat_to_matrix(quat_mujoco)

    t_hab = MUJOCO_TO_HABITAT @ t_mj
    r_hab = MUJOCO_TO_HABITAT @ r_mj @ MUJOCO_TO_HABITAT.T

    return t_hab.tolist(), matrix_to_quat_wxyz(r_hab)


def shader_type_for_obj(obj_path: Path) -> str:
    """Use material shader when OBJ has a valid MTL texture, otherwise phong."""
    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.exists():
        return "phong"

    text = mtl_path.read_text(encoding="utf-8", errors="ignore")
    if "map_Kd" in text or "map_Ka" in text:
        return "material"

    return "phong"


def parse_room_name(scene_root: Path) -> str:
    floor_root = scene_root / "floor_plans"
    rooms = sorted(p.name for p in floor_root.iterdir() if p.is_dir()) if floor_root.exists() else []

    # Ignore SceneSmith internal render folders.
    rooms = [r for r in rooms if not r.startswith("floor_plan")]
    if len(rooms) == 1:
        return rooms[0]

    candidates = sorted((scene_root / "room_geometry").glob("room_geometry_*.sdf"))
    if len(candidates) == 1:
        return candidates[0].stem.replace("room_geometry_", "")

    raise RuntimeError(f"Could not infer room name for scene: {scene_root}")


def load_house_instance_names(house_yaml_path: Path) -> List[str]:
    if not house_yaml_path.exists():
        raise FileNotFoundError(f"Missing house DMD file: {house_yaml_path}")

    data = yaml.load(house_yaml_path.read_text(encoding="utf-8"), Loader=AngleAxisLoader)
    names: List[str] = []

    for entry in data.get("directives", []):
        model = entry.get("add_model")
        if not isinstance(model, dict):
            continue

        name = model.get("name")
        file_uri = model.get("file", "")

        if not name:
            continue
        if name.startswith("room_geometry_"):
            continue
        if "/room_geometry/" in file_uri:
            continue

        names.append(str(name))

    return names


def parse_room_root_transform_from_usd(usd_text: str) -> Dict[str, Any]:
    pattern = re.compile(
        r'def Xform "([^"]*room_geometry[^"]*)"\s*\{\s*'
        r'quatf xformOp:orient = \(([^)]+)\)\s*'
        r'float3 xformOp:scale = \(([^)]+)\)\s*'
        r'double3 xformOp:translate = \(([^)]+)\)',
        re.MULTILINE,
    )

    matches = []
    for match in pattern.finditer(usd_text):
        name = match.group(1)
        matches.append(
            {
                "name": name,
                "orient": [float(v.strip()) for v in match.group(2).split(",")],
                "scale": [float(v.strip()) for v in match.group(3).split(",")],
                "translate": [float(v.strip()) for v in match.group(4).split(",")],
            }
        )

    for item in matches:
        name = item["name"]
        if "_body_link" not in name and "_visual" not in name and "_collision" not in name:
            return item

    raise RuntimeError("Could not find room root transform in USD Geometry.usda")


def parse_object_xforms_from_usd(usd_text: str) -> Dict[str, Dict[str, Any]]:
    pattern = re.compile(
        r'def Xform "([^"]+)"\s*\{\s*'
        r'quatf xformOp:orient = \(([^)]+)\)\s*'
        r'float3 xformOp:scale = \(([^)]+)\)\s*'
        r'double3 xformOp:translate = \(([^)]+)\)',
        re.MULTILINE,
    )

    out: Dict[str, Dict[str, Any]] = {}

    for match in pattern.finditer(usd_text):
        name = match.group(1)

        if "_base_link" in name:
            continue
        if "room_geometry_" in name:
            continue
        if name.startswith("scene_"):
            continue

        out[name] = {
            "orient": [float(v.strip()) for v in match.group(2).split(",")],
            "scale": [float(v.strip()) for v in match.group(3).split(",")],
            "translate": [float(v.strip()) for v in match.group(4).split(",")],
        }

    return out


def parse_floor_geom_from_mujoco_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    root = ET.parse(xml_path).getroot()

    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        low = name.lower()

        if "floor" not in low:
            continue
        if "room_geometry" not in low:
            continue
        if "visual" not in low:
            continue

        return {
            "name": name,
            "translate": [float(v) for v in geom.attrib.get("pos", "0 0 0").split()],
            "orient": [float(v) for v in geom.attrib.get("quat", "1 0 0 0").split()],
        }

    return None


def parse_wall_geoms_from_mujoco_xml(xml_path: Path) -> Dict[str, Dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    out: Dict[str, Dict[str, Any]] = {}

    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        low = name.lower()

        if "wall" not in low:
            continue
        if "exterior" in low:
            continue
        if "room_geometry" not in low:
            continue
        if "visual" not in low:
            continue

        if "north_wall" in low:
            token = "north_wall"
        elif "south_wall" in low:
            token = "south_wall"
        elif "east_wall" in low:
            token = "east_wall"
        elif "west_wall" in low:
            token = "west_wall"
        else:
            continue

        out[token] = {
            "name": name,
            "translate": [float(v) for v in geom.attrib.get("pos", "0 0 0").split()],
            "orient": [float(v) for v in geom.attrib.get("quat", "1 0 0 0").split()],
        }

    return out


def wall_extra_rotation_for_token(wall_token: str) -> List[float]:
    if wall_token in {"north_wall", "south_wall"}:
        return quat_wxyz_from_axis_angle([1, 0, 0], -90)
    if wall_token in {"east_wall", "west_wall"}:
        return quat_wxyz_from_axis_angle([0, 0, 1], 90)
    return [1.0, 0.0, 0.0, 0.0]


def find_floor_obj(meshes_dir: Path) -> Optional[Path]:
    candidates = sorted(meshes_dir.glob("*floor*.obj"))
    candidates = [p for p in candidates if "collision" not in p.name.lower()]
    return candidates[0] if candidates else None


def find_wall_obj(meshes_dir: Path, wall_token: str) -> Optional[Path]:
    candidates = sorted(meshes_dir.glob(f"*{wall_token}*wall*.obj"))
    candidates = [p for p in candidates if "exterior" not in p.name.lower()]
    return candidates[0] if candidates else None


def find_visual_meshes_for_instance(
    scene_xml_path: Path,
    meshes_dir: Path,
    instance_name: str,
) -> List[Path]:
    """Find visual OBJ meshes referenced by MuJoCo XML for one SceneSmith instance."""
    root = ET.parse(scene_xml_path).getroot()

    mesh_file_by_name: Dict[str, str] = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            mesh_name = mesh.attrib.get("name", "")
            mesh_file = mesh.attrib.get("file", "")
            if mesh_name and mesh_file:
                mesh_file_by_name[mesh_name] = mesh_file

    out: List[Path] = []
    prefix = f"{instance_name}_"

    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh", "")
        low = mesh_name.lower()

        if not mesh_name.startswith(prefix):
            continue
        if "visual" not in low:
            continue
        if "collision" in low or "vhacd" in low:
            continue

        mesh_file = mesh_file_by_name.get(mesh_name)
        if not mesh_file:
            continue

        mesh_path = meshes_dir / mesh_file
        if mesh_path.exists() and mesh_path.suffix.lower() == ".obj":
            out.append(mesh_path)

    seen = set()
    unique: List[Path] = []
    for path in out:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    return unique


def build_floor_templates_and_instances(scene_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meshes_dir = scene_root / "mujoco" / "meshes"
    xml_path = scene_root / "mujoco" / "scene.xml"

    floor_geom = parse_floor_geom_from_mujoco_xml(xml_path)
    floor_obj = find_floor_obj(meshes_dir)

    if floor_geom is None or floor_obj is None:
        print("[WARN] floor semantic mesh was not found")
        return {}, []

    translation, rotation = convert_pose_mujoco_to_habitat(
        floor_geom["translate"],
        floor_geom["orient"],
    )
    rotation = qmul(RX_NEG_90, rotation)
    translation[1] += FLOOR_Y_OFFSET

    template_name = "semantic_floor"

    templates = {
        template_name: {
            "render_asset": str(floor_obj.resolve()),
            "collision_asset": str(floor_obj.resolve()),
            "mass": 1.0,
            "join_collision_meshes": True,
            "is_collidable": False,
            "semantic_id": 1,
            "scale": [1.0, 1.0, 1.0],
            "shader_type": shader_type_for_obj(floor_obj),
        }
    }

    instances = [
        {
            "template_name": template_name,
            "translation": translation,
            "rotation": rotation,
            "translation_origin": "asset_local",
            "motion_type": "STATIC",
            "_meta_instance_name": floor_geom["name"],
            "_meta_category": "floor",
        }
    ]

    return templates, instances


def build_wall_templates_and_instances(scene_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meshes_dir = scene_root / "mujoco" / "meshes"
    xml_path = scene_root / "mujoco" / "scene.xml"

    wall_geoms = parse_wall_geoms_from_mujoco_xml(xml_path)

    templates: Dict[str, Any] = {}
    instances: List[Dict[str, Any]] = []

    semantic_id = 2

    for wall_token, wall_data in sorted(wall_geoms.items()):
        wall_obj = find_wall_obj(meshes_dir, wall_token)
        if wall_obj is None:
            print(f"[WARN] wall mesh not found: {wall_token}")
            continue

        translation, rotation = convert_pose_mujoco_to_habitat(
            wall_data["translate"],
            wall_data["orient"],
        )
        rotation = qmul(wall_extra_rotation_for_token(wall_token), rotation)
        translation[1] += WALL_Y_OFFSET

        template_name = f"wall_{wall_token}"

        templates[template_name] = {
            "render_asset": str(wall_obj.resolve()),
            "collision_asset": str(wall_obj.resolve()),
            "mass": 1.0,
            "join_collision_meshes": True,
            "is_collidable": True,
            "semantic_id": semantic_id,
            "scale": [1.0, 1.0, 1.0],
            "shader_type": shader_type_for_obj(wall_obj),
        }
        semantic_id += 1

        instances.append(
            {
                "template_name": template_name,
                "translation": translation,
                "rotation": rotation,
                "translation_origin": "asset_local",
                "motion_type": "STATIC",
                "_meta_instance_name": wall_data["name"],
                "_meta_category": "wall",
            }
        )

    return templates, instances


def build_object_templates_and_instances(
    scene_root: Path,
    instance_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meshes_dir = scene_root / "mujoco" / "meshes"
    xml_path = scene_root / "mujoco" / "scene.xml"
    usd_path = scene_root / "mujoco" / "usd" / "Payload" / "Geometry.usda"

    if not usd_path.exists():
        raise FileNotFoundError(f"Missing USD geometry file: {usd_path}")

    usd_text = usd_path.read_text(encoding="utf-8", errors="ignore")
    object_xforms = parse_object_xforms_from_usd(usd_text)
    room_root = parse_room_root_transform_from_usd(usd_text)
    room_tx, room_ty, room_tz = room_root["translate"]

    templates: Dict[str, Any] = {}
    instances: List[Dict[str, Any]] = []

    semantic_id = 1000

    for instance_name in instance_names:
        if instance_name not in object_xforms:
            print(f"[WARN] missing USD transform for instance: {instance_name}")
            continue

        visual_meshes = find_visual_meshes_for_instance(
            scene_xml_path=xml_path,
            meshes_dir=meshes_dir,
            instance_name=instance_name,
        )

        if not visual_meshes:
            print(f"[WARN] no visual OBJ meshes found for instance: {instance_name}")
            continue

        transform = object_xforms[instance_name]
        tx, ty, tz = transform["translate"]
        qw, qx, qy, qz = transform["orient"]
        sx, sy, sz = transform["scale"]

        # Object transforms in USD are relative to room root.
        local_x = tx - room_tx
        local_y = ty - room_ty
        local_z = tz - room_tz

        translation = [local_x, local_z, -local_y]
        rotation = qmul(RX_NEG_90, [qw, qx, qy, qz])

        for mesh_path in visual_meshes:
            template_name = mesh_path.stem

            if template_name not in templates:
                templates[template_name] = {
                    "render_asset": str(mesh_path.resolve()),
                    "collision_asset": str(mesh_path.resolve()),
                    "mass": 1.0,
                    "join_collision_meshes": True,
                    "is_collidable": True,
                    "semantic_id": semantic_id,
                    "scale": [sx, sy, sz],
                    "shader_type": shader_type_for_obj(mesh_path),
                }
                semantic_id += 1

            instances.append(
                {
                    "template_name": template_name,
                    "translation": translation,
                    "rotation": rotation,
                    "translation_origin": "asset_local",
                    "motion_type": "STATIC",
                    "_meta_instance_name": instance_name,
                    "_meta_mesh_part": mesh_path.name,
                    "_meta_category": "object",
                }
            )

    return templates, instances


def create_habitat_scene_dataset(scene_root: Path, out_root: Path) -> Tuple[Path, Path]:
    scene_root = scene_root.resolve()
    out_root = out_root.resolve()

    room_name = parse_room_name(scene_root)
    scene_name = scene_root.name
    house_yaml = scene_root / "combined_house" / "house.dmd.yaml"

    instance_names = load_house_instance_names(house_yaml)

    floor_gltf = scene_root / "floor_plans" / room_name / "floors" / "floor.gltf"
    if not floor_gltf.exists():
        raise FileNotFoundError(f"Missing floor stage asset: {floor_gltf}")

    floor_templates, floor_instances = build_floor_templates_and_instances(scene_root)
    wall_templates, wall_instances = build_wall_templates_and_instances(scene_root)
    object_templates, object_instances = build_object_templates_and_instances(scene_root, instance_names)

    templates = {
        **floor_templates,
        **wall_templates,
        **object_templates,
    }
    instances = floor_instances + wall_instances + object_instances

    dataset_config = {
        "stages": {"paths": {".json": ["stages/"]}},
        "objects": {"paths": {".object_config.json": ["objects/"]}},
        "scene_instances": {"paths": {".scene_instance.json": ["scenes/"]}},
    }

    dataset_config_path = out_root / f"{scene_name}.scene_dataset_config.json"
    write_json(dataset_config_path, dataset_config)

    stage_config = {
        "render_asset": str(floor_gltf.resolve()),
        "collision_asset": str(floor_gltf.resolve()),
        "up": [0, 1, 0],
        "front": [0, 0, -1],
        "origin": [0, 0, 0],
        "scale": [1, 1, 1],
        "motion_type": "STATIC",
        "translation_origin": "asset_local",
    }
    write_json(out_root / "stages" / f"{scene_name}.stage_config.json", stage_config)

    for template_name, config in sorted(templates.items()):
        write_json(out_root / "objects" / f"{template_name}.object_config.json", config)

    scene_instance = {
        "stage_instance": {"template_name": scene_name},
        "object_instances": [
            {k: v for k, v in inst.items() if not k.startswith("_meta_")}
            for inst in instances
        ],
    }

    scene_instance_path = out_root / "scenes" / f"{scene_name}.scene_instance.json"
    write_json(scene_instance_path, scene_instance)

    debug_summary = {
        "scene_root": str(scene_root),
        "room_name": room_name,
        "scene_name": scene_name,
        "num_house_instances": len(instance_names),
        "num_templates": len(templates),
        "num_instances": len(instances),
        "templates": templates,
        "instances": instances,
    }
    write_json(out_root / "debug_summary.json", debug_summary)

    return dataset_config_path, scene_instance_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one exported SceneSmith scene to Habitat SceneDataset-style configs."
    )
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_config_path, scene_instance_path = create_habitat_scene_dataset(
        scene_root=args.scene_root,
        out_root=args.out_root,
    )

    print("[OK] Habitat configs written")
    print(f"Dataset config : {dataset_config_path}")
    print(f"Scene instance : {scene_instance_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
