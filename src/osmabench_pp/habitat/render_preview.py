#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_magnum


def compute_camera_pose(sim, height_scale: float = 1.6, distance_scale: float = 1.7, target_height_scale: float = 0.25):
    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb

    center = bb.center()
    size = bb.size()

    center_np = np.array([center.x, center.y, center.z], dtype=float)
    size_np = np.array([size.x, size.y, size.z], dtype=float)

    radius = float(max(size_np[0], size_np[2], 1.0))
    scene_height = float(max(size_np[1], 1.0))

    eye = mn.Vector3(
        float(center_np[0] + 0.15 * radius),
        float(center_np[1] + height_scale * scene_height + 1.0),
        float(center_np[2] + distance_scale * radius),
    )

    target = mn.Vector3(
        float(center_np[0]),
        float(center_np[1] + target_height_scale * scene_height),
        float(center_np[2]),
    )

    cam_T = mn.Matrix4.look_at(eye, target, mn.Vector3(0.0, 1.0, 0.0))
    rot = mn.Quaternion.from_matrix(cam_T.rotation())
    return np.array([eye.x, eye.y, eye.z], dtype=np.float32), quat_from_magnum(rot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, required=True,
                        help="Scene handle or absolute path to .scene_instance.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--height-scale", type=float, default=2.0)
    parser.add_argument("--distance-scale", type=float, default=1.5)
    parser.add_argument("--target-height-scale", type=float, default=0.15)
    args = parser.parse_args()

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(args.dataset_config.resolve())
    sim_cfg.scene_id = args.scene_id

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [args.height, args.width]
    color_spec.position = [0.0, 0.0, 0.0]
    color_spec.hfov = 75

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()

    pos, rot = compute_camera_pose(
    sim,
    height_scale=args.height_scale,
    distance_scale=args.distance_scale,
    target_height_scale=args.target_height_scale,
    )
    state.position = pos
    state.rotation = rot
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    color = obs["color_sensor"][..., :3]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(args.output)

    print(f"[OK] saved preview: {args.output}")
    print(f"camera_position: {pos.tolist()}")

    sim.close()


if __name__ == "__main__":
    main()
