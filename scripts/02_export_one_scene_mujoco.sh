#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <scene_root>"
  echo "Example: $0 data/scenes/scenesmith_raw/furniture_stage/scene_001"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_ROOT="$1"

cd "$ROOT"

export OSMAPP_ROOT="$ROOT"
export SCENESMITH_ROOT="$ROOT/data/external/scenesmith"
export SCENESMITH_EXPORT="$SCENESMITH_ROOT/scripts/export_scene_to_mujoco.py"
export PYTHONPATH="$ROOT/src:$SCENESMITH_ROOT:${PYTHONPATH:-}"

python -m osmabench_pp.scenesmith.mujoco_export \
  --scene-root "$SCENE_ROOT" \
  --exporter-script "$SCENESMITH_EXPORT" \
  --python "$(which python)" \
  --overwrite
