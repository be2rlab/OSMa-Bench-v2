#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <packed_scene_name>"
  echo "Example: $0 furniture_stage__scene_001"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_NAME="$1"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATASET_CONFIG="$ROOT/data/scenes/scenesmith_packed/sceneSmith.scene_dataset_config.json"
SCENE_INSTANCE="$ROOT/data/scenes/scenesmith_packed/scenes/${SCENE_NAME}.scene_instance.json"
OUT="$ROOT/outputs/previews/${SCENE_NAME}.png"

mkdir -p "$(dirname "$OUT")"

python -m osmabench_pp.habitat.render_preview \
  --dataset-config "$DATASET_CONFIG" \
  --scene-id "$SCENE_INSTANCE" \
  --output "$OUT" \
  --width 1280 \
  --height 720 \
  --height-scale 1.4 \
  --distance-scale 2.2 \
  --target-height-scale 0.45

echo "[OK] preview: $OUT"
