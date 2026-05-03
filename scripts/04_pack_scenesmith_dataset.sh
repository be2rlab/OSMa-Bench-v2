#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OSMAPP_ROOT="$ROOT"
export SCENES_RAW="$ROOT/data/scenes/scenesmith_raw"
export SCENES_PACKED="$ROOT/data/scenes/scenesmith_packed"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

rm -rf "$SCENES_PACKED"
mkdir -p "$SCENES_PACKED"

ARGS=()

if [ -d "$SCENES_RAW/furniture_stage" ]; then
  ARGS+=(--input-root "$SCENES_RAW/furniture_stage")
fi

if [ -d "$SCENES_RAW/manipuland_stage" ]; then
  ARGS+=(--input-root "$SCENES_RAW/manipuland_stage")
fi

if [ ${#ARGS[@]} -eq 0 ]; then
  echo "[ERROR] no input roots found under $SCENES_RAW"
  exit 1
fi

python -m osmabench_pp.habitat.pack_scenesmith_dataset \
  "${ARGS[@]}" \
  --out-root "$SCENES_PACKED" \
  --write-hadage-sim-settings

echo "[OK] packed dataset: $SCENES_PACKED"
