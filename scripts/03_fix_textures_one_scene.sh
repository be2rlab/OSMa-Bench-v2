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
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m osmabench_pp.habitat.texture_fix \
  --scene-root "$SCENE_ROOT"
