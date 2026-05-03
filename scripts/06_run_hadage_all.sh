#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OSMAPP_ROOT="$ROOT"
export HADAGE_ROOT="$ROOT/data/external/habitat_data_generator"
export SCENES_PACKED="$ROOT/data/scenes/scenesmith_packed"
export HADAGE_DATA="$ROOT/data/hadage_data"
export HADAGE_OUTPUT="$ROOT/outputs/hadage_generated"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

mkdir -p "$HADAGE_DATA" "$HADAGE_OUTPUT"
ln -sfn "$SCENES_PACKED" "$HADAGE_DATA/scenesmith"

cd "$HADAGE_ROOT"

for SIM_SETTINGS in $(find "$SCENES_PACKED/hadage_sim_settings" -name "*.json" | sort); do
  echo "============================================================"
  echo "$SIM_SETTINGS"
  echo "============================================================"

  python generate.py "$SIM_SETTINGS" \
    --package_dir_path "$HADAGE_ROOT" \
    --data_dir "$HADAGE_DATA" \
    --output_path "$HADAGE_OUTPUT"
done
