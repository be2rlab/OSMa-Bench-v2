#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m osmabench_pp.prompts.scenesmith_metadata \
  --root data/scenes/scenesmith_raw \
  --output-csv data/scenes/scene_prompts.csv \
  --subsets furniture_stage,manipuland_stage
