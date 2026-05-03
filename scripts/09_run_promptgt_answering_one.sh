#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 <scene_name> <graph_json> <method> <condition>"
  echo "Example: $0 furniture_stage__scene_001 /path/to/graph.json BBQ baseline"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"
GRAPH="$2"
METHOD="$3"
CONDITION="$4"

OSMA_VQA_ROOT="$ROOT/data/external/OSMa-Bench/vqa"
QUESTIONS="$ROOT/data/osma_vqa_workdir/$SCENE/vqa/${SCENE}_questions.json"
OUT="$ROOT/outputs/vqa_promptgt/$METHOD/evaluated_${CONDITION}/${SCENE}_answered.json"

mkdir -p "$(dirname "$OUT")"

cd "$OSMA_VQA_ROOT"

python -m src.evaluation.scene_graph_answering \
  config/gemini_qa.yml \
  --questions "$QUESTIONS" \
  --graph "$GRAPH" \
  --output "$OUT"

echo "[OK] answered: $OUT"
