#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[ERROR] OPENAI_API_KEY is not set"
  exit 1
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m osmabench_pp.prompts.prompt_gt_qa \
  --csv data/scenes/scene_prompts.csv \
  --output-dir data/vqa/promptgt_questions \
  --osma-vqa-root data/osma_vqa_workdir \
  --prefer-prompt-field house_prompt
