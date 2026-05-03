#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STANDARD_ROOT="${1:-}"
PROMPTGT_ROOT="${2:-$ROOT/outputs/vqa_promptgt}"
OUT_DIR="${3:-$ROOT/outputs/vqa_promptgt_summary}"

if [ -z "$STANDARD_ROOT" ]; then
  echo "Usage: $0 <standard_vqa_eval_root> [promptgt_root] [out_dir]"
  echo "Example: $0 /path/to/VQA_EVAL"
  exit 1
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m osmabench_pp.vqa.promptgt_metrics \
  --standard-root "$STANDARD_ROOT" \
  --promptgt-root "$PROMPTGT_ROOT" \
  --out-dir "$OUT_DIR"
