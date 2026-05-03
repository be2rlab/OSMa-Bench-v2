#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HADAGE_ROOT="$ROOT/data/external/habitat_data_generator"
PATCH="$ROOT/patches/hadage_scenesmith_support.patch"

if [ ! -d "$HADAGE_ROOT/.git" ]; then
  echo "[ERROR] HaDaGe repository not found: $HADAGE_ROOT"
  exit 1
fi

cd "$HADAGE_ROOT"

if git apply --check "$PATCH"; then
  git apply "$PATCH"
  echo "[OK] patch applied"
else
  echo "[WARN] patch cannot be applied cleanly. It may already be applied."
  git diff --stat || true
fi

grep -R "sceneSmith.scene_dataset_config.json" -n hadage/core/settings.py
grep -R "sceneSmith_semantic_lexicon.json" -n hadage/core/settings.py
