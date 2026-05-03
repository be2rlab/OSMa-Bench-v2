#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/external

if [ ! -e data/external/scenesmith ]; then
  git clone https://github.com/nepfaff/scenesmith.git data/external/scenesmith
else
  echo "[SKIP] data/external/scenesmith already exists"
fi

if [ ! -e data/external/habitat_data_generator ]; then
  git clone https://github.com/warmhammer/habitat_data_generator.git data/external/habitat_data_generator
else
  echo "[SKIP] data/external/habitat_data_generator already exists"
fi

if [ ! -e data/external/OSMa-Bench ]; then
  git clone https://github.com/be2rlab/OSMa-Bench.git data/external/OSMa-Bench
else
  echo "[SKIP] data/external/OSMa-Bench already exists"
fi

echo "[OK] external repositories are prepared"
