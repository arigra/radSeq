#!/usr/bin/env bash
# Build the real RADIal 8-frame sequence caches. Resumable per chunk.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(command -v python3)}"
LOG=logs/radial_cache.log
if [ -f data/radial_seq/DONE ]; then echo "$(date '+%F %T') | already built" >> "$LOG"; exit 0; fi
echo "$(date '+%F %T') | start" >> "$LOG"
PYTHONPATH=. "$PY" scripts/build_radial_cache.py >> "$LOG" 2>&1
echo "$(date '+%F %T') | DONE" >> "$LOG"
