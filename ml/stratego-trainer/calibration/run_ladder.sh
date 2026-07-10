#!/usr/bin/env bash
# Full DoI-anchored ladder measurement: a DoI self-ladder (adjacent-level
# pairs 0v4, 4v8, 8v12) plus checkpoint-vs-DoI matches for every checkpoint
# passed via --ckpts. Appends one JSON line per match to
# calibration/ladder_results.jsonl and is idempotent: a match already present
# in that file (matched on its *identifying* fields, not its outcome) is
# skipped, so a killed/resumed run never re-plays or duplicates a match.
#
# Usage:
#   calibration/run_ladder.sh --ckpts runs/marathon_r1/ckpt_100.safetensors,...,runs/marathon_r2/ckpt_1400.safetensors
#
# Flags:
#   --ckpts PATHS       comma-separated checkpoint paths, in x-axis order
#                        (marathon_r1 first, then marathon_r2)
#   --games-self N       games per DoI self-ladder pair (default 24)
#   --games-ckpt N       games per checkpoint-vs-DoI-level match (default 20)
#   --extra-level-frac F later fraction of --ckpts that also plays DoI level 12
#                        (default 0.5, i.e. the later half of the list)
#   --results PATH       ladder_results.jsonl path (default calibration/ladder_results.jsonl)
#   --seed N             base seed (default 0)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=".venv/bin/python"
CAL="calibration"

GAMES_SELF=24
GAMES_CKPT=20
EXTRA_LEVEL_FRAC=0.5
RESULTS="$CAL/ladder_results.jsonl"
SEED=0
CKPTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpts) CKPTS="$2"; shift 2 ;;
    --games-self) GAMES_SELF="$2"; shift 2 ;;
    --games-ckpt) GAMES_CKPT="$2"; shift 2 ;;
    --extra-level-frac) EXTRA_LEVEL_FRAC="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

touch "$RESULTS"

self_match_exists() {
  local a="$1" b="$2"
  $PY - "$RESULTS" "$a" "$b" <<'EOF'
import json, sys
path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for line in open(path):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    if rec.get("level_a") == a and rec.get("level_b") == b:
        sys.exit(0)
sys.exit(1)
EOF
}

ckpt_match_exists() {
  local ckpt="$1" level="$2"
  $PY - "$RESULTS" "$ckpt" "$level" <<'EOF'
import json, sys
path, ckpt, level = sys.argv[1], sys.argv[2], int(sys.argv[3])
for line in open(path):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    if rec.get("ckpt") == ckpt and rec.get("doi_level") == level:
        sys.exit(0)
sys.exit(1)
EOF
}

echo "=== DoI self-ladder ===" >&2
for pair in "0 4" "4 8" "8 12"; do
  set -- $pair
  a="$1"; b="$2"
  if self_match_exists "$a" "$b"; then
    echo "skip: doi_l$a vs doi_l$b already in $RESULTS" >&2
    continue
  fi
  echo "playing: doi_l$a vs doi_l$b ($GAMES_SELF games)" >&2
  $PY $CAL/doi_vs_doi.py --level-a "$a" --level-b "$b" --games "$GAMES_SELF" --seed "$SEED" >> "$RESULTS"
done

if [[ -z "$CKPTS" ]]; then
  echo "no --ckpts given, skipping checkpoint-vs-DoI matches" >&2
  exit 0
fi

IFS=',' read -ra CKPT_ARR <<< "$CKPTS"
n_ckpts=${#CKPT_ARR[@]}
extra_start=$($PY -c "import math; print(math.floor($n_ckpts * (1 - $EXTRA_LEVEL_FRAC)))")

echo "=== checkpoint vs DoI ladder ($n_ckpts checkpoints) ===" >&2
idx=0
for ckpt in "${CKPT_ARR[@]}"; do
  levels="0 4 8"
  if [[ "$idx" -ge "$extra_start" ]]; then
    levels="0 4 8 12"
  fi
  for level in $levels; do
    if ckpt_match_exists "$ckpt" "$level"; then
      echo "skip: $ckpt vs doi_l$level already in $RESULTS" >&2
      continue
    fi
    echo "playing: $ckpt vs doi_l$level ($GAMES_CKPT games)" >&2
    result=$($PY $CAL/eval_vs_doi.py --ckpt "$ckpt" --games "$GAMES_CKPT" --doi-level "$level" --seed "$SEED")
    echo "$result" | $PY -c "
import json, sys
rec = json.loads(sys.stdin.read())
rec['ckpt'] = sys.argv[1]
rec['doi_level'] = int(sys.argv[2])
print(json.dumps(rec))
" "$ckpt" "$level" >> "$RESULTS"
  done
  idx=$((idx + 1))
done
