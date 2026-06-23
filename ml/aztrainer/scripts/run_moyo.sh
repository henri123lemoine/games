#!/usr/bin/env bash
# Overnight 19x19-focused fine-tune (branch azgo-moyo). Warm-starts the
# iter-1020 net (gnugo-level at 9x9, sub-gnugo at 19x19) and specializes it on
# 19x19 with board-area-scaled opening exploration and wider komi. Favors many
# moderate-depth iters over fewer deep ones (deep reading is handled at play
# time by the 1500-sim deploy bot). caffeinate keeps the Mac awake; keep it on
# AC power. Checkpoints every iter, so stopping early still leaves a usable net.
#   metrics:  data/azgo/run_moyo/metrics.jsonl
#   stop:     touch data/azgo/run_moyo/STOP   (graceful) or kill the process
set -euo pipefail
cd "$(dirname "$0")/.."

# Launch via `cargo run` (not the bare binary): torch-sys writes no LC_RPATH on
# macOS, so the bare binary can't find libtorch; cargo sets the dylib path.
exec caffeinate -s cargo run --release --bin go -- run \
  --dir ../../data/azgo/run_moyo \
  --init-from ../../data/azgo/run_full/latest_swa.ot \
  --blocks 6 --ch 96 \
  --sizes 9,13,19 --size-weights 1,1,3 \
  --sims 192 --full-prob 0.25 --full-sims 320 --fast-sims 100 --forced-k 2.0 \
  --alpha 0.15 --temp-plies 10 --komi-range 5 \
  --replay 400000 --reuse 1.8 --samples-per-iter 4096 --batch 1024 \
  --concurrent 768 \
  --optimizer sgd --lr 0.01 --momentum 0.9 --wd 0.0001 --grad-clip 4.0 --warmup-iters 8 \
  --swa-decay 0.99 --value-mix 0.30 \
  --eval-every 1000000 --eval-pairs 1 --eval-sims 100 --snapshot-every 10 \
  --hours 22
