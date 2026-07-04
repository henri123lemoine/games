#!/usr/bin/env bash
# League driver for Liar's Dice: train a champion against a pool of opponents
# (rollout bots, the belief agent, and past champions), measure it on the real
# field harness (games/liars-dice/examples/tournament.rs), probe how exploitable
# it still is with a dedicated best-response, and promote it into next round's
# pool. No CFR anywhere in this loop — data generation is plain self-play
# forward passes (ml/ld-ppo), so it isn't rate-limited by the ReBeL-era
# CFR-per-move bottleneck that capped that effort at ~80 samples/sec.
#
# This is a starting point, not a tuned config: BENCHMARK actual iters/sec at
# your target players/dice/faces before committing to a long ROUNDS x
# ITERS_PER_ROUND run — a `rollout:N` opponent in the pool costs N full
# playouts per decision, and that cost grows with both N and game length (5p5d6f
# games run much longer than the tiny smoke configs this script defaults to
# tuning down for). Watch the first round's `trans/s` in the training log and
# adjust the rollout weights/counts before trusting the schedule below.
#
# Usage: ml/ld-ppo/scripts/league.sh [players] [dice] [faces] [rounds] [iters_per_round] [exploiter_iters] [input]
#   input: flat | history (default history — the anti-gullibility bet needs the
#   full bid line the flat per-round summary collapses away)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PLAYERS="${1:-5}"
DICE="${2:-5}"
FACES="${3:-6}"
ROUNDS="${4:-10}"
ITERS_PER_ROUND="${5:-500}"
EXPLOITER_ITERS="${6:-300}"
INPUT="${7:-history}"

if [[ "$INPUT" != "flat" && "$INPUT" != "history" ]]; then
  echo "input must be flat or history, got '$INPUT'" >&2
  exit 1
fi

RUN="$ROOT/runs/ld_league_$(date -u +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
mkdir -p "$RUN"
MANIFEST="$RUN/manifest.jsonl"

# Tournament agent/flag names differ by input mode (both already exist in
# tournament.rs; this script adds no new eval code, only picks the right one).
if [[ "$INPUT" == "history" ]]; then
  NET_FLAG="history"
  NET_AGENTS="history"
  MULTI_AGENTS="histories"
  MULTI_FLAG="histories"
else
  NET_FLAG="ppo"
  NET_AGENTS="ppo"
  MULTI_AGENTS="ppos"
  MULTI_FLAG="ppos"
fi

echo "league: ${PLAYERS}p${DICE}d${FACES}f input=$INPUT, $ROUNDS rounds x $ITERS_PER_ROUND iters, exploiter probes x $EXPLOITER_ITERS iters -> $RUN"

POOL="self=2,rollout:128=2,rollout:256=1,belief=1"

for round in $(seq 1 "$ROUNDS"); do
  CHAMP_DIR="$RUN/champion_round${round}"
  echo "=== round $round: training champion vs pool [$POOL] -> $CHAMP_DIR ==="
  cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    iters="$ITERS_PER_ROUND" opponents="$POOL" input="$INPUT" \
    outdir="$CHAMP_DIR" \
    2>&1 | tee "$RUN/train_round${round}.log"
  CHAMP="$CHAMP_DIR/best.bin"

  echo "=== round $round: field eval (tournament, hero rotated, fair=1/${PLAYERS}) ==="
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    "agents=belief,rollout,${NET_AGENTS}" "${NET_FLAG}=${CHAMP}" \
    2>&1 | tee "$RUN/tournament_round${round}.log"

  echo "=== round $round: exploiter probe (best-response trained vs the frozen champion) ==="
  EXPLOITER_DIR="$RUN/exploiter_round${round}"
  cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    iters="$EXPLOITER_ITERS" "opponents=ckpt:${CHAMP}=1" input="$INPUT" \
    outdir="$EXPLOITER_DIR" \
    2>&1 | tee "$RUN/exploiter_round${round}.log"
  EXPLOITER="$EXPLOITER_DIR/best.bin"

  echo "=== round $round: exploiter edge (exploiter vs champion-only field — the anti-exploitability metric) ==="
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    "agents=${MULTI_AGENTS}" "${MULTI_FLAG}=exploiter:${EXPLOITER},champion:${CHAMP}" \
    2>&1 | tee "$RUN/exploiter_eval_round${round}.log"

  printf '{"round":%d,"champion":"%s","exploiter":"%s","pool":"%s"}\n' \
    "$round" "$CHAMP" "$EXPLOITER" "$POOL" >>"$MANIFEST"

  # Promote this round's champion into next round's pool as a frozen opponent,
  # alongside the rollout/belief bots, so future champions can't just overfit
  # the newest self. Read the exploiter-edge win-share printed above: if the
  # exploiter's edge over the champion is small (close to the champion's own
  # field win-share, not blowing past it), the champion isn't obviously
  # exploitable yet.
  POOL="self=2,rollout:128=1,rollout:256=1,belief=1,ckpt:${CHAMP}=1"
done

echo "league done. manifest: $MANIFEST"
