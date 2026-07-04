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
STATUS_LOG="$RUN/status.log"
STATUS_JSONL="$RUN/status.jsonl"
HAVE_JQ=0
command -v jq >/dev/null 2>&1 && HAVE_JQ=1

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
  TOURNAMENT_METRICS="$RUN/tournament_round${round}.jsonl"
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    "agents=belief,rollout,${NET_AGENTS}" "${NET_FLAG}=${CHAMP}" \
    "metrics=${TOURNAMENT_METRICS}" \
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
  EXPLOITER_EVAL_METRICS="$RUN/exploiter_eval_round${round}.jsonl"
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    "agents=${MULTI_AGENTS}" "${MULTI_FLAG}=exploiter:${EXPLOITER},champion:${CHAMP}" \
    "metrics=${EXPLOITER_EVAL_METRICS}" \
    2>&1 | tee "$RUN/exploiter_eval_round${round}.log"

  printf '{"round":%d,"champion":"%s","exploiter":"%s","pool":"%s"}\n' \
    "$round" "$CHAMP" "$EXPLOITER" "$POOL" >>"$MANIFEST"

  # Parse this round's numbers straight from the structured metrics (not
  # screen-scraped tables) and append both a machine- and human-readable
  # status line — this is the file to read overnight without re-running
  # anything: field win-share vs fair share, the exploiter's edge over the
  # frozen champion, and whether the belief head's cross-entropy is actually
  # dropping within the round (vs sitting at the uniform-prior baseline).
  CHAMP_LABEL="${NET_AGENTS}-best"
  EXPLOITER_LABEL="${NET_AGENTS}-exploiter"
  CHAMP_VS_LABEL="${NET_AGENTS}-champion"
  FAIR=$(awk "BEGIN{printf \"%.6f\", 1.0/${PLAYERS}}")
  if [[ "$HAVE_JQ" == "1" ]]; then
    FIELD_WS=$(jq -s --arg h "$CHAMP_LABEL" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h) | .win_share] | if length>0 then (add/length) else null end' \
      "$TOURNAMENT_METRICS" 2>/dev/null || echo null)
    EXPLOITER_WS=$(jq -s --arg h "$EXPLOITER_LABEL" --arg f "$CHAMP_VS_LABEL" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h and .field==$f) | .win_share] | if length>0 then .[0] else null end' \
      "$EXPLOITER_EVAL_METRICS" 2>/dev/null || echo null)
    BELIEF_FIRST=$(jq -s '[.[] | select(.event=="ppo_iter") | .belief_loss] | if length>0 then .[0] else null end' \
      "$CHAMP_DIR/metrics.jsonl" 2>/dev/null || echo null)
    BELIEF_LAST=$(jq -s '[.[] | select(.event=="ppo_iter") | .belief_loss] | if length>0 then .[-1] else null end' \
      "$CHAMP_DIR/metrics.jsonl" 2>/dev/null || echo null)
  else
    FIELD_WS=null
    EXPLOITER_WS=null
    BELIEF_FIRST=null
    BELIEF_LAST=null
  fi
  EXPLOITER_EDGE="null"
  if [[ "$EXPLOITER_WS" != "null" ]]; then
    EXPLOITER_EDGE=$(awk "BEGIN{printf \"%.6f\", ${EXPLOITER_WS} - ${FAIR}}")
  fi
  printf '{"round":%d,"champion":"%s","field_win_share":%s,"fair_share":%s,"exploiter_win_share_vs_champion":%s,"exploiter_edge_over_fair":%s,"belief_loss_first":%s,"belief_loss_last":%s}\n' \
    "$round" "$CHAMP" "$FIELD_WS" "$FAIR" "$EXPLOITER_WS" "$EXPLOITER_EDGE" "$BELIEF_FIRST" "$BELIEF_LAST" >>"$STATUS_JSONL"
  {
    echo "round $round  ($(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown))"
    echo "  champion field win-share: ${FIELD_WS} (fair=${FAIR})"
    echo "  exploiter vs champion:    ${EXPLOITER_WS} (edge over fair: ${EXPLOITER_EDGE})"
    echo "  belief loss:              ${BELIEF_FIRST} -> ${BELIEF_LAST}"
  } | tee -a "$STATUS_LOG"

  # Promote this round's champion into next round's pool as a frozen opponent,
  # alongside the rollout/belief bots, so future champions can't just overfit
  # the newest self. Read the exploiter-edge win-share printed above: if the
  # exploiter's edge over the champion is small (close to the champion's own
  # field win-share, not blowing past it), the champion isn't obviously
  # exploitable yet.
  POOL="self=2,rollout:128=1,rollout:256=1,belief=1,ckpt:${CHAMP}=1"
done

echo "league done. manifest: $MANIFEST, status: $STATUS_LOG"
