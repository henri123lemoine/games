#!/usr/bin/env bash
# LD v2 league driver: train a champion against a growing pool of opponents
# (rollout bots, the belief agent, and past exploiters), measure it on the real
# field harness (games/liars-dice/examples/tournament.rs) AND on a fixed
# rollout-768 yardstick, probe how exploitable it still is with a dedicated
# best-response, fold that exploiter into next round's pool, and repeat.
#
# v2 changes over the v1 driver (see git history for v1's own commit messages):
#   1. hidden width is a knob (HIDDEN env var, default 1024) — BENCHMARK actual
#      iters/sec at your target config before trusting a long schedule; a wider
#      net is slower per iteration.
#   2. Keep-best is tracked against a FIXED rollout-768 yardstick (moderate game
#      count, YARDSTICK_GAMES), not the internal training-time field number —
#      v1's internal-field keep-best (the training run's OWN best.bin, judged
#      against a field that itself changes round to round) isn't comparable
#      across rounds and provably picked a worse round than the fixed yardstick
#      would have. The running best-by-yardstick checkpoint is copied to
#      champion_best_overall.bin whenever a new round beats it.
#   3. Exploiter-weighted pool sampling: the RECENT_EXPLOITER_COUNT most
#      recently folded-in exploiters get RECENT_EXPLOITER_WEIGHT, older ones
#      get OLD_EXPLOITER_WEIGHT — recent, more-relevant opponents get sampled
#      more.
#   4. Champion training iters scale with pool size: BASE_ITERS plus
#      ITERS_PER_EXTRA_POOL_ENTRY for every pool entry beyond the initial set
#      (self/rollout:128/rollout:256/belief) — v1 split a fixed iter budget
#      across an ever-growing pool, diluting attention on old opponents.
#   5. Resumable across process restarts / days: RESUME_RUN + START_ROUND env
#      vars continue an existing run dir from wherever it left off, rebuilding
#      the exploiter pool from pool_state.jsonl. The run dir's own README.md
#      (regenerated every round) documents the exact resume command.
#   6. Warm-start: each round's champion training initializes from
#      champion_best_overall.bin (the current best-by-yardstick checkpoint)
#      when one exists, instead of a fresh random net every round. v2 without
#      this trained every round from scratch against an ever-hardening pool —
#      the task got harder each round while the learner kept restarting, and
#      the yardstick oscillated instead of climbing. Read fresh from disk at
#      the top of EVERY round loop iteration (not baked in at launch), same
#      pattern as rayon_threads.txt, so it responds to whichever checkpoint is
#      the running best at that moment — including one from a different
#      process if this run was resumed. The init source used is recorded in
#      manifest.jsonl and status.jsonl for every round (a round's champion
#      might have trained from scratch if no best existed yet, e.g. round 1).
#      NOTE: bash parses this script's `for` loop body once at process start;
#      editing this file does not change an already-running process's
#      behavior for its remaining rounds (verified empirically) — a change
#      here only takes effect for a NEW invocation (fresh launch or a
#      RESUME_RUN/START_ROUND continuation of a stopped run).
#
# v3 changes (2026-07-08, motivated by 9 rounds of yardstick oscillation and a
# round-robin that contradicted the yardstick ranking):
#   7. Keep-best is now gated on a HEAD-TO-HEAD against the incumbent
#      champion_best_overall.bin (H2H_GAMES per cell, both hero/field
#      directions), not the rollout-768 yardstick. Measured 2026-07-08: the
#      round-11 champion lost the yardstick to round 3 (0.410 vs 0.455, within
#      n=200 noise) yet beat it decisively head-to-head (0.220 vs 0.133) and
#      was the least exploitable round ever (edge 0.030) — the scripted
#      yardstick and direct play DISAGREE on ranking, and direct play is the
#      goal. Promotion requires h2h_candidate - h2h_incumbent >= H2H_MARGIN
#      AND exploiter edge <= incumbent's recorded edge + EDGE_SLACK (an
#      unexploitability guard; incumbent edge is persisted in
#      champion_best_overall_edge.txt). The yardstick eval still runs and is
#      logged as a cross-run reference metric.
#   8. Pool is capped at the POOL_MAX_EXPLOITERS most recent exploiters
#      (older ones age out) so round wall-time stops growing without bound —
#      rounds had reached ~8h at pool_size 15 with no yardstick gain to show.
#
# Thread budget: this driver is CPU-only and may be sharing the machine with a
# GPU training job that has priority. It never sets RAYON_NUM_THREADS itself —
# set it in the environment before launching (and see rayon_threads.txt below
# for adjusting it between rounds without restarting the whole driver).
#
# Usage: ml/ld-ppo/scripts/league.sh [players] [dice] [faces] [rounds] [base_iters] [exploiter_iters] [input]
#   input: flat | history (default history)
# Tunable via environment: HIDDEN, ITERS_PER_EXTRA_POOL_ENTRY, YARDSTICK_GAMES,
#   H2H_GAMES, H2H_MARGIN, EDGE_SLACK, POOL_MAX_EXPLOITERS,
#   RECENT_EXPLOITER_WEIGHT, RECENT_EXPLOITER_COUNT, OLD_EXPLOITER_WEIGHT,
#   START_ROUND, RESUME_RUN, START_POOL, RAYON_NUM_THREADS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PLAYERS="${1:-5}"
DICE="${2:-5}"
FACES="${3:-6}"
ROUNDS="${4:-30}"
BASE_ITERS="${5:-500}"
EXPLOITER_ITERS="${6:-300}"
INPUT="${7:-history}"

if [[ "$INPUT" != "flat" && "$INPUT" != "history" ]]; then
  echo "input must be flat or history, got '$INPUT'" >&2
  exit 1
fi

HIDDEN="${HIDDEN:-1024}"
ITERS_PER_EXTRA_POOL_ENTRY="${ITERS_PER_EXTRA_POOL_ENTRY:-150}"
YARDSTICK_GAMES="${YARDSTICK_GAMES:-200}"
H2H_GAMES="${H2H_GAMES:-400}"
H2H_MARGIN="${H2H_MARGIN:-0.05}"
EDGE_SLACK="${EDGE_SLACK:-0.05}"
POOL_MAX_EXPLOITERS="${POOL_MAX_EXPLOITERS:-10}"
RECENT_EXPLOITER_WEIGHT="${RECENT_EXPLOITER_WEIGHT:-2}"
RECENT_EXPLOITER_COUNT="${RECENT_EXPLOITER_COUNT:-2}"
OLD_EXPLOITER_WEIGHT="${OLD_EXPLOITER_WEIGHT:-1}"
START_ROUND="${START_ROUND:-1}"
RESUME_RUN="${RESUME_RUN:-}"
START_POOL="${START_POOL:-}"

INITIAL_POOL="self=2,rollout:128=2,rollout:256=1,belief=1"
INITIAL_POOL_SIZE=4

if [[ -n "$RESUME_RUN" ]]; then
  RUN="$RESUME_RUN"
  mkdir -p "$RUN"
else
  RUN="$ROOT/runs/ld_league_v2_$(date -u +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
  mkdir -p "$RUN"
fi
MANIFEST="$RUN/manifest.jsonl"
STATUS_LOG="$RUN/status.log"
STATUS_JSONL="$RUN/status.jsonl"
POOL_STATE="$RUN/pool_state.jsonl"
README="$RUN/README.md"
THREADS_FILE="$RUN/rayon_threads.txt"

HAVE_JQ=0
command -v jq >/dev/null 2>&1 && HAVE_JQ=1

# rayon_threads.txt is the live thread-budget control: overwrite it from
# another shell to change RAYON_NUM_THREADS for every cargo run invocation
# from the NEXT round onward, without killing this driver. Seeded from the
# environment on first launch (or left absent = rayon's own default).
if [[ ! -f "$THREADS_FILE" && -n "${RAYON_NUM_THREADS:-}" ]]; then
  echo "$RAYON_NUM_THREADS" >"$THREADS_FILE"
fi

# Tournament agent/flag names differ by input mode (both already exist in
# tournament.rs; this driver adds no new eval code, only picks the right one).
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
CHAMP_LABEL="${NET_AGENTS}-best"
EXPLOITER_LABEL="${NET_AGENTS}-exploiter"
CHAMP_VS_LABEL="${NET_AGENTS}-champion"

declare -a EXPLOITERS=()
if [[ -z "$START_POOL" && -f "$POOL_STATE" ]]; then
  while IFS= read -r line; do
    EXPLOITERS+=("$line")
  done < <(jq -r '.checkpoint' "$POOL_STATE" 2>/dev/null || true)
fi

# Rebuild the opponents= string from INITIAL_POOL plus every folded-in
# exploiter, weighting the RECENT_EXPLOITER_COUNT most recent ones higher.
rebuild_pool() {
  local n=${#EXPLOITERS[@]}
  local out="$INITIAL_POOL"
  local i ago w
  for i in "${!EXPLOITERS[@]}"; do
    ago=$((n - 1 - i))
    if ((ago >= POOL_MAX_EXPLOITERS)); then continue; fi
    w="$OLD_EXPLOITER_WEIGHT"
    if ((ago < RECENT_EXPLOITER_COUNT)); then
      w="$RECENT_EXPLOITER_WEIGHT"
    fi
    out="${out},ckpt:${EXPLOITERS[$i]}=${w}"
  done
  echo "$out"
}

if [[ -n "$START_POOL" ]]; then
  POOL="$START_POOL"
else
  POOL="$(rebuild_pool)"
fi

write_readme() {
  local next_round=$1
  local best_round
  best_round=$(cat "$RUN/champion_best_overall_round.txt" 2>/dev/null || echo n/a)
  cat >"$README" <<RM
# LD v2 league run

Resume from the last completed round:

    START_ROUND=${next_round} RESUME_RUN=${RUN} \\
      ml/ld-ppo/scripts/league.sh ${PLAYERS} ${DICE} ${FACES} ${ROUNDS} ${BASE_ITERS} ${EXPLOITER_ITERS} ${INPUT}

(inherits HIDDEN/YARDSTICK_GAMES/exploiter-weight env vars from your original
launch if you re-export them; the pool and round count resume from
pool_state.jsonl and manifest.jsonl automatically.)

Best champion (head-to-head gate, v3) so far: round ${best_round}
  -> champion_best_overall.bin

Current pool (${#EXPLOITERS[@]} exploiters total, most recent ${POOL_MAX_EXPLOITERS} active):
  ${POOL}

Thread budget: edit rayon_threads.txt to change RAYON_NUM_THREADS for rounds
from here on, without restarting this driver.

Status: status.log (human) / status.jsonl (structured). Manifest: manifest.jsonl.
RM
}

echo "league v2: ${PLAYERS}p${DICE}d${FACES}f input=$INPUT hidden=$HIDDEN, rounds ${START_ROUND}..${ROUNDS} base_iters=$BASE_ITERS (+${ITERS_PER_EXTRA_POOL_ENTRY}/extra pool entry) exploiter_iters=$EXPLOITER_ITERS yardstick_games=$YARDSTICK_GAMES -> $RUN"
write_readme "$START_ROUND"

for round in $(seq "$START_ROUND" "$ROUNDS"); do
  if [[ -f "$THREADS_FILE" ]]; then
    export RAYON_NUM_THREADS
    RAYON_NUM_THREADS="$(cat "$THREADS_FILE")"
  fi

  ACTIVE_EXPLOITERS=${#EXPLOITERS[@]}
  if ((ACTIVE_EXPLOITERS > POOL_MAX_EXPLOITERS)); then ACTIVE_EXPLOITERS=$POOL_MAX_EXPLOITERS; fi
  POOL_SIZE=$((INITIAL_POOL_SIZE + ACTIVE_EXPLOITERS))
  EXTRA=$((POOL_SIZE - INITIAL_POOL_SIZE))
  ITERS_PER_ROUND=$((BASE_ITERS + ITERS_PER_EXTRA_POOL_ENTRY * EXTRA))

  # Warm-start from the current best-by-yardstick checkpoint, if one exists
  # yet (round 1, or a resume before any round has completed, trains fresh).
  # Read fresh every round — see the v2.1 note in the header doc.
  BEST_OVERALL="$RUN/champion_best_overall.bin"
  INIT_ARGS=()
  INIT_SOURCE="fresh"
  if [[ -f "$BEST_OVERALL" ]]; then
    INIT_ARGS=("init=${BEST_OVERALL}")
    INIT_SOURCE="$BEST_OVERALL"
  fi

  CHAMP_DIR="$RUN/champion_round${round}"
  echo "=== round $round: training champion (hidden=$HIDDEN pool_size=$POOL_SIZE iters=$ITERS_PER_ROUND rayon=${RAYON_NUM_THREADS:-default} init=$INIT_SOURCE) vs pool [$POOL] -> $CHAMP_DIR ==="
  cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    iters="$ITERS_PER_ROUND" opponents="$POOL" input="$INPUT" hidden="$HIDDEN" \
    "${INIT_ARGS[@]}" \
    outdir="$CHAMP_DIR" \
    2>&1 | tee "$RUN/train_round${round}.log"
  CHAMP="$CHAMP_DIR/best.bin"

  echo "=== round $round: internal field eval (tournament, hero rotated, fair=1/${PLAYERS}) ==="
  TOURNAMENT_METRICS="$RUN/tournament_round${round}.jsonl"
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    "agents=belief,rollout,${NET_AGENTS}" "${NET_FLAG}=${CHAMP}" \
    "metrics=${TOURNAMENT_METRICS}" \
    2>&1 | tee "$RUN/tournament_round${round}.log"

  echo "=== round $round: FIXED yardstick eval (champion vs rollout-768 field, ${YARDSTICK_GAMES} games) ==="
  YARDSTICK_METRICS="$RUN/yardstick_round${round}.jsonl"
  cargo run --release -p liars-dice --example tournament -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" games="$YARDSTICK_GAMES" rollouts=768 \
    "agents=${NET_AGENTS},rollout" "${NET_FLAG}=${CHAMP}" \
    "cells=${CHAMP_LABEL}:rollout-768" \
    "metrics=${YARDSTICK_METRICS}" \
    2>&1 | tee "$RUN/yardstick_round${round}.log"

  echo "=== round $round: exploiter probe (best-response trained vs the frozen champion) ==="
  EXPLOITER_DIR="$RUN/exploiter_round${round}"
  cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
    players="$PLAYERS" dice="$DICE" faces="$FACES" \
    iters="$EXPLOITER_ITERS" "opponents=ckpt:${CHAMP}=1" input="$INPUT" hidden="$HIDDEN" \
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

  echo "=== round $round: head-to-head vs incumbent best (the keep-best gate, ${H2H_GAMES} games/cell) ==="
  H2H_METRICS="$RUN/h2h_round${round}.jsonl"
  if [[ -f "$BEST_OVERALL" ]]; then
    cargo run --release -p liars-dice --example tournament -- \
      players="$PLAYERS" dice="$DICE" faces="$FACES" games="$H2H_GAMES" \
      "agents=${MULTI_AGENTS}" "${MULTI_FLAG}=candidate:${CHAMP},incumbent:${BEST_OVERALL}" \
      "metrics=${H2H_METRICS}" \
      2>&1 | tee "$RUN/h2h_round${round}.log"
  else
    rm -f "$H2H_METRICS"
  fi

  printf '{"round":%d,"champion":"%s","exploiter":"%s","pool":"%s","pool_size":%d,"iters":%d,"init_source":"%s"}\n' \
    "$round" "$CHAMP" "$EXPLOITER" "$POOL" "$POOL_SIZE" "$ITERS_PER_ROUND" "$INIT_SOURCE" >>"$MANIFEST"

  # Parse this round's numbers straight from the structured metrics (not
  # screen-scraped tables): the internal field win-share (not comparable
  # across rounds — the field itself grows), the FIXED rollout-768 yardstick
  # (comparable across every round, and what keep-best is judged on), the
  # exploiter's edge over the frozen champion, and the belief head's
  # within-round loss trend.
  FAIR=$(awk "BEGIN{printf \"%.6f\", 1.0/${PLAYERS}}")
  if [[ "$HAVE_JQ" == "1" ]]; then
    FIELD_WS=$(jq -s --arg h "$CHAMP_LABEL" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h) | .win_share] | if length>0 then (add/length) else null end' \
      "$TOURNAMENT_METRICS" 2>/dev/null || echo null)
    YARDSTICK_WS=$(jq -s --arg h "$CHAMP_LABEL" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h and .field=="rollout-768") | .win_share] | if length>0 then .[0] else null end' \
      "$YARDSTICK_METRICS" 2>/dev/null || echo null)
    EXPLOITER_WS=$(jq -s --arg h "$EXPLOITER_LABEL" --arg f "$CHAMP_VS_LABEL" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h and .field==$f) | .win_share] | if length>0 then .[0] else null end' \
      "$EXPLOITER_EVAL_METRICS" 2>/dev/null || echo null)
    BELIEF_FIRST=$(jq -s '[.[] | select(.event=="ppo_iter") | .belief_loss] | if length>0 then .[0] else null end' \
      "$CHAMP_DIR/metrics.jsonl" 2>/dev/null || echo null)
    BELIEF_LAST=$(jq -s '[.[] | select(.event=="ppo_iter") | .belief_loss] | if length>0 then .[-1] else null end' \
      "$CHAMP_DIR/metrics.jsonl" 2>/dev/null || echo null)
  else
    FIELD_WS=null
    YARDSTICK_WS=null
    EXPLOITER_WS=null
    BELIEF_FIRST=null
    BELIEF_LAST=null
  fi
  EXPLOITER_EDGE="null"
  if [[ "$EXPLOITER_WS" != "null" ]]; then
    EXPLOITER_EDGE=$(awk "BEGIN{printf \"%.6f\", ${EXPLOITER_WS} - ${FAIR}}")
  fi

  # Keep-best (v3): promote only if the candidate beats the incumbent
  # head-to-head by H2H_MARGIN AND is not meaningfully more exploitable.
  H2H_CANDIDATE="null"
  H2H_INCUMBENT="null"
  if [[ -f "$H2H_METRICS" && "$HAVE_JQ" == "1" ]]; then
    H2H_CANDIDATE=$(jq -s --arg h "${NET_AGENTS}-candidate" --arg f "${NET_AGENTS}-incumbent" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h and .field==$f) | .win_share] | if length>0 then .[0] else null end' \
      "$H2H_METRICS" 2>/dev/null || echo null)
    H2H_INCUMBENT=$(jq -s --arg h "${NET_AGENTS}-incumbent" --arg f "${NET_AGENTS}-candidate" \
      '[.[] | select(.event=="tournament_cell" and .hero==$h and .field==$f) | .win_share] | if length>0 then .[0] else null end' \
      "$H2H_METRICS" 2>/dev/null || echo null)
  fi
  EDGE_FILE="$RUN/champion_best_overall_edge.txt"
  IS_NEW_BEST="false"
  if [[ ! -f "$BEST_OVERALL" ]]; then
    IS_NEW_BEST="true"
  elif [[ "$H2H_CANDIDATE" != "null" && "$H2H_INCUMBENT" != "null" ]]; then
    INCUMBENT_EDGE=$(cat "$EDGE_FILE" 2>/dev/null || echo 1.0)
    EDGE_OK=1
    if [[ "$EXPLOITER_EDGE" != "null" ]]; then
      awk "BEGIN{exit !(${EXPLOITER_EDGE} <= ${INCUMBENT_EDGE} + ${EDGE_SLACK})}" || EDGE_OK=0
    fi
    if awk "BEGIN{exit !(${H2H_CANDIDATE} - ${H2H_INCUMBENT} >= ${H2H_MARGIN})}" && ((EDGE_OK)); then
      IS_NEW_BEST="true"
    fi
  fi
  if [[ "$IS_NEW_BEST" == "true" ]]; then
    cp "$CHAMP" "$RUN/champion_best_overall.bin"
    echo "$round" >"$RUN/champion_best_overall_round.txt"
    [[ "$EXPLOITER_EDGE" != "null" ]] && echo "$EXPLOITER_EDGE" >"$EDGE_FILE"
  fi

  printf '{"round":%d,"champion":"%s","pool_size":%d,"iters":%d,"init_source":"%s","field_win_share":%s,"fair_share":%s,"yardstick_win_share":%s,"yardstick_games":%d,"h2h_candidate":%s,"h2h_incumbent":%s,"h2h_games":%d,"exploiter_win_share_vs_champion":%s,"exploiter_edge_over_fair":%s,"belief_loss_first":%s,"belief_loss_last":%s,"is_new_best":%s}\n' \
    "$round" "$CHAMP" "$POOL_SIZE" "$ITERS_PER_ROUND" "$INIT_SOURCE" "$FIELD_WS" "$FAIR" "$YARDSTICK_WS" "$YARDSTICK_GAMES" "$H2H_CANDIDATE" "$H2H_INCUMBENT" "$H2H_GAMES" "$EXPLOITER_WS" "$EXPLOITER_EDGE" "$BELIEF_FIRST" "$BELIEF_LAST" "$IS_NEW_BEST" >>"$STATUS_JSONL"
  {
    echo "round $round  ($(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown))  pool_size=$POOL_SIZE iters=$ITERS_PER_ROUND init=$INIT_SOURCE"
    echo "  internal field win-share: ${FIELD_WS} (fair=${FAIR}, NOT comparable across rounds)"
    echo "  yardstick vs rollout-768: ${YARDSTICK_WS}  (n=${YARDSTICK_GAMES}, reference only)"
    echo "  h2h vs incumbent best:    candidate ${H2H_CANDIDATE} / incumbent ${H2H_INCUMBENT}  (gate: diff >= ${H2H_MARGIN}, n=${H2H_GAMES})$( [[ "$IS_NEW_BEST" == "true" ]] && echo '  *** NEW BEST ***' )"
    echo "  exploiter vs champion:    ${EXPLOITER_WS} (edge over fair: ${EXPLOITER_EDGE})"
    echo "  belief loss:              ${BELIEF_FIRST} -> ${BELIEF_LAST}"
  } | tee -a "$STATUS_LOG"

  # Fold this round's exploiter (not the champion itself — beating it already
  # implies robustness against something that beat an earlier champion) into
  # the pool for next round, then rebuild the weighted pool string.
  EXPLOITERS+=("$EXPLOITER")
  printf '{"round":%d,"checkpoint":"%s"}\n' "$round" "$EXPLOITER" >>"$POOL_STATE"
  POOL="$(rebuild_pool)"
  write_readme $((round + 1))
done

echo "league done. manifest: $MANIFEST, status: $STATUS_LOG"
