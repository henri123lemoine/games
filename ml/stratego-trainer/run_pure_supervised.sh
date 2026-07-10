#!/bin/bash
# Pure tabula-rasa marathon supervisor: random init, magnet-only self-play,
# NO BC warm start and NO anchor (the 2026-07-06 marathon lineage trained
# from a HeuristicBot-BC-anchored ancestor, which the project owner rejects
# as impure; its weights were lost in the 2026-07-10 worktree deletion).
#
# Relaunch-safe, fixing the old supervisor's footgun: on every (re)launch it
# scans runs/pure_r*/ for the newest latest/best checkpoint and resumes from
# it, so a fresh invocation after a crash or reboot NEVER restarts training
# from scratch by accident. First-ever launch (no checkpoints) starts from
# random init — the whole point.
cd "$(dirname "$0")"
DEADLINE_FILE=runs/pure_deadline.txt
if [ ! -f "$DEADLINE_FILE" ]; then
  mkdir -p runs
  echo $(( $(date +%s) + 604800 )) >"$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
N=$(ls -d runs/pure_r* 2>/dev/null | sed 's/.*pure_r//' | sort -n | tail -1)
N=${N:-0}
while [ $(date +%s) -lt $DEADLINE ]; do
  N=$((N+1))
  NAME=pure_r${N}
  RESUME_ARGS=()
  NEWEST=$(ls -t runs/pure_r*/latest.safetensors runs/pure_r*/best.safetensors 2>/dev/null | head -1)
  if [ -n "$NEWEST" ]; then RESUME_ARGS=(--resume "$NEWEST"); fi
  echo "[supervisor] $(date -u +%FT%TZ) launching $NAME resume=${NEWEST:-tabula-rasa}"
  .venv/bin/python -m stratego_trainer.train --run-name "$NAME" --envs 1600 --net-size ref \
    "${RESUME_ARGS[@]}" \
    --anchor-coef 0.0 --anchor-floor 0.0 \
    --work-seconds $(( DEADLINE - $(date +%s) )) --iters 10000000 >> runs/${NAME}.nohup.log 2>&1
  echo "[supervisor] $(date -u +%FT%TZ) $NAME exited"
  sleep 20
done
echo "[supervisor] deadline reached, done"
