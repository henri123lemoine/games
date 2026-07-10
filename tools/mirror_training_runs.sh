#!/usr/bin/env bash
# Mirrors every training-run output directory under
# ~/Documents/Programming/PersonalProjects/games/ to ~/TrainingMirror/,
# outside any git repo and outside any .claude/worktrees/ session sandbox.
#
# Discovery is by directory name ("runs"), not a hardcoded project list, so
# new games/ml crates are covered automatically without editing this file.
#
# Invoked every 30 minutes by the com.henrilemoine.trainingmirror launchd
# agent (see tools/com.henrilemoine.trainingmirror.plist). Install with:
#   cp tools/com.henrilemoine.trainingmirror.plist ~/Library/LaunchAgents/
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.henrilemoine.trainingmirror.plist
#
# One-time manual step required (macOS TCC, cannot be scripted): launchd
# spawns /bin/bash with no Full Disk / Documents-folder grant of its own —
# unlike an interactive shell, it doesn't inherit Terminal's grant — so the
# very first run fails with "Operation not permitted" until you add it in
# System Settings -> Privacy & Security -> Full Disk Access -> + -> /bin/bash.
# Verify with: launchctl kickstart -k gui/$(id -u)/com.henrilemoine.trainingmirror
# then check ~/TrainingMirror/mirror.log (success) vs launchd.err.log (denied).
#
# rsync never runs with --delete: the mirror only accumulates, it never
# propagates a deletion made at the source (that's the whole point — a
# worktree or branch getting wiped must not wipe the backup too).

set -euo pipefail

GAMES_ROOT="$HOME/Documents/Programming/PersonalProjects/games"
MIRROR_ROOT="$HOME/TrainingMirror"
LOG="$MIRROR_ROOT/mirror.log"
LOCK_DIR="/tmp/training_mirror.lock.d"

mkdir -p "$MIRROR_ROOT"

# macOS has no flock(1); use a mkdir-based lock (mkdir is atomic) so an
# overrunning previous invocation can't stack with the next 30-minute tick.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$(date -Iseconds) skip: previous run still in progress" >>"$LOG"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

{
    echo "=== $(date -Iseconds) mirror run start ==="

    if [ ! -d "$GAMES_ROOT" ]; then
        echo "GAMES_ROOT missing ($GAMES_ROOT) — nothing to mirror, is ~/Documents access revoked?"
        exit 0
    fi

    find "$GAMES_ROOT" -maxdepth 8 -type d -name runs \
        -not -path '*/target/*' \
        -not -path '*/target-wasm/*' \
        -not -path '*/node_modules/*' \
        -not -path '*/.git/*' \
        | while IFS= read -r src; do
            rel="${src#"$HOME"/}"
            dest="$MIRROR_ROOT/$rel"
            mkdir -p "$dest"
            echo "-- $src -> $dest"
            rsync -a --partial "$src"/ "$dest"/ --stats 2>&1 \
                | grep -E '^(Number of files|Total file size|Total transferred file size)'
        done

    echo "mirror footprint: $(du -sh "$MIRROR_ROOT" 2>/dev/null | cut -f1)"
    echo "=== $(date -Iseconds) mirror run done ==="
} >>"$LOG" 2>&1
