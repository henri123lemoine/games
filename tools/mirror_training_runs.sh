#!/usr/bin/env bash
# Mirrors every training-run output directory under
# ~/Documents/Programming/PersonalProjects/games/ to ~/TrainingMirror/,
# outside any git repo and outside any .claude/worktrees/ session sandbox.
#
# Discovery is by directory name ("runs"), not a hardcoded project list, so
# new games/ml crates are covered automatically without editing this file.
#
# Runs fine straight from an interactive shell (cron-style loop or alongside
# a training session) — that shell's existing folder access suffices, and no
# extra permissions are involved. The optional launchd route below runs it
# every 30 minutes unattended; the repo plist is a template (launchd cannot
# expand $HOME), installed with:
#   sed -e "s|__REPO__|$(git rev-parse --show-toplevel)|" -e "s|__HOME__|$HOME|" \
#     tools/com.henrilemoine.trainingmirror.plist \
#     > ~/Library/LaunchAgents/com.henrilemoine.trainingmirror.plist
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.henrilemoine.trainingmirror.plist
# Note launchd's bash has no Documents-folder TCC grant of its own, so the
# unattended route only works if you choose to extend one — prefer the
# interactive invocation over widening any grant.
# Verify with: launchctl kickstart -k gui/$(id -u)/com.henrilemoine.trainingmirror
# then check ~/TrainingMirror/mirror.log (success) vs launchd.err.log (denied).
#
# rsync never runs with --delete: the mirror only accumulates, it never
# propagates a deletion made at the source (that's the whole point — a
# worktree or branch getting wiped must not wipe the backup too).

set -euo pipefail

GAMES_ROOT="${GAMES_ROOT:-$HOME/Documents/Programming/PersonalProjects/games}"
MIRROR_ROOT="${MIRROR_ROOT:-$HOME/TrainingMirror}"
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
