#!/usr/bin/env python3
"""Training-run watchdog: reads a run dir's metrics.jsonl every interval and
prints one line per *state change* (problem appears / resolves), so it can
drive an alerting monitor without spamming.

Checks (each born from a real failure mode of these runs):
  dead       training process gone / metrics stale while run not stopped
  gauge-dead elo watcher process gone
  plateau    last 5 elo gauges within a 40-point band
  resign     resign rate >30%, or measured false-positive rate >20%
  vloss      value loss rising over the last 20 iterations
  draws      decisive fraction <30% (draw death)
  short      average game <40 plies (degenerate play)
  stale      fresh samples per batch <2% of buffer (replay dilution)

Usage: health.py <run-dir> [interval-secs]
"""

import json
import os
import subprocess
import sys
import time

RUN = sys.argv[1]
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
STATE_PATH = os.path.join(RUN, ".health_state.json")
METRICS = os.path.join(RUN, "metrics.jsonl")


def load_rows():
    rows = []
    try:
        with open(METRICS) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def proc_running(pattern):
    return (
        subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, check=False
        ).returncode
        == 0
    )


def check(rows):
    """Returns {key: human message} of currently-firing problems."""
    bad = {}
    its = [r for r in rows if "policy_loss" in r]
    elos = [r for r in rows if r.get("event") == "elo"]
    # The run is stopped if the last *lifecycle* row is a stop event —
    # gauges may append elo events after the run ends.
    lifecycle = [
        r for r in rows if "policy_loss" in r or r.get("event") in ("start", "stop")
    ]
    stopped = bool(lifecycle) and lifecycle[-1].get("event") == "stop"

    if not stopped:
        age = time.time() - os.path.getmtime(METRICS) if os.path.exists(METRICS) else 1e9
        if not proc_running("azt run") or age > 900:
            bad["dead"] = (
                f"training not running or metrics stale {age/60:.0f}m "
                "(crash? lid closed is fine, this fires only past 15m)"
            )
        if not proc_running("azt elo"):
            bad["gauge-dead"] = "elo watcher process is not running"

    if len(elos) >= 5:
        recent = [e["est"] for e in elos[-5:]]
        if max(recent) - min(recent) < 40:
            bad["plateau"] = (
                f"rating flat at ~{sum(recent)/5:.0f} over last 5 gauges "
                f"({(elos[-1]['time'] - elos[-5]['time'])/3600:.1f}h)"
            )

    recent_its = its[-10:]
    if len(recent_its) >= 5:
        games = sum(r.get("games", 0) for r in recent_its)
        would = sum(r.get("would_resign", 0) for r in recent_its)
        fps = sum(r.get("resign_fp", 0) for r in recent_its)
        if games > 0:
            resign_rate = sum(r.get("resigned", 0) for r in recent_its) / games
            # A high rate with clean control games is resignation working;
            # alert only when false positives corroborate a spiral, or at
            # degenerate rates before any control evidence exists.
            if would >= 5 and fps / would > 0.15:
                bad["resign"] = (
                    f"resignation false-positive rate {100*fps/would:.0f}% "
                    f"({fps}/{would} no-resign games salvaged) — tighten resign-q"
                )
            elif resign_rate > 0.60 and would < 5:
                bad["resign"] = (
                    f"resign rate {100*resign_rate:.0f}% with no control-game "
                    "evidence yet — watch for a spiral"
                )
            decisive = sum(r.get("decisive", 0) for r in recent_its) / games
            if decisive < 0.30:
                bad["draws"] = f"only {100*decisive:.0f}% decisive games (draw death)"
        plies = sum(r.get("avg_plies", 0) for r in recent_its) / len(recent_its)
        if plies < 40:
            bad["short"] = f"games averaging {plies:.0f} plies (degenerate play)"

    if len(its) >= 20:
        window = [r["value_loss"] for r in its[-20:]]
        rise = window[-1] - window[0]
        halves = sum(window[10:]) / 10 - sum(window[:10]) / 10
        if rise > 0.03 and halves > 0.02:
            bad["vloss"] = f"value loss rising ({window[0]:.3f} -> {window[-1]:.3f} over 20 iters)"

    if its:
        last = its[-1]
        starts = [r for r in rows if r.get("event") == "start"]
        spi = starts[-1].get("samples_per_iter", 16384) if starts else 16384
        if last.get("buffer", 0) > 0 and spi / last["buffer"] < 0.02:
            bad["stale"] = (
                f"fresh data {100*spi/last['buffer']:.1f}% of replay buffer "
                f"({last['buffer']}) — shrink --replay"
            )
    return bad, stopped


def main():
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    print(f"watchdog armed on {RUN} (every {INTERVAL}s)", flush=True)

    while True:
        rows = load_rows()
        bad, stopped = check(rows)
        for key, msg in bad.items():
            if state.get(key) != msg:
                print(f"WARN [{key}] {msg}", flush=True)
                state[key] = msg
        for key in [k for k in state if k not in bad and k != "stopped"]:
            if state.pop(key, None):
                print(f"ok [{key}] resolved", flush=True)
        if stopped and not state.get("stopped"):
            elos = [r for r in rows if r.get("event") == "elo"]
            final = f", last gauge {elos[-1]['est']:.0f}" if elos else ""
            reason = next((r.get("reason", "?") for r in reversed(rows) if r.get("event") == "stop"), "?")
            print(f"RUN ENDED ({reason}){final}", flush=True)
            state["stopped"] = True
        if not stopped:
            state.pop("stopped", None)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
