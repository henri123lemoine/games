"""Plays N games between two "Demon of Ignorance" (DoI) Java subprocesses at
potentially different `-l<level>` search depths, no net involved. Uses the
same `stratego_sim`-refereed lockstep discipline as `eval_vs_doi.py` (see
`doi_bridge.py` and calibration/README.md) so a rules-mismatch divergence
between DoI's own move and our sim's legality mask crashes loudly instead of
silently diverging.

    .venv/bin/python calibration/doi_vs_doi.py --level-a A --level-b B --games N [--seed S]

Prints one JSON line: {"level_a": A, "level_b": B, "wins_a": w, "draws": d, "wins_b": l}

Which physical seat (our sim's seat 0/1, i.e. DoI colour RED/BLUE) plays level
A vs level B alternates every game, so a coordinate- or colour-dependent bias
in the bridge can't systematically favour either level.
"""

import argparse
import json
import sys

import stratego_sim as sim

from doi_bridge import BaseReferee, DoIProcess, build_doi_abs_board, deploy_slot_needs_mirror, mirror_columns


class DoIVsDoIReferee(BaseReferee):
    """`self.doi[0]` is our sim's seat 0 (red), `self.doi[1]` is seat 1 (blue)."""

    def __init__(self, level_seat0: int, level_seat1: int, seed: int, move_cap: int, timeout: float):
        super().__init__(seed=seed, move_cap=move_cap)
        self.level = [level_seat0, level_seat1]
        self.doi = [DoIProcess(level=level_seat0, timeout=timeout), DoIProcess(level=level_seat1, timeout=timeout)]

    def close(self) -> None:
        for d in self.doi:
            d.kill()

    def run_setup(self) -> None:
        colours = ["RED", "BLUE"]
        lines = [self.doi[seat].setup(colours[seat]) for seat in (0, 1)]
        boards = [build_doi_abs_board(lines[seat], seat) for seat in (0, 1)]

        self.mirror_seat = [False, False]
        for seat in (0, 1):
            if deploy_slot_needs_mirror(boards[seat], seat):
                self.mirror_seat[seat] = True
                boards[seat] = mirror_columns(boards[seat], seat)
                if deploy_slot_needs_mirror(boards[seat], seat):
                    raise RuntimeError(
                        "DoI setup still violates our sim's forced flag handedness after "
                        "mirroring -- rules-mismatch divergence, aborting"
                    )
            self.board.update(boards[seat])

        placed = [0, 0]
        while placed[0] < 40 or placed[1] < 40:
            b = self.s.collect()
            assert b["deploy_obs"].shape[0] == 1, "lockstep divergence: expected exactly one live deploy row"
            seat = int(b["deploy_player"][0])
            cell = placed[seat] if seat == 0 else 99 - placed[seat]
            ptype = boards[seat][cell]
            legal = b["deploy_legal"][0]
            if not legal[ptype]:
                raise RuntimeError(
                    f"lockstep divergence: DoI seat {seat}'s own setup placed a type our "
                    f"sim's deploy legality forbids at cell {cell} (slot {placed[seat]}, type {ptype})"
                )
            import numpy as np

            logits = np.full((1, sim.DEPLOY_WIDTH), -1e9, np.float32)
            logits[0, ptype] = 30.0
            self.commit_deploy(logits)
            placed[seat] += 1

    def play_turn(self, mover: int) -> tuple[bool, float]:
        # `self.mirror` (used by `BaseReferee`'s coordinate helpers) is
        # per-mover here, unlike the single-DoI referee, so set it for the
        # duration of this ply's decode/encode calls.
        self.mirror = self.mirror_seat[mover]
        b = self.s.collect()
        assert b["move_obs"].shape[0] == 1 and int(b["move_player"][0]) == mover
        legal_mask = b["move_legal"][0]

        src, dst = self.decode_doi_move(self.doi[mover].query_move())
        action = sim.srcdst_to_action(src, dst, mover)
        self.assert_legal_or_die(action, legal_mask, f"DoI seat {mover} (level {self.level[mover]})", src, dst)

        # `encode_relay` (via `_to_doi_xy`) must run once per recipient in
        # *that recipient's own* mirror frame, since the two seats can have
        # independently mirrored setups.
        suffix, flag_captured = self._outcome_message(src, dst)
        for seat in (0, 1):
            self.mirror = self.mirror_seat[seat]
            self.doi[seat].send(self.encode_relay(src, dst, suffix))
        self.mirror = self.mirror_seat[mover]

        term, reward = self.force_move(action)
        return term or flag_captured, reward

    def play(self) -> str:
        """Returns "a"/"b"/"draw" where seat0 plays level_a's assignment."""
        self.run_setup()
        self.doi[0].send("START")
        turn = 0
        for _ in range(self.move_cap * 2 + 10):
            mover = turn % 2
            term, reward = self.play_turn(mover)
            if term:
                if reward > 0:
                    return "seat0"
                if reward < 0:
                    return "seat1"
                return "draw"
            turn += 1
        raise RuntimeError("game exceeded move cap without terminating -- treat as a hang, not a draw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level-a", type=int, required=True)
    ap.add_argument("--level-b", type=int, required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-cap", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    wins_a = draws = wins_b = 0
    for i in range(args.games):
        a_is_seat0 = i % 2 == 0
        level_seat0 = args.level_a if a_is_seat0 else args.level_b
        level_seat1 = args.level_b if a_is_seat0 else args.level_a
        ref = DoIVsDoIReferee(level_seat0, level_seat1, seed=args.seed + i,
                               move_cap=args.move_cap, timeout=args.timeout)
        try:
            outcome = ref.play()
        finally:
            ref.close()
        if outcome == "draw":
            winner = "draw"
        else:
            winner_seat = 0 if outcome == "seat0" else 1
            winner = "a" if (winner_seat == 0) == a_is_seat0 else "b"
        print(f"game {i}: seat0_level={level_seat0} seat1_level={level_seat1} -> {outcome} ({winner})",
              file=sys.stderr)
        wins_a += winner == "a"
        draws += winner == "draw"
        wins_b += winner == "b"

    print(json.dumps({"level_a": args.level_a, "level_b": args.level_b,
                       "wins_a": wins_a, "draws": draws, "wins_b": wins_b}))


if __name__ == "__main__":
    main()
