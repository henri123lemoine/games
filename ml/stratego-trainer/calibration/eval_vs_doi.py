"""Plays our checkpoint against the vendored "Demon of Ignorance" (DoI) Java
engine (github.com/braathwaate/stratego), alternating colours across games.

    .venv/bin/python calibration/eval_vs_doi.py --ckpt <path.safetensors> --games N [--seed S]

Prints one JSON line: {"ws_doi": <win share>, "games": N, "wins": w, "draws": d, "losses": l}

See calibration/README.md for the wire protocol this drives DoI over (its `-t`
testing hook, which implements the interface documented by
github.com/braathwaate/strategoevaluator) and — important — the rules
caveats that bound how much a win-rate from this bridge should be trusted.

Architecture (this is a from-scratch thin Python referee, not a reuse of the
strategoevaluator manager binary): a `stratego_sim.BatchSim(num_envs=1, ...)`
instance is the authoritative rules engine for legality and termination on
both seats. DoI is driven as a subprocess speaking its manager wire protocol
via `doi_bridge.DoIProcess`. Every ply, the acting side's move is decoded to
absolute board coordinates, checked against the sim's own `move_legal` mask
*before* it is applied (the lockstep agreement assertion), forced into the
sim via a one-hot logit row so the sim's board stays byte-identical to DoI's,
and mirrored into `doi_bridge.BaseReferee.board`, this referee's own plain
Python board, so it can compute the battle-outcome tokens DoI's protocol
requires without any extra engine introspection.
"""

import argparse
import json
import sys

import numpy as np

import mlx.core as mx

import stratego_nets as S
import stratego_sim as sim

from stratego_trainer.rundir import load_checkpoint

from doi_bridge import BaseReferee, DoIProcess, build_doi_abs_board, deploy_slot_needs_mirror, mirror_columns


class Referee(BaseReferee):
    """Drives one full game: our sim as the authoritative rules engine on both
    seats, DoI as a subprocess on `doi_seat`, our net directly on `hero_seat`."""

    def __init__(self, move_net, setup_net, doi_seat: int, seed: int, move_cap: int,
                 temperature: float, doi_level: int, timeout: float):
        super().__init__(seed=seed, move_cap=move_cap)
        self.doi_seat = doi_seat
        self.hero_seat = 1 - doi_seat
        self.move_net = move_net
        self.setup_net = setup_net
        self.temperature = temperature
        self.doi = DoIProcess(level=doi_level, timeout=timeout)
        self.rng = np.random.default_rng(seed)

    def close(self) -> None:
        self.doi.kill()

    # ---- setup -----------------------------------------------------------

    def run_setup(self) -> None:
        colour = "RED" if self.doi_seat == 0 else "BLUE"
        lines = self.doi.setup(colour)
        doi_board = build_doi_abs_board(lines, self.doi_seat)
        if deploy_slot_needs_mirror(doi_board, self.doi_seat):
            self.mirror = True
            doi_board = mirror_columns(doi_board, self.doi_seat)
            if deploy_slot_needs_mirror(doi_board, self.doi_seat):
                raise RuntimeError(
                    "DoI setup still violates our sim's forced flag handedness after "
                    "mirroring -- rules-mismatch divergence, aborting"
                )
        self.board.update(doi_board)

        placed = [0, 0]
        while placed[0] < 40 or placed[1] < 40:
            b = self.s.collect()
            assert b["deploy_obs"].shape[0] == 1, "lockstep divergence: expected exactly one live deploy row"
            seat = int(b["deploy_player"][0])
            if seat == self.doi_seat:
                self._deploy_doi_step(doi_board, placed[seat], b["deploy_legal"][0])
            else:
                self._deploy_hero_step(b["deploy_obs"], placed[seat], b["deploy_legal"][0])
            placed[seat] += 1

    def _deploy_doi_step(self, doi_board: dict[int, int], placed: int, legal: np.ndarray) -> None:
        seat = self.doi_seat
        cell = placed if seat == 0 else 99 - placed
        ptype = doi_board[cell]
        if not legal[ptype]:
            raise RuntimeError(
                f"lockstep divergence: DoI's own setup placed a type our sim's deploy "
                f"legality forbids at cell {cell} (slot {placed}, type {ptype})"
            )
        logits = np.full((1, sim.DEPLOY_WIDTH), -1e9, np.float32)
        logits[0, ptype] = 30.0
        self.commit_deploy(logits)

    def _deploy_hero_step(self, obs: np.ndarray, placed: int, legal: np.ndarray) -> None:
        seat = self.hero_seat
        out = self.setup_net(mx.array(obs), mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32))
        n_placed = int(obs.reshape(40, 14).sum())
        slot_logits = np.clip(np.array(out["logits"].astype(mx.float32))[0, min(n_placed, 39)], -1e4, 1e4)
        slot_logits = np.where(legal, slot_logits / max(self.temperature, 1e-6), -1e30)
        slot_logits = np.minimum(slot_logits, 60.0)
        probs = np.exp(slot_logits - slot_logits.max())
        probs /= probs.sum()
        choice = int(self.rng.choice(len(probs), p=probs))
        cell = placed if seat == 0 else 99 - placed
        self.board[cell] = choice
        logits = np.full((1, sim.DEPLOY_WIDTH), -1e9, np.float32)
        logits[0, choice] = 30.0
        self.commit_deploy(logits)

    # ---- move phase --------------------------------------------------

    def play_doi_turn(self) -> tuple[bool, float]:
        b = self.s.collect()
        assert b["move_obs"].shape[0] == 1 and int(b["move_player"][0]) == self.doi_seat
        legal_mask = b["move_legal"][0]

        src, dst = self.decode_doi_move(self.doi.query_move())
        action = sim.srcdst_to_action(src, dst, self.doi_seat)
        self.assert_legal_or_die(action, legal_mask, "DoI", src, dst)

        flag_captured = self.apply_and_relay(src, dst, [self.doi])
        term, reward = self.force_move(action)
        return term or flag_captured, reward

    def play_hero_turn(self) -> tuple[bool, float]:
        b = self.s.collect()
        assert b["move_obs"].shape[0] == 1 and int(b["move_player"][0]) == self.hero_seat
        legal_mask = b["move_legal"][0]
        obs = b["move_obs"]
        out = self.move_net(mx.array(obs), legal_mask=mx.array(legal_mask[None]))
        logits = np.clip(np.array(out["move_logits"].astype(mx.float32))[0], -1e4, 1e4)
        logits = np.where(legal_mask, logits / max(self.temperature, 1e-6), -1e30)
        logits = np.minimum(logits, 60.0)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        action = int(self.rng.choice(len(probs), p=probs))

        src, dst = sim.action_to_srcdst(action, self.hero_seat)
        flag_captured = self.apply_and_relay(src, dst, [self.doi])
        term, reward = self.force_move(action)
        return term or flag_captured, reward

    def play(self) -> str:
        """Returns "win"/"draw"/"loss" from the hero's point of view."""
        self.run_setup()
        if self.doi_seat == 0:
            self.doi.send("START")
        turn = 0
        for _ in range(self.move_cap * 2 + 10):
            mover = turn % 2
            if mover == self.doi_seat:
                term, reward = self.play_doi_turn()
            else:
                term, reward = self.play_hero_turn()
            if term:
                hero_reward = reward if self.hero_seat == 0 else -reward
                if hero_reward > 0:
                    return "win"
                if hero_reward < 0:
                    return "loss"
                return "draw"
            turn += 1
        raise RuntimeError("game exceeded move cap without terminating -- treat as a hang, not a draw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-cap", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--doi-level", type=int, default=0, help="DoI's -l search-depth setting (0 = fastest)")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-line read timeout on the DoI subprocess")
    args = ap.parse_args()

    _flat, meta = mx.load(args.ckpt, return_metadata=True)
    move_cfg, setup_cfg = S.NET_SIZES[meta.get("net_size", "default")]
    move_net = S.MoveTransformer.from_config(move_cfg)
    setup_net = S.ArrangementTransformer.from_config(setup_cfg)
    load_checkpoint(args.ckpt, move=move_net, setup=setup_net, prefer_ema=True)

    wins = draws = losses = 0
    for i in range(args.games):
        doi_seat = i % 2
        ref = Referee(move_net, setup_net, doi_seat=doi_seat, seed=args.seed + i,
                      move_cap=args.move_cap, temperature=args.temperature,
                      doi_level=args.doi_level, timeout=args.timeout)
        try:
            result = ref.play()
        finally:
            ref.close()
        print(f"game {i}: hero_seat={ref.hero_seat} doi_seat={doi_seat} mirror={ref.mirror} -> {result}",
              file=sys.stderr)
        wins += result == "win"
        draws += result == "draw"
        losses += result == "loss"

    total = wins + draws + losses
    ws = (wins + 0.5 * draws) / total if total else 0.5
    print(json.dumps({"ws_doi": ws, "games": total, "wins": wins, "draws": draws, "losses": losses}))


if __name__ == "__main__":
    main()
