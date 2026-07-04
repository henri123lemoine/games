"""Human-vs-checkpoint Stratego, played in the terminal.

    python -m stratego_trainer.play_cli --ckpt runs/<run>/best.safetensors [--seat 0] [--ema]

A single-env `BatchSim` drives real game rules and legality (deployment order,
two-square/chase, combat) -- this file only renders the board (via the Rust
`render`, which never leaks a hidden opponent piece -- see
`render_never_leaks_a_hidden_opponent_rank`) and translates the human's typed
input into the same action space the net's logits live in. The net's turns
reuse `eval.py`'s exact net-forward -> logits convention; the human's turn
reuses the "spike one legal slot's logit, let the sim's softmax-sample
collapse onto it" trick already proven by `_heuristic_move_logits`.
"""

import argparse
import sys

import numpy as np

import mlx.core as mx

import stratego_nets as S
import stratego_sim as sim

from .eval import _net_logits
from .rundir import load_checkpoint

_SPIKE_LOGIT = 30.0
_PIECE_LABEL = {
    0: "Spy(1)", 1: "Scout(2)", 2: "Miner(3)", 3: "Sgt(4)", 4: "Lt(5)",
    5: "Cpt(6)", 6: "Maj(7)", 7: "Col(8)", 8: "Gen(9)", 9: "Marshal(10)",
    10: "Flag", 11: "Bomb",
}


def _load_net(ckpt: str, prefer_ema: bool):
    _, meta = mx.load(ckpt, return_metadata=True)
    move_cfg, setup_cfg = S.NET_SIZES[meta.get("net_size", "default")]
    move = S.MoveTransformer.from_config(move_cfg)
    setup = S.ArrangementTransformer.from_config(setup_cfg)
    load_checkpoint(ckpt, move=move, setup=setup, prefer_ema=prefer_ema)
    return move, setup


def _prompt_cell(label: str) -> int:
    """A cell as `row col` (0-9 each, matching the board's printed headers) or
    the fused 2-digit form the engine itself uses (`10*row + col`, e.g. `35`)."""
    while True:
        raw = input(f"  {label}: ").strip()
        parts = raw.replace(",", " ").split()
        try:
            if len(parts) == 2:
                row, col = int(parts[0]), int(parts[1])
            elif len(parts) == 1 and len(parts[0]) in (1, 2):
                v = int(parts[0])
                row, col = v // 10, v % 10
            else:
                raise ValueError
            if not (0 <= row <= 9 and 0 <= col <= 9):
                raise ValueError
            return 10 * row + col
        except ValueError:
            print("    enter as 'row col' (each 0-9), e.g. '3 5' or '35'")


def _human_move_logits(legal_row, player):
    while True:
        src = _prompt_cell("from")
        dst = _prompt_cell("to")
        action = sim.srcdst_to_action(src, dst, player)
        if action is None:
            print("  not a straight orthogonal slide -- try again")
            continue
        if not legal_row[action]:
            print("  illegal move (blocked, out of range, or not your piece) -- try again")
            continue
        logits = np.zeros(sim.N_ACTION, np.float32)
        logits[action] = _SPIKE_LOGIT
        return logits


def _human_deploy_logits(legal_row):
    choices = [t for t in range(len(_PIECE_LABEL)) if legal_row[t]]
    print("  available pieces: " + ", ".join(f"{t}={_PIECE_LABEL[t]}" for t in choices))
    while True:
        raw = input("  place which piece type (index): ").strip()
        try:
            t = int(raw)
            if t not in choices:
                raise ValueError
        except ValueError:
            print("  pick one of: " + ", ".join(str(t) for t in choices))
            continue
        logits = np.zeros(len(legal_row), np.float32)
        logits[t] = _SPIKE_LOGIT
        return logits


def play(ckpt: str, human_seat: int, prefer_ema: bool, temperature: float, move_cap: int, seed: int):
    move_net, setup_net = _load_net(ckpt, prefer_ema)
    s = sim.BatchSim(num_envs=1, move_cap=move_cap, seed=seed)
    print(f"Loaded {ckpt} ({'EMA' if prefer_ema else 'working'} net). "
          f"You are player {human_seat} ({'red' if human_seat == 0 else 'blue'}).")

    while True:
        b = s.collect()
        n_move, n_deploy = b["move_obs"].shape[0], b["deploy_obs"].shape[0]
        assert n_move + n_deploy == 1, "num_envs=1 has exactly one pending decision"

        if n_move:
            player = int(b["move_player"][0])
            print()
            print(s.render(0, human_seat))
            if player == human_seat:
                m_logits = _human_move_logits(b["move_legal"][0], player)[None, :]
                m_vals, m_probs = np.zeros(1, np.float32), np.full((1, 3), 1 / 3, np.float32)
            else:
                m_logits, m_vals, m_probs = _net_logits(
                    move_net, setup_net, b["move_obs"], b["move_legal"], "move", temperature=temperature)
            d_logits = np.zeros((0, sim.DEPLOY_WIDTH), np.float32)
            d_vals, d_probs = np.zeros(0, np.float32), np.zeros((0, 3), np.float32)
        else:
            player = int(b["deploy_player"][0])
            print()
            print(s.render(0, human_seat))
            if player == human_seat:
                d_logits = _human_deploy_logits(b["deploy_legal"][0])[None, :]
                d_vals, d_probs = np.zeros(1, np.float32), np.full((1, 3), 1 / 3, np.float32)
            else:
                d_logits, d_vals, d_probs = _net_logits(
                    move_net, setup_net, b["deploy_obs"], b["deploy_legal"], "deploy", temperature=temperature)
            m_logits = np.zeros((0, sim.N_ACTION), np.float32)
            m_vals, m_probs = np.zeros(0, np.float32), np.zeros((0, 3), np.float32)

        out = s.commit(m_logits, m_vals, m_probs, d_logits, d_vals, d_probs)
        if bool(out["terminal"][0]):
            r = float(out["reward_pl0"][0])
            capped = bool(out["capped"][0])
            print()
            print(s.render(0, human_seat))
            if capped:
                print(f"Game over: timed out (ply cap) -- draw.")
            elif r == 0.0:
                print(f"Game over: draw.")
            else:
                winner = 0 if r > 0 else 1
                print(f"Game over: player {winner} wins "
                      f"({'you' if winner == human_seat else 'the net'}).")
            again = input("Play again? [y/N]: ").strip().lower()
            if again != "y":
                return
            print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Play Stratego against a trained checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seat", type=int, choices=(0, 1), default=0, help="0=red (first), 1=blue")
    ap.add_argument("--ema", action="store_true", help="play against the EMA net instead of working")
    ap.add_argument("--temperature", type=float, default=0.25, help="net sampling temperature")
    ap.add_argument("--move-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        play(args.ckpt, args.seat, args.ema, args.temperature, args.move_cap, args.seed)
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
