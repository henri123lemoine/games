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
both seats. DoI is driven as a subprocess speaking its manager wire protocol.
Every ply, the acting side's move is decoded to absolute board coordinates,
checked against the sim's own `move_legal` mask *before* it is applied (the
lockstep agreement assertion — see `assert_legal_or_die`), forced into the sim
via a one-hot logit row so the sim's board stays byte-identical to DoI's, and
mirrored into a plain Python board (`Referee.board`) this script maintains
itself so it can compute the battle-outcome tokens DoI's protocol requires
without any extra engine introspection.
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

import stratego_nets as S
import stratego_sim as sim

from stratego_trainer.rundir import load_checkpoint

HERE = Path(__file__).resolve().parent
DOI_DIR = HERE / "vendor" / "stratego"
DOI_MAIN = "com.cjmalloy.stratego.player.StrategoDriver"

# Our `games/stratego` PieceType order (board.rs): Spy=0 .. Marshal=9, Flag=10,
# Bomb=11. DoI's manager-protocol rank tokens (`Piece::tokens` in
# strategoevaluator/manager/stratego.cpp), indexed the same way.
SPY, SCOUT, MINER, MARSHAL, FLAG, BOMB = 0, 1, 2, 9, 10, 11
TYPE_TO_DOI_CHAR = ["s", "9", "8", "7", "6", "5", "4", "3", "2", "1", "F", "B"]
DOI_CHAR_TO_TYPE = {c: i for i, c in enumerate(TYPE_TO_DOI_CHAR)}

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
OPPOSITE_DIR = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "UP", "DOWN": "DOWN"}


def defender_wins(f: int, t: int) -> bool:
    return (t < FLAG and t > f and not (t == MARSHAL and f == SPY)) or (t == BOMB and f != MINER)


class DoIProcess:
    """One `java ... StrategoDriver -t -l0` subprocess speaking the manager
    wire protocol reconstructed from `strategoevaluator/manager/{ai_controller,
    game,controller,stratego}.cpp` (see calibration/README.md)."""

    def __init__(self, level: int = 0, timeout: float = 30.0):
        self.timeout = timeout
        self.proc = subprocess.Popen(
            ["java", "-cp", "bin", DOI_MAIN, "-t", f"-l{level}"],
            cwd=str(DOI_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._q: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put("")  # EOF sentinel

    def _readline(self) -> str:
        try:
            line = self._q.get(timeout=self.timeout)
        except queue.Empty:
            self.kill()
            raise RuntimeError(f"DoI subprocess timed out after {self.timeout}s waiting for a line")
        if line == "":
            err = self.proc.stderr.read()
            raise RuntimeError(f"DoI subprocess exited unexpectedly. stderr:\n{err}")
        return line.rstrip("\n")

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            err = self.proc.stderr.read()
            raise RuntimeError(f"DoI subprocess pipe broke sending {line!r}. stderr:\n{err}")

    def setup(self, colour: str, opponent: str = "hero") -> list[str]:
        self.send(f"{colour} {opponent} 10 10")
        return [self._readline() for _ in range(4)]

    def query_move(self) -> str:
        try:
            self.proc.stdin.write((("." * 10 + "\n") * 10))
            self.proc.stdin.flush()
        except BrokenPipeError:
            err = self.proc.stderr.read()
            raise RuntimeError(f"DoI subprocess pipe broke sending the board. stderr:\n{err}")
        return self._readline()

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass


def build_doi_abs_board(setup_lines: list[str], doi_seat: int) -> dict[int, int]:
    """`Controller::Setup`'s placement rule: response row `y` (0..4), column
    `x` (0..10) -> absolute cell `(y_start + y) * 10 + x`, `y_start = 0` for
    seat 0 (red) or `6` for seat 1 (blue). No reflection either way — this is
    the manager-side rule, independent of anything DoI's own internal board
    orientation means to itself."""
    y_start = 0 if doi_seat == 0 else 6
    board: dict[int, int] = {}
    for y, line in enumerate(setup_lines):
        if len(line) != 10:
            raise RuntimeError(f"DoI setup row {y} wrong width: {line!r}")
        for x, ch in enumerate(line):
            if ch not in DOI_CHAR_TO_TYPE:
                raise RuntimeError(f"DoI setup row {y} has unrecognised rank token {ch!r}: {line!r}")
            board[(y_start + y) * 10 + x] = DOI_CHAR_TO_TYPE[ch]
    return board


def mirror_columns(board: dict[int, int], doi_seat: int) -> dict[int, int]:
    rows = range(0, 4) if doi_seat == 0 else range(6, 10)
    out = dict(board)
    for r in rows:
        for x in range(10):
            out[r * 10 + x] = board[r * 10 + (9 - x)]
    return out


def deploy_slot_needs_mirror(board: dict[int, int], doi_seat: int) -> bool:
    """`arrangement.rs::flag_allowed_on_right`: the deploy slot `j` (row-major
    within the player's own 4 home rows) must have `j % 10 >= 5` to legally
    hold the flag under our sim's hard-coded `force_handedness=True`. `j` maps
    to absolute cell `j` (seat 0) or `99 - j` (seat 1, point-reflected) — see
    `arrangement.rs::board_from_arrangements`."""
    flag_cell = next(c for c, t in board.items() if t == FLAG)
    j = flag_cell if doi_seat == 0 else 99 - flag_cell
    return j % 10 < 5


class Referee:
    """Drives one full game: our sim as the authoritative rules engine on both
    seats, DoI as a subprocess on `doi_seat`, our net directly on `hero_seat`.
    `self.board[cell] = piece_type` (colour is implied by which half of the
    board / by tracking separately) mirrors both sides' knowledge losslessly,
    since this referee itself chose every placement and knows every move."""

    def __init__(self, move_net, setup_net, doi_seat: int, seed: int, move_cap: int,
                 temperature: float, doi_level: int, timeout: float):
        self.doi_seat = doi_seat
        self.hero_seat = 1 - doi_seat
        self.move_net = move_net
        self.setup_net = setup_net
        self.temperature = temperature
        self.move_cap = move_cap
        self.s = sim.BatchSim(num_envs=1, move_cap=move_cap, seed=seed)
        self.doi = DoIProcess(level=doi_level, timeout=timeout)
        self.rng = np.random.default_rng(seed)
        self.mirror = False
        # cell -> (colour, piece_type); colour 0 = red (cells 0-39), 1 = blue
        # (cells 60-99) is implied by cell range, so we only store type.
        self.board: dict[int, int] = {}

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
        self._commit_deploy(logits)

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
        self._commit_deploy(logits)

    def _commit_deploy(self, logits: np.ndarray) -> None:
        vals = np.zeros(1, np.float32)
        probs = np.full((1, 3), 1.0 / 3.0, np.float32)
        empty_move = np.zeros((0, sim.N_ACTION), np.float32)
        self.s.commit(empty_move, np.zeros(0, np.float32), np.zeros((0, 3), np.float32),
                      logits, vals, probs)

    # ---- move phase --------------------------------------------------

    def _to_doi_xy(self, cell: int) -> tuple[int, int]:
        x, y = cell % 10, cell // 10
        return (9 - x, y) if self.mirror else (x, y)

    def _from_doi_xy(self, x: int, y: int) -> int:
        if self.mirror:
            x = 9 - x
        return y * 10 + x

    def _outcome_message(self, src: int, dst: int) -> tuple[str, str | None, bool]:
        """Returns `(result_suffix, note, flag_captured)`; updates `self.board`."""
        f = self.board.pop(src)
        t = self.board.get(dst)
        if t is None:
            self.board[dst] = f
            return "OK", None, False
        if t == FLAG:
            self.board[dst] = f
            return "VICTORY_FLAG", "flag", True
        atk_tok, def_tok = TYPE_TO_DOI_CHAR[f], TYPE_TO_DOI_CHAR[t]
        if defender_wins(f, t):
            self.board[dst] = t
            return f"DIES {atk_tok} {def_tok}", "defender_wins", False
        if f == t:
            del self.board[dst]
            return f"BOTHDIE {atk_tok} {def_tok}", "tie", False
        self.board[dst] = f
        return f"KILLS {atk_tok} {def_tok}", "attacker_wins", False

    def _apply_and_report(self, mover: int, src: int, dst: int) -> tuple[str, bool]:
        """Resolves the battle in `self.board`, forces the same action into
        our sim, sends the resulting confirmation/relay line to DoI. Returns
        `(action_debug_str, flag_captured)`."""
        suffix, _note, flag_captured = self._outcome_message(src, dst)

        dx, dy = self._to_doi_xy(src)
        ddst_x, ddst_y = self._to_doi_xy(dst)
        # Recover direction/multiplier in DoI's own (possibly mirrored) frame.
        if dx == ddst_x:
            d, mult = ("DOWN" if ddst_y > dy else "UP"), abs(ddst_y - dy)
        else:
            d, mult = ("RIGHT" if ddst_x > dx else "LEFT"), abs(ddst_x - dx)
        parts = [str(dx), str(dy), d]
        if mult > 1:
            parts.append(str(mult))
        buffer = " ".join(parts) + " " + suffix
        self.doi.send(buffer)
        return buffer, flag_captured

    def assert_legal_or_die(self, action, legal_mask, mover: int, src: int, dst: int) -> None:
        if action is None or not legal_mask[action]:
            raise RuntimeError(
                "RULES-MISMATCH DIVERGENCE: mover "
                f"{'DoI' if mover == self.doi_seat else 'hero'} (seat {mover}) played "
                f"src={src} dst={dst}, which our sim's move_legal mask forbids. This is "
                "exactly the kind of two-square/continuous-chase rule divergence flagged "
                "in calibration/README.md -- aborting rather than silently diverging."
            )

    def play_doi_turn(self) -> tuple[bool, float]:
        b = self.s.collect()
        assert b["move_obs"].shape[0] == 1 and int(b["move_player"][0]) == self.doi_seat
        legal_mask = b["move_legal"][0]

        line = self.doi.query_move()
        toks = line.split()
        if toks and toks[0] in ("QUIT", "SURRENDER", "NO_MOVE"):
            raise RuntimeError(f"DoI ended the game unexpectedly: {line!r}")
        x, y, d = int(toks[0]), int(toks[1]), toks[2]
        mult = int(toks[3]) if len(toks) > 3 and toks[3].lstrip("-").isdigit() else 1
        src = self._from_doi_xy(x, y)
        real_dir = OPPOSITE_DIR[d] if (self.mirror and d in ("LEFT", "RIGHT")) else d
        drow, dcol = DIRS[real_dir]
        srow, scol = src // 10, src % 10
        dst = (srow + drow * mult) * 10 + (scol + dcol * mult)

        action = sim.srcdst_to_action(src, dst, self.doi_seat)
        self.assert_legal_or_die(action, legal_mask, self.doi_seat, src, dst)

        _buf, flag_captured = self._apply_and_report(self.doi_seat, src, dst)
        term, reward = self._force_move(action)
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
        # `_apply_and_report` sends the relay line to DoI itself (game.cpp's
        # `Message()` goes to both controllers regardless of who moved).
        _buf, flag_captured = self._apply_and_report(self.hero_seat, src, dst)
        term, reward = self._force_move(action)
        return term or flag_captured, reward

    def _force_move(self, action: int) -> tuple[bool, float]:
        logits = np.full((1, sim.N_ACTION), -1e9, np.float32)
        logits[0, action] = 30.0
        vals = np.zeros(1, np.float32)
        probs = np.full((1, 3), 1.0 / 3.0, np.float32)
        empty_deploy = np.zeros((0, sim.DEPLOY_WIDTH), np.float32)
        out = self.s.commit(logits, vals, probs, empty_deploy, np.zeros(0, np.float32), np.zeros((0, 3), np.float32))
        return bool(out["terminal"][0]), float(out["reward_pl0"][0])

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
