"""Shared machinery for driving one or two "Demon of Ignorance" (DoI) Java
subprocesses through `stratego_sim`'s lockstep referee protocol.

See calibration/README.md for the wire protocol this drives DoI over (its
`-t` testing hook) and the rules caveats that bound how much a win-rate from
any of these bridges should be trusted.

`eval_vs_doi.py` (our net vs one DoI subprocess) and `doi_vs_doi.py` (two DoI
subprocesses, no net) both build on the `DoIProcess` subprocess wrapper and
the coordinate/outcome-token helpers here; each supplies its own referee loop
because the two acting sides differ (net inference vs a second DoI query).
"""

import queue
import subprocess
import threading
from pathlib import Path

import stratego_sim as sim

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
    """One `java ... StrategoDriver -t -l<level>` subprocess speaking the
    manager wire protocol reconstructed from `strategoevaluator/manager/
    {ai_controller,game,controller,stratego}.cpp` (see calibration/README.md)."""

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


class BaseReferee:
    """Common lockstep-referee state: `stratego_sim.BatchSim` is authoritative
    for legality/termination on both seats; `self.board[cell] = piece_type`
    mirrors both sides' knowledge losslessly since the referee itself chose
    every placement and knows every move."""

    def __init__(self, seed: int, move_cap: int):
        self.move_cap = move_cap
        self.s = sim.BatchSim(num_envs=1, move_cap=move_cap, seed=seed)
        self.mirror = False
        # cell -> piece_type; colour (0 = red cells 0-39, 1 = blue cells 60-99)
        # is implied by cell range, so we only store type.
        self.board: dict[int, int] = {}

    def _to_doi_xy(self, cell: int) -> tuple[int, int]:
        x, y = cell % 10, cell // 10
        return (9 - x, y) if self.mirror else (x, y)

    def _from_doi_xy(self, x: int, y: int) -> int:
        if self.mirror:
            x = 9 - x
        return y * 10 + x

    def _outcome_message(self, src: int, dst: int) -> tuple[str, bool]:
        """Returns `(result_suffix, flag_captured)`; updates `self.board`."""
        f = self.board.pop(src)
        t = self.board.get(dst)
        if t is None:
            self.board[dst] = f
            return "OK", False
        if t == FLAG:
            self.board[dst] = f
            return "VICTORY_FLAG", True
        atk_tok, def_tok = TYPE_TO_DOI_CHAR[f], TYPE_TO_DOI_CHAR[t]
        if defender_wins(f, t):
            self.board[dst] = t
            return f"DIES {atk_tok} {def_tok}", False
        if f == t:
            del self.board[dst]
            return f"BOTHDIE {atk_tok} {def_tok}", False
        self.board[dst] = f
        return f"KILLS {atk_tok} {def_tok}", False

    def decode_doi_move(self, line: str) -> tuple[int, int]:
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
        return src, dst

    def encode_relay(self, src: int, dst: int, suffix: str) -> str:
        dx, dy = self._to_doi_xy(src)
        ddst_x, ddst_y = self._to_doi_xy(dst)
        if dx == ddst_x:
            d, mult = ("DOWN" if ddst_y > dy else "UP"), abs(ddst_y - dy)
        else:
            d, mult = ("RIGHT" if ddst_x > dx else "LEFT"), abs(ddst_x - dx)
        parts = [str(dx), str(dy), d]
        if mult > 1:
            parts.append(str(mult))
        return " ".join(parts) + " " + suffix

    def apply_and_relay(self, src: int, dst: int, doi_recipients: list[DoIProcess]) -> bool:
        """Resolves the battle in `self.board`, sends the resulting
        confirmation/relay line to every DoI recipient (`game.cpp`'s
        `red->Message(buffer); blue->Message(buffer)` — sent to **both** sides
        after every move, including back to the mover itself). Returns
        `flag_captured`."""
        suffix, flag_captured = self._outcome_message(src, dst)
        line = self.encode_relay(src, dst, suffix)
        for doi in doi_recipients:
            doi.send(line)
        return flag_captured

    def assert_legal_or_die(self, action, legal_mask, mover_name: str, src: int, dst: int) -> None:
        if action is None or not legal_mask[action]:
            raise RuntimeError(
                f"RULES-MISMATCH DIVERGENCE: mover {mover_name} played "
                f"src={src} dst={dst}, which our sim's move_legal mask forbids. This is "
                "exactly the kind of two-square/continuous-chase rule divergence flagged "
                "in calibration/README.md -- aborting rather than silently diverging."
            )

    def force_move(self, action: int):
        import numpy as np

        logits = np.full((1, sim.N_ACTION), -1e9, np.float32)
        logits[0, action] = 30.0
        vals = np.zeros(1, np.float32)
        probs = np.full((1, 3), 1.0 / 3.0, np.float32)
        empty_deploy = np.zeros((0, sim.DEPLOY_WIDTH), np.float32)
        out = self.s.commit(logits, vals, probs, empty_deploy, np.zeros(0, np.float32), np.zeros((0, 3), np.float32))
        return bool(out["terminal"][0]), float(out["reward_pl0"][0])

    def commit_deploy(self, logits) -> None:
        import numpy as np

        vals = np.zeros(1, np.float32)
        probs = np.full((1, 3), 1.0 / 3.0, np.float32)
        empty_move = np.zeros((0, sim.N_ACTION), np.float32)
        self.s.commit(empty_move, np.zeros(0, np.float32), np.zeros((0, 3), np.float32), logits, vals, probs)
