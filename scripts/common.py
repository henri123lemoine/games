from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import IO, Any, Literal

from loguru import logger


def find_or_build_bridge() -> str:
    """Return the path to the compiled `twentyone_bridge` binary.

    Respects the `TWENTYONE_BRIDGE_BIN` env var. If the binary is not found
    under the project `target/debug` directory, triggers a `cargo build` for
    the bridge binary.
    """
    override = os.environ.get("TWENTYONE_BRIDGE_BIN")
    if override:
        logger.debug("Using TWENTYONE_BRIDGE_BIN override: {}", override)
        return override

    root = Path(__file__).resolve().parents[1]
    bin_path = root / "target" / "debug" / "twentyone_bridge"
    if not bin_path.exists():
        logger.info("Building missing bridge binary at {}", bin_path)
        subprocess.run(["cargo", "build", "--bin", "twentyone_bridge"], cwd=root, check=True)
    return str(bin_path)


class Bridge:
    """Thin JSON line protocol wrapper around the Rust bridge binary.

    Usage:
        with Bridge() as br:
            br.send({"cmd": "new", "seed": 42})
            ...
    """

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = find_or_build_bridge()
        self._proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to open bridge stdio pipes")
        self._in: IO[str] = self._proc.stdin
        self._out: IO[str] = self._proc.stdout

    def send(self, obj: dict[str, Any]) -> Any:
        """Send a single JSON command and return the `data` of the response.

        Raises RuntimeError on protocol errors or if the bridge reports an error.
        """
        line = json.dumps(obj)
        self._in.write(line + "\n")
        self._in.flush()
        out = self._out.readline()
        if not out:
            raise RuntimeError("bridge closed")
        try:
            resp = json.loads(out)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from bridge: {}", out.strip())
            raise RuntimeError(f"invalid bridge response: {e}") from e
        if resp.get("status") == "err":
            raise RuntimeError(str(resp.get("error")))
        return resp["data"]

    def close(self) -> None:
        try:
            self.send({"cmd": "quit"})
        except Exception as e:  # noqa: BLE001 - best effort shutdown
            logger.debug("Bridge close encountered: {}", e)
        finally:
            self._proc.terminate()

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def show_game_start(bridge: Bridge, seed: int, player_names: tuple[str, str]) -> None:
    """Display game initialization information."""
    print(f"Seed: {seed}")
    hearts = bridge.send({"cmd": "hearts"})
    print("New Game.")
    print(f"Hearts: {player_names[0]}={hearts['p0']} {player_names[1]}={hearts['p1']}")


def show_round_start(round_num: int) -> None:
    """Display round start information."""
    print("")
    print(f"Round {round_num}!")
    print("")


def get_public_info(bridge: Bridge) -> tuple[list[int], list[int]]:
    """Get public card information."""
    pub = bridge.send({"cmd": "public_info"})
    return pub.get("p0_up", []), pub.get("p1_up", [])


def show_turn_info(
    bridge: Bridge, current_player: int, obs: dict[str, Any], player_names: tuple[str, str]
) -> tuple[list[int], list[int]]:
    """Display turn information and return public card info."""
    p0_up, p1_up = get_public_info(bridge)

    if current_player == 0:
        print(f"{player_names[0]}'s turn.")
        print(f"{player_names[0]}: hidden=[{obs['self_face_down']}], shown={p0_up}")
        print(f"{player_names[1]}: shown={p1_up}")
    else:
        print(f"{player_names[1]}'s turn.")
        print(f"{player_names[1]}: hidden=[{obs['self_face_down']}], shown={p1_up}")
        print(f"{player_names[0]}: shown={p0_up}")

    print(f"Cards remaining in the deck: {obs['deck_count']}")
    return p0_up, p1_up


def show_custom_turn_info(
    current_player: int,
    obs: dict[str, Any],
    p0_up: list[int],
    p1_up: list[int],
    turn_message: str,
    self_message: str,
    opp_message: str,
) -> None:
    """Display custom turn information with flexible messaging."""
    print(turn_message)
    print(self_message)
    print(opp_message)
    print(f"Cards remaining in the deck: {obs['deck_count']}")


def show_action_choice(player_name: str, action: Literal["draw", "stand"]) -> None:
    """Display player's action choice."""
    print(f"{player_name} chooses: {action}")


def show_draw_effect(
    bridge: Bridge,
    current_player: int,
    action: Literal["draw", "stand"],
    prev_cards: tuple[list[int], list[int]],
    player_names: tuple[str, str],
) -> None:
    """Show the effect of a draw action."""
    if action == "draw":
        pub2 = bridge.send({"cmd": "public_info"})
        p0_up2 = pub2.get("p0_up", [])
        p1_up2 = pub2.get("p1_up", [])

        if current_player == 0 and len(p0_up2) > len(prev_cards[0]):
            print(f"{player_names[0]} drew {p0_up2[-1]}")
        elif current_player == 1 and len(p1_up2) > len(prev_cards[1]):
            print(f"{player_names[1]} drew {p1_up2[-1]}")
    elif action == "stand":
        print(f"{player_names[current_player]} stood")


def show_round_result(resp: dict[str, Any], player_names: tuple[str, str]) -> None:
    """Display round end results including final hands and outcome."""
    reveal = resp.get("reveal", {})
    final_up = resp.get("final_up", {})
    p0_up = final_up.get("p0", [])
    p1_up = final_up.get("p1", [])
    p0_dn = reveal.get("p0_down")
    p1_dn = reveal.get("p1_down")

    p0_final = [*p0_up, p0_dn] if p0_dn is not None else p0_up
    p1_final = [*p1_up, p1_dn] if p1_dn is not None else p1_up
    p0_total = sum(p0_final)
    p1_total = sum(p1_final)

    print("")
    print("Round over. Final cards:")
    print(f"{player_names[0]}: {p0_final}, total={p0_total}")
    print(f"{player_names[1]}: {p1_final}, total={p1_total}")

    step = resp["step"]
    out = step["outcome"] or {"winner": None, "damage": 0}
    if out["winner"] is None:
        print("Outcome: Tie.")
    else:
        print(f"Outcome: {player_names[out['winner']]} wins.")


def show_hearts_status(bridge: Bridge, player_names: tuple[str, str]) -> dict[str, int]:
    """Display current hearts status and return hearts dict."""
    hearts = bridge.send({"cmd": "hearts"})
    print("")
    print("Hearts:")
    print(f"{player_names[0]}: {hearts['p0']}")
    print(f"{player_names[1]}: {hearts['p1']}")
    return hearts


def is_game_over(hearts: dict[str, int]) -> bool:
    """Check if the game is over."""
    if hearts["p0"] == 0 or hearts["p1"] == 0:
        print("")
        print("Game over.")
        return True
    return False


def run_turn_sequence(
    bridge: Bridge, action_func: callable, player_names: tuple[str, str], *action_args: Any
) -> dict[str, Any] | None:
    """Run a complete turn sequence and return response if round is over."""
    cur = int(bridge.send({"cmd": "current_player"})["current_player"])
    obs = bridge.send({"cmd": "observation", "player": cur})["observation"]

    prev_cards = show_turn_info(bridge, cur, obs, player_names)

    action = action_func(obs, cur, *action_args)
    show_action_choice(player_names[cur], action)

    resp = bridge.send({"cmd": "step", "action": action})
    step = resp["step"]

    if not step["round_over"]:
        show_draw_effect(bridge, cur, action, prev_cards, player_names)
        return None

    return resp
