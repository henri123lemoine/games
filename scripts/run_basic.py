import sys
from secrets import randbits
from typing import Any, Literal

from common import (
    Bridge,
    is_game_over,
    run_turn_sequence,
    show_game_start,
    show_hearts_status,
    show_round_result,
    show_round_start,
)


def bot_action(obs: dict[str, Any], player: int) -> Literal["draw", "stand"]:
    return "draw" if int(obs["self_total"]) < 17 else "stand"


def main() -> int:
    seed = randbits(64)
    player_names = ("Bot0", "Bot1")

    with Bridge() as bridge:
        bridge.send({"cmd": "new", "seed": seed})
        show_game_start(bridge, seed, player_names)

        while True:
            bridge.send({"cmd": "start_round"})
            rnd = bridge.send({"cmd": "round"})["round"]
            show_round_start(rnd)

            # play the round
            while True:
                resp = run_turn_sequence(bridge, bot_action, player_names)

                if resp is not None:  # round is over
                    show_round_result(resp, player_names)
                    hearts = show_hearts_status(bridge, player_names)

                    if is_game_over(hearts):
                        return 0
                    break


if __name__ == "__main__":
    sys.exit(main())
