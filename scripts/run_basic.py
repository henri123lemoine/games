import sys
from typing import Literal

from common import Bridge


def bot_action(obs: dict) -> Literal["draw", "stand"]:
    return "draw" if int(obs["self_total"]) < 17 else "stand"


def main() -> int:
    seed = 0x1234_5678_9ABC_DEF0
    with Bridge() as bridge:
        bridge.send({"cmd": "new", "seed": seed})
        total_rounds = 0
        while True:
            bridge.send({"cmd": "start_round"})
            # play the round
            while True:
                cur = int(bridge.send({"cmd": "current_player"})["current_player"])  # type: ignore[index]
                obs = bridge.send({"cmd": "observation", "player": cur})["observation"]
                act = bot_action(obs)
                step = bridge.send({"cmd": "step", "action": act})["step"]
                if step["round_over"]:
                    total_rounds += 1
                    hearts = bridge.send({"cmd": "hearts"})
                    # quick assertions: hearts in [0,6], damage accounted by round
                    if hearts["p0"] == 0 or hearts["p1"] == 0:
                        print(f"Game over in {total_rounds} rounds. Hearts: {hearts}")
                        print(f"Last outcome: {step['outcome']}")
                        return 0
                    break


if __name__ == "__main__":
    sys.exit(main())
