import ast
import json
import sys
from secrets import randbits
from typing import Literal

from common import (
    Bridge,
    get_public_info,
    is_game_over,
    show_action_choice,
    show_custom_turn_info,
    show_game_start,
    show_hearts_status,
    show_round_result,
    show_round_start,
)


def load_policy(path: str):
    with open(path) as f:
        raw = json.load(f)
    # keys are stringified tuples; parse back with ast.literal_eval
    pol = {ast.literal_eval(k): v for k, v in raw.items()}
    return pol


def infoset_from_obs(obs, player, deck_mask_est=0):
    # We do not know the exact deck mask from observation; use count proxy only.
    # For policy lookup, approximate by combining deck_count to a coarse mask bucket.
    # Map deck_count to an integer bucket 0..11.
    bucket = int(obs["deck_count"]) & 0xF
    return (
        player,
        obs["self_total"],
        obs["opp_face_up"],
        obs["self_stood"],
        obs["opp_stood"],
        bucket,
    )


def choose_action(policy, obs: dict, player: int) -> Literal["draw", "stand"]:
    info = infoset_from_obs(obs, player)
    strat = policy.get(info)
    if strat is None:
        # fallback heuristic
        return "draw" if obs["self_total"] < 17 else "stand"
    return "draw" if strat[0] >= strat[1] else "stand"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: play_vs_agent.py policy_mccfr.json [seed]")
        return 2
    policy = load_policy(sys.argv[1])
    # Default to OS-random seed when not provided for non-repro runs.
    seed = int(sys.argv[2], 0) if len(sys.argv) > 2 else randbits(64)
    with Bridge() as bridge:
        bridge.send({"cmd": "new", "seed": seed})
        p_user = 0  # user plays as player 0
        player_names = ("You", "Agent")
        show_game_start(bridge, seed, player_names)

        while True:
            bridge.send({"cmd": "start_round"})
            rnd = bridge.send({"cmd": "round"})["round"]
            show_round_start(rnd)
            while True:
                cur = bridge.send({"cmd": "current_player"})["current_player"]
                obs = bridge.send({"cmd": "observation", "player": cur})["observation"]
                p0_up, p1_up = get_public_info(bridge)

                if cur == p_user:
                    my_up = p0_up if p_user == 0 else p1_up
                    opp_up = p1_up if p_user == 0 else p0_up
                    show_custom_turn_info(
                        cur,
                        obs,
                        p0_up,
                        p1_up,
                        "Your turn.",
                        f"You: hidden=[{obs['self_face_down']}], shown={my_up}",
                        f"Opp: shown={opp_up}",
                    )
                    act: Literal["draw", "stand"] = (
                        "draw"
                        if input("Type d=draw, s=stand: ").strip().lower().startswith("d")
                        else "stand"
                    )
                else:
                    # Agent's pov: opp_face_up is YOUR face-up card
                    your_up = p0_up if p_user == 0 else p1_up
                    agent_up = p1_up if p_user == 0 else p0_up
                    prev_len = len(agent_up)
                    show_custom_turn_info(
                        cur,
                        obs,
                        p0_up,
                        p1_up,
                        "Agent's turn.",
                        f"Agent sees your shown={your_up}, its shown={agent_up}",
                        "",
                    )
                    act = choose_action(policy, obs, cur)
                    show_action_choice("Agent", act)

                resp = bridge.send({"cmd": "step", "action": act})
                step = resp["step"]
                if cur != p_user and not step["round_over"]:
                    # After agent acts, show the effect immediately for clarity
                    p0_up2, p1_up2 = get_public_info(bridge)
                    agent_up2 = p1_up2 if p_user == 0 else p0_up2
                    if act == "draw" and len(agent_up2) > prev_len:
                        print(f"Agent drew {agent_up2[-1]}")
                    elif act == "stand":
                        print("Agent stood")
                if step["round_over"]:
                    show_round_result(resp, player_names)
                    hearts = show_hearts_status(bridge, player_names)

                    if is_game_over(hearts):
                        return 0
                    break


if __name__ == "__main__":
    sys.exit(main())
