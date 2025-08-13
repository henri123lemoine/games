import ast
import json
import sys
from typing import Literal

from common import Bridge


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
    seed = int(sys.argv[2], 0) if len(sys.argv) > 2 else 42
    with Bridge() as bridge:
        bridge.send({"cmd": "new", "seed": seed})
        p_user = 0  # user plays as player 0

        # Intro: New Game + hearts
        hearts = bridge.send({"cmd": "hearts"})
        print("New Game.")
        print(f"Hearts: You={hearts['p0']} Agent={hearts['p1']}")

        while True:
            bridge.send({"cmd": "start_round"})
            rnd = bridge.send({"cmd": "round"})["round"]
            print("")
            print(f"Round {rnd}!")
            print("")
            while True:
                cur = bridge.send({"cmd": "current_player"})["current_player"]
                obs = bridge.send({"cmd": "observation", "player": cur})["observation"]
                pub = bridge.send({"cmd": "public_info"})
                p0_up = pub.get("p0_up", [])
                p1_up = pub.get("p1_up", [])

                if cur == p_user:
                    my_up = p0_up if p_user == 0 else p1_up
                    opp_up = p1_up if p_user == 0 else p0_up
                    print("Your turn.")
                    print(f"You: hidden=[{obs['self_face_down']}], shown={my_up}")
                    print(f"Opp: shown={opp_up}")
                    print(f"Cards remaining in the deck: {obs['deck_count']}")
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
                    print("Agent's turn.")
                    print(f"Agent sees your shown={your_up}, its shown={agent_up}")
                    print(f"Cards remaining in the deck: {obs['deck_count']}")
                    act = choose_action(policy, obs, cur)
                    print(f"Agent chooses: {act}")

                resp = bridge.send({"cmd": "step", "action": act})
                step = resp["step"]
                if cur != p_user and not step["round_over"]:
                    # After agent acts, show the effect immediately for clarity
                    pub2 = bridge.send({"cmd": "public_info"})
                    p0_up2 = pub2.get("p0_up", [])
                    p1_up2 = pub2.get("p1_up", [])
                    agent_up2 = p1_up2 if p_user == 0 else p0_up2
                    if act == "draw" and len(agent_up2) > prev_len:
                        print(f"Agent drew {agent_up2[-1]}")
                    elif act == "stand":
                        print("Agent stood")
                if step["round_over"]:
                    # Reveal down cards and show final hands
                    reveal = resp.get("reveal", {})
                    final_up = resp.get("final_up", {})
                    p0_up = final_up.get("p0", [])
                    p1_up = final_up.get("p1", [])
                    p0_dn = reveal.get("p0_down")
                    p1_dn = reveal.get("p1_down")
                    you_up = p0_up if p_user == 0 else p1_up
                    opp_up = p1_up if p_user == 0 else p0_up
                    you_dn = p0_dn if p_user == 0 else p1_dn
                    opp_dn = p1_dn if p_user == 0 else p0_dn
                    you_final = [*you_up, you_dn] if you_dn is not None else you_up
                    opp_final = [*opp_up, opp_dn] if opp_dn is not None else opp_up
                    you_total = sum(you_final)
                    opp_total = sum(opp_final)

                    print("")
                    print("Round over. Final cards:")
                    print(f"You: {you_final}, total={you_total}")
                    print(f"Opp: {opp_final}, total={opp_total}")
                    out = step["outcome"] or {"winner": None, "damage": 0}
                    if out["winner"] is None:
                        print("Outcome: Tie.")
                    else:
                        winner = "You" if out["winner"] == p_user else "Opp"
                        print(f"Outcome: {winner} wins.")

                    hearts = bridge.send({"cmd": "hearts"})
                    print("")
                    print("Hearts:")
                    print(f"You: {hearts['p0' if p_user == 0 else 'p1']}")
                    print(f"Opp: {hearts['p1' if p_user == 0 else 'p0']}")

                    if hearts["p0"] == 0 or hearts["p1"] == 0:
                        print("")
                        print("Game over.")
                        return 0
                    break


if __name__ == "__main__":
    sys.exit(main())
