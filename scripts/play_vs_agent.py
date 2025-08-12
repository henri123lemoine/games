import ast
import json
import subprocess
import sys


class Bridge:
    def __init__(self, path="target/debug/twentyone_bridge"):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def send(self, obj):
        line = json.dumps(obj)
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline()
        if not out:
            raise RuntimeError("bridge closed")
        resp = json.loads(out)
        if resp.get("status") == "err":
            raise RuntimeError(resp.get("error"))
        return resp["data"]

    def close(self):
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        self.p.terminate()


def load_policy(path):
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


def choose_action(policy, obs, player):
    info = infoset_from_obs(obs, player)
    strat = policy.get(info)
    if strat is None:
        # fallback heuristic
        return "draw" if obs["self_total"] < 17 else "stand"
    return "draw" if strat[0] >= strat[1] else "stand"


def main():
    if len(sys.argv) < 2:
        print("Usage: play_vs_agent.py data/policy_mccfr.json [seed]")
        return 2
    policy = load_policy(sys.argv[1])
    seed = int(sys.argv[2], 0) if len(sys.argv) > 2 else 42
    bridge = Bridge()
    try:
        bridge.send({"cmd": "new", "seed": seed})
        p_user = 0  # user plays as player 0
        while True:
            bridge.send({"cmd": "start_round"})
            print("New round!")
            while True:
                cur = bridge.send({"cmd": "current_player"})["current_player"]
                obs = bridge.send({"cmd": "observation", "player": cur})["observation"]
                if cur == p_user:
                    print(
                        f"Your turn. You: total={obs['self_total']}, up={obs['self_face_up']}, down=?, stood={obs['self_stood']}. Opp up={obs['opp_face_up']} stood={obs['opp_stood']}. Deck={obs['deck_count']}"
                    )
                    act = input("Type d=draw, s=stand: ").strip().lower()
                    act = "draw" if act.startswith("d") else "stand"
                else:
                    act = choose_action(policy, obs, cur)
                    print(f"Agent chooses: {act}")
                step = bridge.send({"cmd": "step", "action": act})["step"]
                if step["round_over"]:
                    print(f"Round over. Outcome: {step['outcome']}")
                    hearts = bridge.send({"cmd": "hearts"})
                    print(f"Hearts: You={hearts['p0']} Agent={hearts['p1']}")
                    if hearts["p0"] == 0 or hearts["p1"] == 0:
                        print("Game over.")
                        return 0
                    break
    finally:
        bridge.close()


if __name__ == "__main__":
    sys.exit(main())
