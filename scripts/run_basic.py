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


def bot_action(obs):
    return "draw" if obs["self_total"] < 17 else "stand"


def main():
    seed = 0x1234_5678_9ABC_DEF0
    bridge = Bridge()
    try:
        bridge.send({"cmd": "new", "seed": seed})
        total_rounds = 0
        while True:
            bridge.send({"cmd": "start_round"})
            # play the round
            while True:
                cur = bridge.send({"cmd": "current_player"})["current_player"]
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
    finally:
        bridge.close()


if __name__ == "__main__":
    sys.exit(main())
