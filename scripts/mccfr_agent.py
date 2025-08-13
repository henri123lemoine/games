import json
import random
from collections import defaultdict
from pathlib import Path

CARDS = list(range(1, 12))
TARGET = 21


class RoundState:
    def __init__(self, rng: random.Random):
        deck = CARDS.copy()
        rng.shuffle(deck)
        self.deck = deck
        self.idx = 0
        self.up = [0, 0]
        self.down = [0, 0]
        self.total = [0, 0]
        self.stood = [False, False]
        # deal: p0 up, p1 up, p0 down, p1 down
        self.up[0] = self.draw_card()
        self.up[1] = self.draw_card()
        self.down[0] = self.draw_card()
        self.down[1] = self.draw_card()
        self.total[0] = self.up[0] + self.down[0]
        self.total[1] = self.up[1] + self.down[1]
        self.player = 0

    def draw_card(self):
        c = self.deck[self.idx]
        self.idx += 1
        return c

    def deck_count(self):
        # Number of cards remaining in the deck (0..11)
        return len(self.deck) - self.idx

    def terminal(self):
        return (self.stood[0] and self.stood[1]) or (self.idx >= len(self.deck))

    def winner(self):
        a_over = self.total[0] > TARGET
        b_over = self.total[1] > TARGET
        if a_over ^ b_over:
            return 1 if a_over else 0
        da = abs(TARGET - self.total[0])
        db = abs(TARGET - self.total[1])
        if da < db:
            return 0
        if db < da:
            return 1
        return None

    def obs_tuple(self, p):
        # Match the runtime agent’s infoset shape: use deck_count instead of full mask.
        # Tuple: (player, self_total, opp_face_up, self_stood, opp_stood, deck_count)
        return (
            p,
            self.total[p],
            self.up[1 - p],
            self.stood[p],
            self.stood[1 - p],
            self.deck_count(),
        )

    def apply(self, action):
        p = self.player
        if action == 0:  # draw
            c = self.draw_card()
            self.total[p] += c
            if self.total[p] > TARGET:
                self.stood[p] = True
        else:  # stand
            self.stood[p] = True
        # end or switch
        if not self.terminal():
            self.player ^= 1


class MCCFR:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.regret = defaultdict(lambda: [0.0, 0.0])
        self.strategy_sum = defaultdict(lambda: [0.0, 0.0])

    @staticmethod
    def _strategy_from_regret(regret):
        rp = [max(0.0, r) for r in regret]
        s = sum(rp)
        if s <= 1e-9:
            return [0.5, 0.5]
        return [rp[0] / s, rp[1] / s]

    def _get_strategy(self, info):
        return self._strategy_from_regret(self.regret[info])

    def _accumulate(self, info, strategy, reach):
        self.strategy_sum[info][0] += reach * strategy[0]
        self.strategy_sum[info][1] += reach * strategy[1]

    def cfr(self, state: RoundState, traverser: int, reach_my: float, reach_opp: float):
        if state.terminal():
            w = state.winner()
            if w is None:
                return 0.0
            return 1.0 if w == traverser else -1.0
        p = state.player
        info = state.obs_tuple(p)
        strategy = self._get_strategy(info)
        if p != traverser:
            # sample external opponent action
            a = 0 if self.rng.random() < strategy[0] else 1
            next_state = self._clone_state(state)
            next_state.apply(a)
            return self.cfr(next_state, traverser, reach_my, reach_opp * strategy[a])
        # traverser: evaluate both actions
        utils = [0.0, 0.0]
        node_util = 0.0
        for a in (0, 1):
            next_state = self._clone_state(state)
            next_state.apply(a)
            utils[a] = self.cfr(next_state, traverser, reach_my * strategy[a], reach_opp)
            node_util += strategy[a] * utils[a]
        # regret update
        cf_reach = reach_opp
        for a in (0, 1):
            self.regret[info][a] += cf_reach * (utils[a] - node_util)
        # accumulate average strategy
        self._accumulate(info, strategy, reach_my)
        return node_util

    @staticmethod
    def _clone_state(s: RoundState) -> RoundState:
        ns = object.__new__(RoundState)
        ns.deck = s.deck.copy()
        ns.idx = s.idx
        ns.up = s.up.copy()
        ns.down = s.down.copy()
        ns.total = s.total.copy()
        ns.stood = s.stood.copy()
        ns.player = s.player
        return ns

    def train(self, iterations=10000):
        for t in range(iterations):
            # alternate traverser
            trav = t & 1
            s = RoundState(self.rng)
            self.cfr(s, trav, reach_my=1.0, reach_opp=1.0)

    def average_policy(self):
        pol = {}
        for info, sums in self.strategy_sum.items():
            s = sums[0] + sums[1]
            if s <= 1e-9:
                pol[info] = [0.5, 0.5]
            else:
                pol[info] = [sums[0] / s, sums[1] / s]
        return pol


def save_policy(policy, path):
    # convert tuple keys to strings for JSON
    sp = {str(k): v for k, v in policy.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(sp, f)


def main():
    m = MCCFR(seed=42)
    m.train(iterations=5000000)
    pol = m.average_policy()
    save_policy(pol, "data/policy_mccfr.json")
    print(f"Saved policy with {len(pol)} infosets to data/policy_mccfr.json")


if __name__ == "__main__":
    main()
