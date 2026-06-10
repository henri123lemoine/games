use game_core::{Game, Turn};

use crate::FastMap;

/// Index of the maximum element (first on ties).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Regret-matching strategy from a regret vector (CFR+: only positive regrets).
fn regret_match(regret: &[f64]) -> Vec<f64> {
    let sum: f64 = regret.iter().map(|r| r.max(0.0)).sum();
    if sum > 0.0 {
        regret.iter().map(|r| r.max(0.0) / sum).collect()
    } else {
        let n = regret.len();
        vec![1.0 / n as f64; n]
    }
}

/// Generic CFR+ solver with exact best-response exploitability.
///
/// Training is vanilla CFR+ (full tree traversal each iteration: chance and both
/// players enumerated, with explicit reach probabilities), which is exact and
/// deterministic — appropriate for games small enough that their best response
/// is enumerable. Monte-Carlo sampling is a later optimization for larger games.
pub struct Cfr<G: Game> {
    game: G,
    regret: FastMap<u64, Vec<f64>>,
    strategy: FastMap<u64, Vec<f64>>,
    iterations: u64,
}

impl<G: Game> Cfr<G> {
    pub fn new(game: G, _seed: u64) -> Self {
        Self {
            game,
            regret: FastMap::default(),
            strategy: FastMap::default(),
            iterations: 0,
        }
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn num_infosets(&self) -> usize {
        self.strategy.len()
    }

    pub fn game(&self) -> &G {
        &self.game
    }

    /// Run `iters` CFR+ iterations. Each is one full traversal updating both
    /// players' regrets; the average strategy is weighted linearly by iteration.
    pub fn solve(&mut self, iters: u64) {
        let base = self.iterations;
        for t in 1..=iters {
            let weight = (base + t) as f64;
            let state = self.game.initial_state();
            self.cfr(&state, 1.0, 1.0, 1.0, weight);
        }
        self.iterations += iters;
    }

    /// CFR+ traversal returning the node value to player 0. `r0`/`r1` are the
    /// players' reach probabilities and `chance` the chance reach to this node.
    fn cfr(&mut self, state: &G::State, r0: f64, r1: f64, chance: f64, weight: f64) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, 0);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.cfr(&child, r0, r1, chance * p, weight);
                }
                v
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let n = actions.len();
                let key = self.game.infoset_key(state, pl);
                let sigma = {
                    let r = self.regret.entry(key).or_insert_with(|| vec![0.0; n]);
                    regret_match(r)
                };
                let mut child_v = vec![0.0; n];
                let mut v0 = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let (nr0, nr1) = if pl == 0 {
                        (r0 * sigma[i], r1)
                    } else {
                        (r0, r1 * sigma[i])
                    };
                    child_v[i] = self.cfr(&child, nr0, nr1, chance, weight);
                    v0 += sigma[i] * child_v[i];
                }
                let (my_reach, opp_reach) = if pl == 0 { (r0, r1) } else { (r1, r0) };
                let sign = if pl == 0 { 1.0 } else { -1.0 };
                let cf = opp_reach * chance;
                {
                    let r = self.regret.get_mut(&key).unwrap();
                    for i in 0..n {
                        r[i] = (r[i] + cf * sign * (child_v[i] - v0)).max(0.0);
                    }
                }
                {
                    let s = self.strategy.entry(key).or_insert_with(|| vec![0.0; n]);
                    for i in 0..n {
                        s[i] += weight * my_reach * sigma[i];
                    }
                }
                v0
            }
        }
    }

    /// Average strategy at an information set (normalized strategy sums), or a
    /// uniform distribution over `n` actions if the set was never visited.
    fn average_strategy(&self, key: u64, n: usize) -> Vec<f64> {
        match self.strategy.get(&key) {
            Some(s) => {
                let sum: f64 = s.iter().sum();
                if sum > 0.0 {
                    s.iter().map(|x| x / sum).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
            None => vec![1.0 / n as f64; n],
        }
    }

    /// Exact best-response exploitability. Returns `(br0, br1, nashconv)` where
    /// `br_i` is the value player `i` obtains by best-responding to the current
    /// average strategy (chance integrated exactly), NashConv = br0 + br1, and
    /// exploitability = NashConv / 2 (zero exactly at a Nash equilibrium).
    pub fn exploitability(&self) -> (f64, f64, f64) {
        let br0 = self.best_response_value(0);
        let br1 = self.best_response_value(1);
        (br0, br1, br0 + br1)
    }

    /// Exact best-response value for player `br` against the current average
    /// strategy. The best response is over *information sets*: `br` commits to one
    /// action per infoset (it cannot see the opponent's hidden state), avoiding
    /// the strategy fusion of a per-state maximum.
    ///
    /// Pass 1 (`gather`) records, for each `br` infoset, the states in it weighted
    /// by counterfactual (opponent + chance) reach. Pass 2 is a memoized mutual
    /// recursion: `br_value(state)` returns the value under `br`'s optimal play,
    /// and `best_action(infoset)` picks the action maximizing the reach-weighted
    /// value over that infoset's states. Both memoize, so the game DAG is walked
    /// once. (Counterfactual reach excludes `br`'s own probabilities, so an
    /// infoset's states and weights don't depend on `br`'s own choices.)
    fn best_response_value(&self, br: usize) -> f64 {
        let mut occ: FastMap<u64, Vec<(G::State, f64)>> = FastMap::default();
        let root = self.game.initial_state();
        self.gather(&root, br, 1.0, &mut occ);
        let mut value_memo: FastMap<u64, f64> = FastMap::default();
        let mut action_memo: FastMap<u64, usize> = FastMap::default();
        self.br_value(&root, br, &occ, &mut value_memo, &mut action_memo)
    }

    fn gather(
        &self,
        state: &G::State,
        br: usize,
        opp_reach: f64,
        occ: &mut FastMap<u64, Vec<(G::State, f64)>>,
    ) {
        if self.game.is_terminal(state) {
            return;
        }
        match self.game.turn(state) {
            Turn::Chance => {
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    self.gather(&child, br, opp_reach * p, occ);
                }
            }
            Turn::Player(p) if p != br => {
                let actions = self.game.legal_actions(state);
                let sigma = self.average_strategy(self.game.infoset_key(state, p), actions.len());
                for (i, &a) in actions.iter().enumerate() {
                    if sigma[i] == 0.0 {
                        continue;
                    }
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    self.gather(&child, br, opp_reach * sigma[i], occ);
                }
            }
            Turn::Player(_) => {
                let key = self.game.infoset_key(state, br);
                occ.entry(key).or_default().push((state.clone(), opp_reach));
                for a in self.game.legal_actions(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    self.gather(&child, br, opp_reach, occ);
                }
            }
        }
    }

    fn br_value(
        &self,
        state: &G::State,
        br: usize,
        occ: &FastMap<u64, Vec<(G::State, f64)>>,
        value_memo: &mut FastMap<u64, f64>,
        action_memo: &mut FastMap<u64, usize>,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, br);
        }
        let sk = self.game.state_key(state);
        if let Some(k) = sk
            && let Some(v) = value_memo.get(&k)
        {
            return *v;
        }
        let v = match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.br_value(&child, br, occ, value_memo, action_memo);
                }
                v
            }
            Turn::Player(p) if p != br => {
                let actions = self.game.legal_actions(state);
                let sigma = self.average_strategy(self.game.infoset_key(state, p), actions.len());
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    if sigma[i] == 0.0 {
                        continue;
                    }
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += sigma[i] * self.br_value(&child, br, occ, value_memo, action_memo);
                }
                v
            }
            Turn::Player(_) => {
                let key = self.game.infoset_key(state, br);
                let a = self.best_action(key, br, occ, value_memo, action_memo);
                let actions = self.game.legal_actions(state);
                let mut child = state.clone();
                self.game.apply(&mut child, actions[a]);
                self.br_value(&child, br, occ, value_memo, action_memo)
            }
        };
        if let Some(k) = sk {
            value_memo.insert(k, v);
        }
        v
    }

    fn best_action(
        &self,
        infoset: u64,
        br: usize,
        occ: &FastMap<u64, Vec<(G::State, f64)>>,
        value_memo: &mut FastMap<u64, f64>,
        action_memo: &mut FastMap<u64, usize>,
    ) -> usize {
        if let Some(a) = action_memo.get(&infoset) {
            return *a;
        }
        let states = &occ[&infoset];
        let n = self.game.legal_actions(&states[0].0).len();
        let mut av = vec![0.0; n];
        for (s, reach) in states {
            let actions = self.game.legal_actions(s);
            for (i, &a) in actions.iter().enumerate() {
                let mut child = s.clone();
                self.game.apply(&mut child, a);
                av[i] += reach * self.br_value(&child, br, occ, value_memo, action_memo);
            }
        }
        let a = argmax(&av);
        action_memo.insert(infoset, a);
        a
    }

    /// The average-strategy distribution over the legal actions at `state`'s
    /// information set (for play / inspection).
    pub fn policy(&self, state: &G::State, player: usize) -> Vec<f64> {
        let n = self.game.legal_actions(state).len();
        self.average_strategy(self.game.infoset_key(state, player), n)
    }

    /// Sample an action index from the average strategy, given a uniform `r` in
    /// `[0, 1)`. Suitable as an [`crate::Agent`] for the arena.
    pub fn sample_action(&self, state: &G::State, player: usize, r: f64) -> usize {
        let policy = self.policy(state, player);
        let mut acc = 0.0;
        for (i, p) in policy.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        policy.len() - 1
    }

    /// Expected value to player 0 of the root when *both* players play their
    /// average strategy (chance integrated). At a Nash this equals the game value.
    pub fn expected_value(&self) -> f64 {
        let state = self.game.initial_state();
        self.avg_value(&state)
    }

    fn avg_value(&self, state: &G::State) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, 0);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.avg_value(&child);
                }
                v
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let sigma = self.average_strategy(self.game.infoset_key(state, pl), actions.len());
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += sigma[i] * self.avg_value(&child);
                }
                v
            }
        }
    }
}
