use game_core::{Game, Rng, Turn};

use crate::FastMap;
use crate::tabular::{argmax, normalized_or_uniform, regret_match};

/// Generic CFR+ solver with exact best-response exploitability.
///
/// Training is vanilla CFR+ (full tree traversal, chance and opponent
/// enumerated with explicit reach probabilities), which is exact and
/// deterministic — appropriate for games small enough that their best response
/// is enumerable. Updates alternate between players each iteration, as CFR+'s
/// convergence guarantee requires (simultaneous updates with the regret floor
/// are known to oscillate on some games). Monte-Carlo sampling is a later
/// optimization for larger games.
pub struct Cfr<G: Game> {
    game: G,
    regret: FastMap<u64, Vec<f64>>,
    strategy: FastMap<u64, Vec<f64>>,
    iterations: u64,
}

impl<G: Game> Cfr<G> {
    pub fn new(game: G) -> Self {
        assert_eq!(
            game.num_players(),
            2,
            "Cfr is a 2-player solver; {}-player games need Mccfr or a bespoke decomposition",
            game.num_players()
        );
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

    /// Run `iters` CFR+ iterations. Each iteration traverses the tree once per
    /// player, updating only that player's regrets and average strategy; the
    /// average strategy is weighted linearly by iteration.
    pub fn solve(&mut self, iters: u64) {
        let base = self.iterations;
        for t in 1..=iters {
            let weight = (base + t) as f64;
            for traverser in 0..2 {
                let state = self.game.initial_state();
                self.cfr(&state, traverser, 1.0, 1.0, weight);
            }
        }
        self.iterations += iters;
    }

    /// Alternating CFR+ traversal returning the node value to `traverser`.
    /// `my_reach` is the traverser's own reach probability; `ext_reach` is
    /// everyone else's (opponent and chance combined) — the counterfactual
    /// weight of this node.
    fn cfr(
        &mut self,
        state: &G::State,
        traverser: usize,
        my_reach: f64,
        ext_reach: f64,
        weight: f64,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, traverser);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.cfr(&child, traverser, my_reach, ext_reach * p, weight);
                }
                v
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let n = actions.len();
                let key = self.game.infoset_key(state, pl);
                let sigma = {
                    let r = self.regret.entry(key).or_insert_with(|| vec![0.0; n]);
                    debug_assert_eq!(
                        r.len(),
                        n,
                        "action count changed for infoset {key:#x} — legal_actions must be \
                         stable per information set"
                    );
                    regret_match(r)
                };
                if pl != traverser {
                    let mut v = 0.0;
                    for (i, &a) in actions.iter().enumerate() {
                        if sigma[i] == 0.0 {
                            continue;
                        }
                        let mut child = state.clone();
                        self.game.apply(&mut child, a);
                        v += sigma[i]
                            * self.cfr(&child, traverser, my_reach, ext_reach * sigma[i], weight);
                    }
                    return v;
                }
                let mut child_v = vec![0.0; n];
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    child_v[i] =
                        self.cfr(&child, traverser, my_reach * sigma[i], ext_reach, weight);
                    v += sigma[i] * child_v[i];
                }
                let r = self.regret.get_mut(&key).unwrap();
                for i in 0..n {
                    r[i] = (r[i] + ext_reach * (child_v[i] - v)).max(0.0);
                }
                let s = self.strategy.entry(key).or_insert_with(|| vec![0.0; n]);
                for i in 0..n {
                    s[i] += weight * my_reach * sigma[i];
                }
                v
            }
        }
    }

    /// Average strategy at an information set (normalized strategy sums), or a
    /// uniform distribution over `n` actions if the set was never visited.
    fn average_strategy(&self, key: u64, n: usize) -> Vec<f64> {
        normalized_or_uniform(self.strategy.get(&key), n)
    }

    /// Exact best-response exploitability. Returns `(br0, br1, nashconv)` where
    /// `br_i` is the value player `i` obtains by best-responding to the current
    /// average strategy (chance integrated exactly), NashConv = br0 + br1, and
    /// exploitability = NashConv / 2 (zero exactly at a Nash equilibrium).
    pub fn exploitability(&self) -> (f64, f64, f64) {
        #[cfg(feature = "parallel")]
        let (br0, br1) = rayon::join(
            || self.best_response_value(0),
            || self.best_response_value(1),
        );
        #[cfg(not(feature = "parallel"))]
        let (br0, br1) = (self.best_response_value(0), self.best_response_value(1));
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

    /// Sample an action index from the average strategy. Suitable as an
    /// [`game_core::Agent`] for the arena.
    pub fn sample_action(&self, state: &G::State, player: usize, rng: &mut Rng) -> usize {
        rng.pick(&self.policy(state, player))
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
