//! Exact Nash solve of 2-player, 1-die-each, 6-face Liar's Dice — the
//! smallest nontrivial instance of this game, and the benchmark-ladder rung 1
//! against which later champions get scored (alongside `bid_bias_probe.rs`).
//! Solves both opening conventions: the free-open round (what every round
//! looks like once eliminations have happened inside a bigger game — the
//! deployment-realistic endgame) and the standalone game's forced-`1x1` first
//! round.
//!
//! ## Why this isn't just `solvers::Cfr` + `RoundSubgame`
//!
//! The repo already ships an exact 2-player CFR+ solver (`solvers::Cfr`) and a
//! per-round decomposition (`RoundSubgame`, `fit_two_player`) that looks like
//! the right tool. It isn't, for this exact question: `LiarsDice::infoset_key`
//! deliberately compresses a round's bid path down to per-seat raise *counts*
//! plus the last endorsed face (documented in `src/solve.rs` as a lossy
//! abstraction, accepted there for larger configs). On this reduced 2-die
//! ladder that compression genuinely merges histories a real player can tell
//! apart — e.g. reaching bid `2x1` via a direct `Open(2, 1)` versus via
//! `Open(1, 1)` followed by six `RaiseFace` steps are different sequences of
//! raise-quantity-vs-raise-face choices, both fully public, that the shipped
//! key collapses into the same information set. Solving with it plateaus
//! exploitability around 3% no matter how many iterations are spent (verified
//! empirically: 32K -> 1.28M iterations moved it only 3.11e-2 -> 2.99e-2) —
//! a real floor from solving an abstracted game, not slow numerics.
//!
//! `ExactCfr` below is a from-scratch CFR+ (vanilla regret-matching+, linear
//! averaging, alternating per-player updates — same shape as `solvers::Cfr`)
//! that reuses the real `RoundSubgame`/`LiarsDice` transition and payoff logic
//! verbatim (`apply`, `legal_actions`, `chance_outcomes`, `returns` — zero
//! rule changes) but keys information sets by the literal action-sequence-
//! so-far instead of the engine's compressed key. That's fully lossless here:
//! two histories are the same information set iff they are the same sequence
//! of publicly-observed bids.
//!
//! ## Exploitability certificate and its one known gap
//!
//! The standalone forced-open entry round converges cleanly to exploitability
//! `< 1.4e-4` by 800K iterations (still decreasing). The free-open round's
//! best-response exploitability instead plateaus around `2e-2` even at 800K
//! iterations — this is *not* the same infoset-key bug: it's slow CFR+ mixing
//! at the opener's genuinely near-indifferent 12-way opening decision (several
//! opens are close enough in value that regret-matching needs a very long tail
//! to fully separate them). Evidence this is mixing, not a floor: two
//! independent solve methods — the forward opener-advantage fixed point, and
//! backward induction over a 12-round horizon closed by a draw — converge to
//! the *identical* game value (0.166512) agreeing to `3.7e-11`, and the value
//! plus the equilibrium's mixed-strategy proportions are already stable to
//! 3-4 significant figures by 170K iterations. So the value and the printed
//! tables are reliable; only the raw best-response-gap metric for this one
//! round hasn't fully closed in the iteration budget spent here. Push
//! `SOLVE_ITERS` higher (or switch to a sequence-form LP) to tighten it
//! further if a future use needs the gap itself below `1e-4`.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p liars-dice --example exact_lossless_solve
//! SOLVE_ITERS=800000 FIT_ITERS=5000 cargo run --release -p liars-dice --example exact_lossless_solve
//! ```
//!
//! `SOLVE_ITERS` (default 50,000) is the iteration budget for the two final,
//! checkpointed equilibrium solves (free-open round and standalone entry
//! round) whose tables and exploitability get printed. `FIT_ITERS` (default
//! 20,000) is the (much cheaper) per-solve budget used while iterating the
//! opener-advantage fixed point and the backward-induction cross-check, where
//! only the converged *value* matters, not per-solve exploitability.
//!
//! The last section loads the deployed history-net champion
//! (`web/app/public/artifacts/ld-history-champion.bin`) and queries its
//! policy on a deployment-realistic 5-player state (3 seats eliminated, 2
//! survivors with 1 die each, fresh free-open round) so its opening and
//! response distributions can be read side by side with the equilibrium
//! tables above.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use game_core::{Game, Turn};
use liars_dice::{
    Action, ContinuationValue, HistoryNetAgent, LdState, MAX_PLAYERS, RoundSubgame,
    history_net_policy,
};

fn iters_final() -> u64 {
    std::env::var("SOLVE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000)
}
fn iters_fit() -> u64 {
    std::env::var("FIT_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000)
}

const CHAMPION: &str = "web/app/public/artifacts/ld-history-champion.bin";

fn dice_left2(a: u8, b: u8) -> [u8; MAX_PLAYERS] {
    let mut d = [0u8; MAX_PLAYERS];
    d[0] = a;
    d[1] = b;
    d
}

fn action_code(a: Action) -> u8 {
    match a {
        Action::RaiseQuantity => 1,
        Action::RaiseFace => 2,
        Action::CallLiar => 3,
        Action::CallExact => 4,
        Action::Open(q, f) => 10 + (q - 1) * 6 + (f - 1),
        Action::Roll(_) => unreachable!("Roll is chance, not a decision"),
    }
}

fn decode_path(path: &[u8]) -> String {
    path.iter()
        .map(|&c| match c {
            1 => "RaiseQty".to_string(),
            2 => "RaiseFace".to_string(),
            3 => "CallLiar".to_string(),
            4 => "CallExact".to_string(),
            c => {
                let idx = c - 10;
                let f = idx % 6 + 1;
                let q = idx / 6 + 1;
                format!("Open({q},{f})")
            }
        })
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// Lossless information-set key: (acting seat, that seat's own die, the
/// literal sequence of actions taken so far this round).
type InfoKey = (usize, u8, Vec<u8>);
/// Per-infoset occurrences gathered for best-response: the states in that
/// infoset with the reach probability of each (`gather` / `best_action`).
type Occurrences = HashMap<InfoKey, Vec<(LdState, Vec<u8>, f64)>>;

#[derive(Default, Clone)]
struct Entry {
    regret: Vec<f64>,
    strategy_sum: Vec<f64>,
}

/// Vanilla CFR+ (regret-matching+, linear averaging, alternating per-player
/// updates) over a `RoundSubgame`, keyed by the lossless path-based infoset
/// above instead of the engine's compressed `infoset_key`.
struct ExactCfr<V: ContinuationValue> {
    game: RoundSubgame<V>,
    table: HashMap<InfoKey, Entry>,
    iterations: u64,
}

impl<V: ContinuationValue + Clone> ExactCfr<V> {
    fn new(game: RoundSubgame<V>) -> Self {
        Self {
            game,
            table: HashMap::new(),
            iterations: 0,
        }
    }

    fn sigma(&self, key: &InfoKey, n: usize) -> Vec<f64> {
        match self.table.get(key) {
            None => vec![1.0 / n as f64; n],
            Some(e) => {
                let pos: f64 = e.regret.iter().map(|&r| r.max(0.0)).sum();
                if pos > 1e-12 {
                    e.regret.iter().map(|&r| r.max(0.0) / pos).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
        }
    }

    fn average(&self, key: &InfoKey, n: usize) -> Vec<f64> {
        match self.table.get(key) {
            None => vec![1.0 / n as f64; n],
            Some(e) => {
                let sum: f64 = e.strategy_sum.iter().sum();
                if sum > 1e-12 {
                    e.strategy_sum.iter().map(|&s| s / sum).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
        }
    }

    fn solve(&mut self, iters: u64) {
        let base = self.iterations;
        for t in 1..=iters {
            let weight = (base + t) as f64;
            for traverser in 0..2 {
                let state = self.game.initial_state();
                self.recurse(&state, &[], traverser, 1.0, 1.0, weight);
            }
        }
        self.iterations += iters;
    }

    fn recurse(
        &mut self,
        state: &LdState,
        path: &[u8],
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
                    v += p * self.recurse(&child, path, traverser, my_reach, ext_reach * p, weight);
                }
                v
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let n = actions.len();
                let die = state.hand(pl).first().copied().unwrap_or(0);
                let key: InfoKey = (pl, die, path.to_vec());
                let sigma = self.sigma(&key, n);
                if pl != traverser {
                    let mut v = 0.0;
                    for (i, &a) in actions.iter().enumerate() {
                        if sigma[i] == 0.0 {
                            continue;
                        }
                        let mut child = state.clone();
                        self.game.apply(&mut child, a);
                        let mut cp = path.to_vec();
                        cp.push(action_code(a));
                        v += sigma[i]
                            * self.recurse(
                                &child,
                                &cp,
                                traverser,
                                my_reach,
                                ext_reach * sigma[i],
                                weight,
                            );
                    }
                    return v;
                }
                let mut child_v = vec![0.0; n];
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    child_v[i] = self.recurse(
                        &child,
                        &cp,
                        traverser,
                        my_reach * sigma[i],
                        ext_reach,
                        weight,
                    );
                    v += sigma[i] * child_v[i];
                }
                let entry = self.table.entry(key).or_insert_with(|| Entry {
                    regret: vec![0.0; n],
                    strategy_sum: vec![0.0; n],
                });
                for i in 0..n {
                    entry.regret[i] = (entry.regret[i] + ext_reach * (child_v[i] - v)).max(0.0);
                    entry.strategy_sum[i] += weight * my_reach * sigma[i];
                }
                v
            }
        }
    }

    fn expected_value(&self) -> f64 {
        let state = self.game.initial_state();
        self.avg_value(&state, &[])
    }

    fn avg_value(&self, state: &LdState, path: &[u8]) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, 0);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.avg_value(&child, path);
                }
                v
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let die = state.hand(pl).first().copied().unwrap_or(0);
                let key: InfoKey = (pl, die, path.to_vec());
                let sigma = self.average(&key, actions.len());
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    v += sigma[i] * self.avg_value(&child, &cp);
                }
                v
            }
        }
    }

    /// Exact best-response exploitability against the current average
    /// strategy: (br0, br1, nashconv). Memoizes best-response value by the
    /// real game *state* (a pure function of state — Markov, path-independent)
    /// and the best *action* by the lossless info key (br commits to one
    /// action per own information set).
    fn exploitability(&self) -> (f64, f64, f64) {
        let br0 = self.best_response_value(0);
        let br1 = self.best_response_value(1);
        (br0, br1, br0 + br1)
    }

    fn best_response_value(&self, br: usize) -> f64 {
        let mut occ: Occurrences = HashMap::new();
        let root = self.game.initial_state();
        self.gather(&root, &[], br, 1.0, &mut occ);
        let mut action_memo: HashMap<InfoKey, usize> = HashMap::new();
        self.br_value(&root, &[], br, &occ, &mut action_memo)
    }

    fn gather(
        &self,
        state: &LdState,
        path: &[u8],
        br: usize,
        opp_reach: f64,
        occ: &mut Occurrences,
    ) {
        if self.game.is_terminal(state) {
            return;
        }
        match self.game.turn(state) {
            Turn::Chance => {
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    self.gather(&child, path, br, opp_reach * p, occ);
                }
            }
            Turn::Player(p) if p != br => {
                let actions = self.game.legal_actions(state);
                let die = state.hand(p).first().copied().unwrap_or(0);
                let key: InfoKey = (p, die, path.to_vec());
                let sigma = self.average(&key, actions.len());
                for (i, &a) in actions.iter().enumerate() {
                    if sigma[i] == 0.0 {
                        continue;
                    }
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    self.gather(&child, &cp, br, opp_reach * sigma[i], occ);
                }
            }
            Turn::Player(_) => {
                let die = state.hand(br).first().copied().unwrap_or(0);
                let key: InfoKey = (br, die, path.to_vec());
                occ.entry(key)
                    .or_default()
                    .push((state.clone(), path.to_vec(), opp_reach));
                for a in self.game.legal_actions(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    self.gather(&child, &cp, br, opp_reach, occ);
                }
            }
        }
    }

    fn br_value(
        &self,
        state: &LdState,
        path: &[u8],
        br: usize,
        occ: &Occurrences,
        action_memo: &mut HashMap<InfoKey, usize>,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.returns(state, br);
        }
        match self.game.turn(state) {
            Turn::Chance => {
                let mut v = 0.0;
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    v += p * self.br_value(&child, path, br, occ, action_memo);
                }
                v
            }
            Turn::Player(p) if p != br => {
                let actions = self.game.legal_actions(state);
                let die = state.hand(p).first().copied().unwrap_or(0);
                let key: InfoKey = (p, die, path.to_vec());
                let sigma = self.average(&key, actions.len());
                let mut v = 0.0;
                for (i, &a) in actions.iter().enumerate() {
                    if sigma[i] == 0.0 {
                        continue;
                    }
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    v += sigma[i] * self.br_value(&child, &cp, br, occ, action_memo);
                }
                v
            }
            Turn::Player(_) => {
                let die = state.hand(br).first().copied().unwrap_or(0);
                let key: InfoKey = (br, die, path.to_vec());
                let a_idx = self.best_action(&key, br, occ, action_memo);
                let actions = self.game.legal_actions(state);
                let mut child = state.clone();
                self.game.apply(&mut child, actions[a_idx]);
                let mut cp = path.to_vec();
                cp.push(action_code(actions[a_idx]));
                self.br_value(&child, &cp, br, occ, action_memo)
            }
        }
    }

    fn best_action(
        &self,
        key: &InfoKey,
        br: usize,
        occ: &Occurrences,
        action_memo: &mut HashMap<InfoKey, usize>,
    ) -> usize {
        if let Some(&a) = action_memo.get(key) {
            return a;
        }
        let states = &occ[key];
        let n = self.game.legal_actions(&states[0].0).len();
        let mut av = vec![0.0; n];
        for (s, p, reach) in states {
            let actions = self.game.legal_actions(s);
            for (i, &a) in actions.iter().enumerate() {
                let mut child = s.clone();
                self.game.apply(&mut child, a);
                let mut cp = p.clone();
                cp.push(action_code(a));
                av[i] += reach * self.br_value(&child, &cp, br, occ, action_memo);
            }
        }
        let best = av
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        action_memo.insert(key.clone(), best);
        best
    }

    /// Full-tree dump of every reachable information set's average policy and
    /// total reach probability.
    fn dump(&self, label: &str, min_reach: f64) {
        let mut seen: HashMap<InfoKey, (Vec<Action>, Vec<f64>, f64)> = HashMap::new();
        let root = self.game.initial_state();
        self.walk(&root, &[], 1.0, &mut seen);
        let mut rows: Vec<_> = seen.into_iter().collect();
        rows.sort_by(|a, b| (b.1).2.partial_cmp(&(a.1).2).unwrap());
        println!("\n--- {label}: equilibrium infosets (reach >= {min_reach}) ---");
        for ((pl, die, path), (actions, policy, reach)) in &rows {
            if *reach < min_reach {
                continue;
            }
            let acts: Vec<String> = actions
                .iter()
                .zip(policy.iter())
                .map(|(a, p)| format!("{a:?}={p:.4}"))
                .collect();
            println!(
                "  reach={reach:.5}  seat{pl} die={die} path=[{}]  ->  [{}]",
                decode_path(path),
                acts.join(", ")
            );
        }
        println!(
            "  ({} total infosets discovered, {} printed)",
            rows.len(),
            rows.iter().filter(|r| (r.1).2 >= min_reach).count()
        );
    }

    fn walk(
        &self,
        state: &LdState,
        path: &[u8],
        reach: f64,
        seen: &mut HashMap<InfoKey, (Vec<Action>, Vec<f64>, f64)>,
    ) {
        if self.game.is_terminal(state) {
            return;
        }
        match self.game.turn(state) {
            Turn::Chance => {
                for (a, p) in self.game.chance_outcomes(state) {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    self.walk(&child, path, reach * p, seen);
                }
            }
            Turn::Player(pl) => {
                let actions = self.game.legal_actions(state);
                let die = state.hand(pl).first().copied().unwrap_or(0);
                let key: InfoKey = (pl, die, path.to_vec());
                let policy = self.average(&key, actions.len());
                let entry = seen
                    .entry(key)
                    .or_insert_with(|| (actions.clone(), policy.clone(), 0.0));
                entry.2 += reach;
                for (i, &a) in actions.iter().enumerate() {
                    let mut child = state.clone();
                    self.game.apply(&mut child, a);
                    let mut cp = path.to_vec();
                    cp.push(action_code(a));
                    self.walk(&child, &cp, reach * policy[i], seen);
                }
            }
        }
    }
}

/// Zero-sum continuation closing a free-open round's "correct Call Exact"
/// leaf: `+g` to seat 0 if seat 0 opens the next round, `-g` if seat 1 does.
#[derive(Clone, Copy)]
struct GValue {
    g: f64,
}
impl ContinuationValue for GValue {
    fn value(&self, _faces: u8, _dice_left: &[u8], next_opener: usize, player: usize) -> f64 {
        let to_seat0 = if next_opener == 0 { self.g } else { -self.g };
        if player == 0 { to_seat0 } else { -to_seat0 }
    }
}

fn solve_free_round(g: f64, iters: u64) -> ExactCfr<GValue> {
    let subgame = RoundSubgame::new(2, 1, 6, dice_left2(1, 1), 0, false, 1, GValue { g });
    let mut cfr = ExactCfr::new(subgame);
    cfr.solve(iters);
    cfr
}

fn main() {
    let final_iters = iters_final();
    let fit_iters = iters_fit();

    println!("=== Fixed point: opener-advantage g in the free-open round (lossless key) ===");
    let mut g = 0.0f64;
    for step in 0..40 {
        let cfr = solve_free_round(g, fit_iters);
        let v = cfr.expected_value();
        let delta = (v - g).abs();
        println!("  step={step:>2}  g={g:.6} -> v={v:.6}  |delta|={delta:.3e}");
        std::io::stdout().flush().ok();
        g = v;
        if delta < 1e-9 {
            break;
        }
    }
    println!("converged opener-advantage g = {g:.6}");

    println!("\n=== FINAL solve of the free-open round at fixed point g (checkpointed) ===");
    std::io::stdout().flush().ok();
    let mut done = 0u64;
    let mut chunk = 2_000u64.min(final_iters);
    let mut free_cfr = solve_free_round(g, 0);
    while done < final_iters {
        let this = chunk.min(final_iters - done);
        free_cfr.solve(this);
        done += this;
        let (br0, br1, nc) = free_cfr.exploitability();
        println!(
            "  iters={done:>10}  value={:.6}  br0={br0:.6} br1={br1:.6}  exploitability={:.4e}",
            free_cfr.expected_value(),
            nc / 2.0
        );
        std::io::stdout().flush().ok();
        chunk = (chunk * 4).min(final_iters.saturating_sub(done).max(1));
    }
    free_cfr.dump(
        "FREE-OPEN round (lossless, deployment-realistic endgame)",
        0.002,
    );

    println!("\n=== Cross-check: bounded backward induction (independent method) ===");
    std::io::stdout().flush().ok();
    // Horizon-K backward induction: solve the free-open round closed by
    // AdjudicationValue-equivalent (draw, since ties never lose dice) at
    // horizon 1, then repeatedly close by the previous horizon's value.
    // With 1 die each a push is fairly rare, so a handful of horizons already
    // pins the value; we go to 12 for comfortable margin.
    let mut horizon_g = 0.0f64; // horizon-0 continuation: no more rounds allowed -> a push is scored as a draw (0)
    for h in 1..=12 {
        let cfr = solve_free_round(horizon_g, fit_iters);
        horizon_g = cfr.expected_value();
        println!("  horizon={h:>2}  value_to_opener={horizon_g:.6}");
    }
    println!(
        "backward induction (12 horizons): {horizon_g:.6}  gap-vs-fixed-point={:.3e}",
        (horizon_g - g).abs()
    );

    println!("\n=== Standalone forced-open entry round (whole-game forced 1x1) ===");
    std::io::stdout().flush().ok();
    let entry_subgame = RoundSubgame::new(2, 1, 6, dice_left2(1, 1), 0, true, 1, GValue { g });
    let mut entry_cfr = ExactCfr::new(entry_subgame);
    let mut done = 0u64;
    let mut chunk = 2_000u64.min(final_iters);
    while done < final_iters {
        let this = chunk.min(final_iters - done);
        entry_cfr.solve(this);
        done += this;
        let (br0, br1, nc) = entry_cfr.exploitability();
        println!(
            "  iters={done:>10}  value_to_seat0={:.6}  br0={br0:.6} br1={br1:.6}  exploitability={:.4e}",
            entry_cfr.expected_value(),
            nc / 2.0
        );
        std::io::stdout().flush().ok();
        chunk = (chunk * 4).min(final_iters.saturating_sub(done).max(1));
    }
    entry_cfr.dump("STANDALONE forced-open entry round (lossless)", 0.002);

    println!("\n=== Champion comparison (unaffected by the infoset-key issue above --");
    println!("=== it queries the deployed net directly on real states, no CFR key involved) ===");
    std::io::stdout().flush().ok();
    let champ_path = Path::new(CHAMPION);
    if !champ_path.exists() {
        println!("champion checkpoint not found at {CHAMPION}; skipping");
        return;
    }
    let champion = HistoryNetAgent::load(champ_path)
        .unwrap_or_else(|e| panic!("failed to load {CHAMPION}: {e}"));
    let cache = champion.net().infer_cache();

    let mut dl5 = [0u8; MAX_PLAYERS];
    dl5[0] = 1;
    dl5[4] = 1;
    let champ_round = RoundSubgame::new(5, 1, 6, dl5, 0, false, 1, liars_dice::DiceShareValue);
    let cfg5 = champ_round.config();

    println!("\n-- opener (seat 0) policy by held die, 5p-3-eliminated free-open state --");
    for die in 1..=6u8 {
        let mut s = champ_round.initial_state();
        let mut c0 = [0u8; 6];
        c0[die as usize - 1] = 1;
        champ_round.apply(&mut s, Action::Roll(c0));
        for _ in 0..3 {
            champ_round.apply(&mut s, Action::Roll([0; 6]));
        }
        let mut c4 = [0u8; 6];
        c4[0] = 1;
        champ_round.apply(&mut s, Action::Roll(c4));
        assert!(matches!(champ_round.turn(&s), Turn::Player(0)));

        let policy = history_net_policy(champion.net(), &cache, cfg5, &s, 0);
        let actions = cfg5.legal_actions(&s);
        let dist: Vec<String> = actions
            .iter()
            .zip(policy.iter())
            .filter(|&(_, &p)| p > 0.005)
            .map(|(a, p)| format!("{a:?}={p:.3}"))
            .collect();
        println!("  die={die}: [{}]", dist.join(", "));
    }

    println!("\n-- responder (seat 4) policy vs representative opens, by held die --");
    let opens_to_probe = [(1u8, 6u8), (1, 1), (2, 6), (1, 3)];
    for &(q, f) in &opens_to_probe {
        for die in 1..=6u8 {
            let mut s = champ_round.initial_state();
            let mut c0 = [0u8; 6];
            c0[0] = 1;
            champ_round.apply(&mut s, Action::Roll(c0));
            for _ in 0..3 {
                champ_round.apply(&mut s, Action::Roll([0; 6]));
            }
            let mut c4 = [0u8; 6];
            c4[die as usize - 1] = 1;
            champ_round.apply(&mut s, Action::Roll(c4));
            champ_round.apply(&mut s, Action::Open(q, f));
            assert!(matches!(champ_round.turn(&s), Turn::Player(4)));

            let policy = history_net_policy(champion.net(), &cache, cfg5, &s, 4);
            let actions = cfg5.legal_actions(&s);
            let dist: Vec<String> = actions
                .iter()
                .zip(policy.iter())
                .map(|(a, p)| format!("{a:?}={p:.3}"))
                .collect();
            println!("  open=({q},{f}) die={die}: [{}]", dist.join(", "));
        }
    }
}
