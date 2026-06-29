//! Online subgame solving for Liar's Dice — DeepStack/Pluribus-style test-time
//! search.
//!
//! Liar's Dice re-rolls *all* dice at the start of every round, so each round
//! begins with uniform, independent hidden hands and no belief carries across
//! rounds. The optimal play at the hero's current decision can therefore be
//! computed by solving the *current round* from its opening (uniform hand priors
//! over every seat + the actual within-round bid sequence) against a
//! [`ContinuationValue`] for the post-round states — exactly the
//! [`RoundSubgame`] decomposition the offline fitting already relies on. This is
//! the *test-time* counterpart: instead of fitting a value table or a net once,
//! [`OnlineSolveAgent`] re-solves the live round on every move and plays its
//! converged strategy at the hero's own information set.
//!
//! ## Live state -> round subgame -> infoset
//!
//! At a decision the live [`LdState`] has already had this round's bids applied.
//! Three public facts reconstruct the round's opening:
//!   * the current per-seat dice vector ([`LdState::dice_left`]),
//!   * whether this is the game's forced-`1×1` first round
//!     ([`LdState::first_round`]),
//!   * the seat that opened the round ([`LiarsDice::round_opener`], recovered
//!     from `last_bidder` and the per-seat bid counts).
//!
//! [`RoundSubgame::new`] rebuilds the round-opening state from those, with the
//! inner [`LiarsDice`] configured for the live `players`/`faces`. Because the
//! subgame *wraps* the real game (byte-for-byte bidding, call resolution, and —
//! crucially — the same [`Game::infoset_key`]), replaying the live bid sequence
//! through it reaches an information set keyed identically to the hero's live
//! one. So after solving we look the hero's policy up at the *live* state and
//! sample a legal action from it. The agent verifies this infoset-reproduction
//! invariant in debug builds (`debug_assert`) before trusting the lookup.
//!
//! ## Solver
//!
//! The single uniform mechanism across every round size and player count is
//! external-sampling MCCFR ([`Mccfr`]). It samples chance and the opponents to
//! one action each but expands *all* of the traverser's actions, so the per-seat
//! cost is independent of opponents' branching and it scales to any player count
//! and round size while *converging the wide opening node*. That last point is
//! decisive here: a free-open round offers `total_dice × faces` opening bids
//! (54 at 3p3d6f), and outcome-sampling MCCFR — which samples a single opening
//! per traversal — leaves the average strategy near uniform over them, opening
//! absurd bids like `7×4` a third of the time and losing dice immediately.
//! External sampling estimates every opening's value each traversal, so it
//! concentrates the opener on plausible bids at a fraction of the iterations.
//!
//! Exact [`solvers::Cfr`] is 2-player only and enumerates chance and the full
//! bidding tree — it matches the equilibrium exactly on 2p1d6f (the correctness
//! reference) but is already > 1 s/move at 2p2d6f and intractable beyond
//! (chance fan-out is `C(d + f - 1, f - 1)` per seat), so it cannot be the
//! deploy path. The validation pass confirms `Mccfr` at the deploy budget
//! reproduces the exact-CFR call frequencies on the thin-bid probes (the over-
//! calling fix) *and* opens sanely on wide multiplayer rounds.
//!
//! ## Continuation value
//!
//! The agent is parameterized over a *factory* `FnMut() -> V` rather than a
//! single `V`, so each per-move solve gets a fresh continuation. That is what
//! [`NetValue`](crate::NetValue) (the fitted value head) needs — a fresh
//! per-round instance referencing the shared net/cache, with its own bounded
//! memo — and it costs the cheap [`DiceShareValue`](crate::DiceShareValue) used
//! for this validation pass nothing. Deploying against the value net is the same
//! agent with a `|| NetValue::new(net, cache, players, faces)` factory; no other
//! code changes.

use game_core::{Agent, Game, Rng};
use solvers::Mccfr;
use solvers::azero::{InferCache, Mlp};

use crate::{ContinuationValue, LdState, LiarsDice, MAX_PLAYERS, NetValue, RoundSubgame};

/// Test-time subgame-solving agent: re-solves the live round with `Mccfr`
/// (external-sampling MCCFR+) against a fresh continuation value `V` and plays
/// the converged strategy at the hero's information set.
///
/// `make_value` builds the post-round continuation for each solve. The
/// validation pass uses `|| DiceShareValue`; deploy uses a closure capturing the
/// value net. The factory takes `&self`-borrowed captures, so it must be `Sync`
/// for parallel match play (the arena shares one agent across seats/games).
pub struct OnlineSolveAgent<V, F>
where
    V: ContinuationValue,
    F: Fn() -> V + Sync,
{
    make_value: F,
    cfg: OnlineSolveConfig,
}

/// Tunables for the online solve.
///
/// The defaults are calibrated on the 2p1d6f thin-bid probe and the multiplayer
/// opening (see `examples/online_eval.rs`): they land the agent's `P(CallLiar)`
/// near the exact equilibrium on the thin `1×5` and ~0.00 when the hero holds
/// the bid face — the over-calling fix — while opening sanely on wide rounds.
#[derive(Clone, Copy, Debug)]
pub struct OnlineSolveConfig {
    /// Base `Mccfr` iteration budget *per restart*. One iteration runs one
    /// external-sampling traversal per seat (every traverser action expanded),
    /// so a restart's traversal count is `iters * players` and a move's total is
    /// `restarts * iters * players`. Scaled down for large rounds (see
    /// [`OnlineSolveAgent::budget`]). External sampling converges the wide
    /// opening in a few thousand iterations, far fewer than outcome sampling.
    pub iters: u64,
    /// Hard ceiling on the per-restart `iters` after the size scaling, so a
    /// pathologically large round can never make a single move unboundedly slow.
    pub max_iters: u64,
    /// Independent solves whose hero policies are averaged. MCCFR's single-
    /// infoset average strategy still carries sampling noise; averaging `restarts`
    /// independent solves shrinks the spread by ~√restarts, steadying the play.
    /// 1 disables averaging.
    pub restarts: usize,
    /// PRNG seed base. Mixed with a per-call nonce so repeated solves of the same
    /// situation are re-solved independently (the agent stays non-deterministic
    /// across a match) yet a fixed run seed reproduces the run.
    pub seed: u64,
    /// Diagnostic override: when `Some(it)`, the per-round budget is a flat
    /// `(it, restarts)` regardless of round size — the `total_dice²` work
    /// ceiling is bypassed entirely. `None` keeps the calibrated size scaling
    /// (the deploy path). Used to isolate the solve-budget bottleneck from the
    /// continuation value (see `examples/budget_diag.rs`).
    pub flat_iters: Option<u64>,
}

impl Default for OnlineSolveConfig {
    fn default() -> Self {
        Self {
            iters: 8_000,
            max_iters: 8_000,
            restarts: 3,
            seed: 0xA5_0117_0E50_17E5,
            flat_iters: None,
        }
    }
}

impl<V, F> OnlineSolveAgent<V, F>
where
    V: ContinuationValue,
    F: Fn() -> V + Sync,
{
    /// Build the agent with `make_value` as the per-solve continuation factory
    /// and the default solve budget.
    pub fn new(make_value: F) -> Self {
        Self {
            make_value,
            cfg: OnlineSolveConfig::default(),
        }
    }

    /// Build with an explicit solve configuration.
    pub fn with_config(make_value: F, cfg: OnlineSolveConfig) -> Self {
        Self { make_value, cfg }
    }

    /// The per-move solve budget for the live round, as `(iters_per_restart,
    /// restarts)`. `Mccfr::run` does one external-sampling traversal *per seat*
    /// per iteration, expanding every traverser action down the bidding ladder,
    /// so the measured per-iteration cost grows with roughly the *square* of the
    /// total dice (ladder depth × the per-node fan-out, both ∝ dice): ≈5 µs at
    /// 2p2d6f, ≈90 µs at 2p5d6f, ≈800 µs at 6p8d6f.
    ///
    /// To keep a single move near the ~1 s target the budget holds the total
    /// per-move iteration count `× total_dice²` under a fixed ceiling: it first
    /// trims `iters` (down to `MIN_ITERS`, below which the wide opening node would
    /// not leave its uniform prior and the agent would open absurd bids), then —
    /// only once `iters` is already floored — trims the restart count. Small and
    /// medium rounds keep the full `iters × restarts`; only the wide, many-dice
    /// tables are throttled (the largest, 6p8d6f, drops to a single floor-budget
    /// solve, ≈0.8 s — also the configs `DiceShareValue` under-serves anyway, the
    /// report's documented caveat). Calibrated against the speed table in
    /// `examples/online_eval.rs`.
    fn budget(&self, total_dice: u32) -> (u64, usize) {
        // Diagnostic override: a flat per-round budget, no size scaling.
        if let Some(it) = self.cfg.flat_iters {
            return (it.max(1), self.cfg.restarts.max(1));
        }
        // Enough iterations for the wide opening to leave the uniform prior even
        // at a single restart; below this the agent opens absurd bids.
        const MIN_ITERS: u64 = 1_000;
        // Per-move ceiling in `iters × total_dice²` units, tuned so 2p5d6f lands
        // ~0.6 s and 6p8d6f ~0.8 s — i.e. under the ~1 s target across the board.
        const WORK_CEILING: u64 = 700_000;
        let td = u64::from(total_dice.max(1));
        let min_iters = MIN_ITERS.min(self.cfg.max_iters);
        let full_restarts = self.cfg.restarts.max(1);
        // Total iterations across all restarts the ceiling affords, never above
        // the requested `iters × restarts`, never below one floor-budget solve.
        let affordable_total = WORK_CEILING / (td * td);
        let total = affordable_total
            .min(self.cfg.iters * full_restarts as u64)
            .max(min_iters);
        // Split into restarts of >= MIN_ITERS each (each solve must converge the
        // opening); drop restarts before going below the floor.
        let restarts = ((total / min_iters).max(1) as usize).min(full_restarts);
        let iters = (total / restarts as u64).clamp(min_iters, self.cfg.max_iters);
        (iters, restarts)
    }

    /// Build the round subgame for the live state, matching the conventions of
    /// the round the hero is in (dice vector, opener, first-round flag). The
    /// continuation is a fresh `(self.make_value)()`. Exposed so the validation
    /// harness rebuilds the exact same subgame the agent solves (single source of
    /// truth for the live-state -> subgame mapping).
    pub fn round_subgame(&self, game: &LiarsDice, state: &LdState) -> RoundSubgame<V> {
        let mut dice_left = [0u8; MAX_PLAYERS];
        dice_left[..game.players as usize]
            .copy_from_slice(&state.dice_left()[..game.players as usize]);
        let opener = game.round_opener(state);
        RoundSubgame::new(
            game.players,
            game.dice,
            game.faces,
            dice_left,
            opener,
            state.first_round(),
            1, // a fresh single round: never near the cap, so the start round is 1.
            (self.make_value)(),
        )
    }

    /// The live round's per-restart iteration count and restart count, after the
    /// size-based budget scaling (see [`OnlineSolveAgent::budget`]).
    fn round_budget(&self, game: &LiarsDice, state: &LdState) -> (u64, usize) {
        let total_dice: u32 = state.dice_left()[..game.players as usize]
            .iter()
            .map(|&d| u32::from(d))
            .sum();
        self.budget(total_dice)
    }

    /// One external-sampling MCCFR solve of the live round at `iters`, returning
    /// the solver (so the caller can read any infoset's converged policy).
    /// `nonce` seeds it.
    fn solve_once(
        &self,
        game: &LiarsDice,
        state: &LdState,
        iters: u64,
        nonce: u64,
    ) -> Mccfr<RoundSubgame<V>> {
        let round = self.round_subgame(game, state);
        let mut solver = Mccfr::new(round, self.cfg.seed ^ nonce);
        solver.run(iters);
        solver
    }

    /// The hero's averaged policy over the budgeted restarts of independent
    /// solves of the live round — the agent's actual mixed strategy at `state`.
    /// Averaging reduces MCCFR's single-infoset sampling variance (see
    /// [`OnlineSolveConfig`]). `nonce` seeds the batch; restarts use distinct
    /// derived seeds.
    pub fn solve_policy(
        &self,
        game: &LiarsDice,
        state: &LdState,
        player: usize,
        nonce: u64,
    ) -> Vec<f64> {
        let n = game.num_actions(state);
        let (iters, restarts) = self.round_budget(game, state);
        let mut avg = vec![0.0; n];
        for k in 0..restarts {
            // A distinct sub-seed per restart so the solves are independent.
            let sub = nonce
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(k as u64 + 1);
            let solver = self.solve_once(game, state, iters, sub);
            debug_assert_eq!(
                game.infoset_key(state, player),
                solver.game().infoset_key(state, player),
                "online-solve subgame must reproduce the hero's live infoset key"
            );
            let p = solver.policy(state, player);
            debug_assert_eq!(
                p.len(),
                n,
                "policy width must match live legal-action count"
            );
            for (a, &pv) in avg.iter_mut().zip(&p) {
                *a += pv / restarts as f64;
            }
        }
        avg
    }
}

impl<V, F> Agent<LiarsDice> for OnlineSolveAgent<V, F>
where
    V: ContinuationValue,
    F: Fn() -> V + Sync,
{
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let actions = game.legal_actions(state);
        // A forced move needs no solve — and a solve would waste a budget on a
        // singleton infoset.
        if actions.len() == 1 {
            return 0;
        }

        // The nonce from the agent's private RNG stream re-solves repeated
        // identical situations independently (no readable determinism) while a
        // fixed run seed stays reproducible. The averaged policy is keyed on the
        // hero's live information set (own hand + within-round bid context),
        // because the subgame wraps the real game's `infoset_key`.
        let nonce = rng.next_u64();
        let probs = self.solve_policy(game, state, player, nonce);
        // Sampling from a distribution over the live legal actions always yields
        // a legal index (the table read is uniform when unvisited — still legal).
        rng.pick(&probs)
    }
}

/// The deploy online-solver: an [`OnlineSolveAgent`] whose per-move continuation
/// is the trained value head ([`NetValue`]).
///
/// [`OnlineSolveAgent`] is parameterized over a continuation *factory*
/// `Fn() -> V`, and `NetValue<'a>` borrows the net and its [`InferCache`], so the
/// factory must hand out fresh `NetValue`s that borrow some owner. This wrapper is
/// that owner: it holds the `Mlp` and `InferCache`, and builds the
/// `OnlineSolveAgent` locally inside [`Agent::act`] with a closure that borrows
/// them for the duration of the move. Building it per-call (rather than storing
/// the agent) sidesteps a self-referential struct cleanly and costs nothing — the
/// move's work is the MCCFR solve, not the agent construction. The continuation is
/// built for the live game's `(players, faces)`, so one wrapper plays any config.
pub struct NetOnlineSolveAgent {
    net: Mlp,
    cache: InferCache,
    cfg: OnlineSolveConfig,
}

impl NetOnlineSolveAgent {
    /// Wrap `net` as the online-solver's value continuation with the default
    /// deploy budget.
    pub fn new(net: Mlp) -> Self {
        Self::with_config(net, OnlineSolveConfig::default())
    }

    /// Wrap `net` with an explicit solve configuration.
    pub fn with_config(net: Mlp, cfg: OnlineSolveConfig) -> Self {
        let cache = net.infer_cache();
        Self { net, cache, cfg }
    }

    /// Load the value net from a serialized [`Mlp`] checkpoint.
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::from_bytes(data)?))
    }
}

impl Agent<LiarsDice> for NetOnlineSolveAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let players = game.players;
        let faces = game.faces;
        let make_value = || NetValue::new(&self.net, &self.cache, players, faces);
        OnlineSolveAgent::with_config(make_value, self.cfg).act(game, state, player, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, DiceShareValue, MAX_FACES};
    use game_core::Turn;

    /// The reconstructed subgame must reproduce the hero's live infoset key at
    /// every in-round decision node of a real game — the invariant the policy
    /// lookup depends on. Checked across player counts and a random bid line.
    #[test]
    fn subgame_reproduces_live_infoset_key() {
        let agent = OnlineSolveAgent::new(|| DiceShareValue);
        for &players in &[2u8, 3, 4] {
            let game = LiarsDice::new(players, 3, 6);
            let mut rng = Rng::new(0xF00D + u64::from(players));
            for _ in 0..30 {
                let mut s = game.initial_state();
                while !game.is_terminal(&s) {
                    match game.turn(&s) {
                        Turn::Chance => {
                            let a = game.sample_chance_action(&s, &mut rng);
                            game.apply(&mut s, a);
                        }
                        Turn::Player(_) => {
                            let acts = game.legal_actions(&s);
                            if acts.len() > 1 {
                                // The subgame for this live state must key the
                                // hero's infoset identically — for *every* seat,
                                // not just the actor.
                                let round = agent.round_subgame(&game, &s);
                                for q in 0..players as usize {
                                    assert_eq!(
                                        game.infoset_key(&s, q),
                                        round.infoset_key(&s, q),
                                        "players={players} seat={q} qty={}",
                                        s.qty
                                    );
                                }
                            }
                            let a = acts[rng.below(acts.len())];
                            game.apply(&mut s, a);
                        }
                    }
                }
            }
        }
    }

    /// The agent always returns a legal action index, including at singleton
    /// infosets and after a real solve.
    #[test]
    fn act_returns_a_legal_index() {
        let agent = OnlineSolveAgent::with_config(
            || DiceShareValue,
            OnlineSolveConfig {
                iters: 200,
                max_iters: 200,
                ..OnlineSolveConfig::default()
            },
        );
        let game = LiarsDice::new(3, 2, 6);
        let mut rng = Rng::new(7);
        for _ in 0..20 {
            let mut s = game.initial_state();
            while !game.is_terminal(&s) {
                match game.turn(&s) {
                    Turn::Chance => {
                        let a = game.sample_chance_action(&s, &mut rng);
                        game.apply(&mut s, a);
                    }
                    Turn::Player(p) => {
                        let acts = game.legal_actions(&s);
                        let i = agent.act(&game, &s, p, &mut rng);
                        assert!(i < acts.len(), "index {i} out of {} actions", acts.len());
                        game.apply(&mut s, acts[i]);
                    }
                }
            }
        }
    }

    /// On the thin 2p1d6f probe the agent's call frequency must land near the
    /// exact equilibrium (~0.4) and far from the raw-net over-call (~0.98) — a
    /// fast, self-contained version of the correctness section.
    #[test]
    fn thin_probe_call_frequency_is_near_equilibrium() {
        fn hand_with(face: u8) -> [u8; MAX_FACES] {
            let mut h = [0u8; MAX_FACES];
            h[face as usize - 1] = 1;
            h
        }
        let agent = OnlineSolveAgent::new(|| DiceShareValue);
        let game = LiarsDice::new(2, 1, 6);
        // Build the thin probe: seat 0 holds a 1 and opens 1x5, seat 1 holds a 2.
        let round = RoundSubgame::new(
            2,
            1,
            6,
            [1, 1, 0, 0, 0, 0, 0, 0],
            0,
            false,
            4,
            DiceShareValue,
        );
        let mut s = round.initial_state();
        let hands = [hand_with(1), hand_with(2)];
        let mut rolled = 0;
        while let Turn::Chance = round.turn(&s) {
            round.apply(&mut s, Action::Roll(hands[rolled]));
            rolled += 1;
        }
        round.apply(&mut s, Action::Open(1, 5));

        // The averaged policy should land near the ~0.42 equilibrium — far from
        // the raw net's ~0.98 over-call. A single batch still carries MCCFR
        // sampling noise, so average a few batches to pin the *mean* call
        // frequency the agent plays in expectation.
        let acts = game.legal_actions(&s);
        let ci = acts
            .iter()
            .position(|a| matches!(a, Action::CallLiar))
            .unwrap();
        let batches = 4u64;
        let call: f64 = (0..batches)
            .map(|b| agent.solve_policy(&game, &s, 1, 0xBEEF + b * 7919)[ci])
            .sum::<f64>()
            / batches as f64;
        assert!(
            (0.3..=0.55).contains(&call),
            "thin-bid mean call freq {call} should be near the ~0.42 equilibrium, not the net's ~0.98"
        );
    }

    /// The size-based budget keeps small rounds at the full requested
    /// `iters × restarts`, scales the total down for large rounds, and never lets
    /// a single solve drop below the opening-convergence floor — the speed
    /// guarantees the strength/speed tables depend on.
    #[test]
    fn budget_scales_down_for_large_rounds_with_a_floor() {
        let agent = OnlineSolveAgent::new(|| DiceShareValue);
        // Small heads-up round: full budget, full restarts.
        let (it_small, rs_small) = agent.budget(3);
        assert_eq!((it_small, rs_small), (8_000, 3));
        // Large round: total iterations strictly fewer, restarts may shrink, but
        // each solve still runs >= the floor.
        let (it_big, rs_big) = agent.budget(47);
        assert!(
            it_big * rs_big as u64 <= it_small * rs_small as u64,
            "large-round total ({it_big}x{rs_big}) must not exceed small ({it_small}x{rs_small})"
        );
        assert!(
            it_big >= 1_000,
            "each solve keeps the opening-convergence floor"
        );
        assert!(rs_big >= 1, "always at least one solve");
        // The size scaling is monotone in total dice (more dice -> not more work).
        let (it_mid, rs_mid) = agent.budget(15);
        assert!(it_mid * rs_mid as u64 <= it_small * rs_small as u64);
        assert!(it_big * rs_big as u64 <= it_mid * rs_mid as u64);
    }
}
