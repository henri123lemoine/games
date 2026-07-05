//! The high-throughput self-play simulator: `N` parallel envs, each playing full
//! Stratego games (serialized deployment then the move phase), terminating and
//! auto-resetting independently so the batch stays desynced for phase coverage.
//!
//! ## collect / commit core
//! One decision step is two halves with the neural-net forward in between:
//! 1. [`Simulator::collect`] resets any already-terminal env (so the batch is all
//!    live decisions), then snapshots and encodes every env's decision state into
//!    a [`Collected`] batch — the per-env obs, legal set, phase, and player.
//! 2. The caller runs the evaluator/net **once** over that batch (the single
//!    point a real GPU forward runs — one-GPU-thread discipline).
//! 3. [`Simulator::commit`] takes the per-env [`Evaluation`]s, softmax-samples a
//!    legal action per env, applies it, records the transition into the
//!    [`ReplayBuffer`], and auto-resets any env that just terminated.
//!
//! [`Simulator::step`] is the in-process wrapper: `collect` → one
//! `Evaluator::evaluate_batch` → `commit`. The Python bridge (`ml/stratego-py`)
//! calls `collect` and `commit` directly with an MLX net in between, reusing the
//! same core — there is no second code path. The collector owns the per-env
//! arenas; the encode and the sample/apply work each run over a `rayon`
//! par-iter, with the evaluator call the one serial batched boundary.

use game_core::rand::pick_weighted;
use game_core::{Game, Rng};
use rayon::prelude::*;

use crate::board::PieceType;
use crate::buffer::{ReplayBuffer, Snapshot, Transition};
use crate::encode::{EncoderConfig, encode_tokens};
use crate::evaluator::{Decision, Evaluation, Evaluator, Phase};
use crate::game::{Move, State, Stratego};

/// Distinct per-env seeds are derived from a counter advanced by this odd
/// stride, matching the slither collector's deterministic seeding.
const SEED_STRIDE: u64 = 0x1000193;

/// Per-env arena: the live game state and its RNG stream.
#[derive(Debug, Clone)]
pub struct Arena {
    pub state: State,
    pub rng: Rng,
    /// Half-moves played in the *current* game (move phase), for desync stats.
    pub plies: u32,
}

/// The self-play collector: owns the arenas and the buffer, driven by [`step`].
pub struct Simulator {
    arenas: Vec<Arena>,
    cfg: EncoderConfig,
    rng: Rng,
    seed_ctr: u64,
    /// Move cap before an env is force-reset (defensive; the rules' own
    /// termination should fire first via the k-move rule).
    move_cap: u32,
    /// No-attack draw clock passed to [`rules::is_terminal_with_clock`] /
    /// [`rules::reward_pl0_with_clock`] for this simulator's own termination
    /// checks (reference-parity default 100; a training curriculum anneals it
    /// via [`Simulator::set_attack_clock`]).
    attack_clock: u32,
}

/// Aggregate counters from a [`Simulator::run`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunStats {
    /// Total decision steps applied (deployment placements + moves).
    pub decision_steps: u64,
    /// Games that reached a terminal move-phase position and reset.
    pub games_completed: u64,
    /// Sum of player-0 terminal rewards over completed games (should hover near
    /// 0 for a symmetric uniform policy).
    pub reward_pl0_sum: f64,
    /// Move-phase decisions (the denominator for an attack-rate measurement;
    /// `decision_steps` also counts deployment placements, which can never be
    /// attacks).
    pub move_decisions: u64,
    /// Of those, how many landed on an occupied (enemy) cell — the direct
    /// signature of the passivity trap: a from-scratch policy's attack rate
    /// collapsing toward 0 is what starves the value head under the no-attack
    /// draw clock (see the 2026-07-04 measurement this counter exists for).
    pub move_attacks: u64,
}

impl RunStats {
    fn add(&mut self, other: RunStats) {
        self.decision_steps += other.decision_steps;
        self.games_completed += other.games_completed;
        self.reward_pl0_sum += other.reward_pl0_sum;
        self.move_decisions += other.move_decisions;
        self.move_attacks += other.move_attacks;
    }
}

/// The result of one [`Simulator::commit`]: aggregate stats plus the per-env
/// completion signal the caller's loop keys on. `completed[env]` is
/// `Some(player0_reward)` exactly when this step finished `env`'s game (and the
/// env was auto-reset), `None` otherwise — the env's terminal flag and reward,
/// in env order.
#[derive(Debug, Clone, Default)]
pub struct CommitResult {
    pub stats: RunStats,
    pub completed: Vec<Option<f64>>,
    /// True when an env completed only because the defensive `move_cap` fired.
    /// These boundaries break replay bootstrapping, but do not provide a real
    /// setup-game outcome.
    pub capped: Vec<bool>,
}

/// The decision inputs encoded from one env, owning the buffers the batched
/// [`Decision`] slices borrow.
pub struct EnvDecision {
    pub obs: Vec<f32>,
    pub legal: Vec<u16>,
    pub phase: Phase,
    pub player: usize,
}

/// A whole batch of per-env decisions, produced by [`Simulator::collect`] and
/// consumed by [`Simulator::commit`]. One [`EnvDecision`] per env, in env order;
/// the evaluator runs once over [`Collected::requests`].
pub struct Collected {
    decisions: Vec<EnvDecision>,
}

impl Collected {
    /// The per-env decisions, in env order.
    pub fn decisions(&self) -> &[EnvDecision] {
        &self.decisions
    }

    /// Borrowed [`Decision`] views over every env, parallel to the arenas — the
    /// argument to one [`Evaluator::evaluate_batch`] call.
    pub fn requests(&self) -> Vec<Decision<'_>> {
        self.decisions
            .iter()
            .map(|d| Decision {
                phase: d.phase,
                obs: &d.obs,
                legal: &d.legal,
                player: d.player,
            })
            .collect()
    }
}

/// Everything the serial commit pass needs for one env after sampling and
/// applying on a cloned RNG: the recorded transition, the resulting state, the
/// advanced RNG, ply count, and the terminal stats this env produced.
struct EnvOutcome {
    transition: Transition,
    next_state: State,
    rng: Rng,
    plies: u32,
    /// `Some(player0_reward)` when this step completed a game and the env must
    /// reset; the reset itself happens serially so seeds stay deterministic.
    completed: Option<f64>,
    capped: bool,
    was_move: bool,
    was_attack: bool,
}

impl Simulator {
    /// `num_envs` arenas, seeded deterministically from `seed`. `move_cap` is the
    /// defensive per-game ply cap. Each env is desynced by a different number of
    /// random legal steps so the batch spans deployment and varied move depths.
    pub fn new(num_envs: usize, cfg: EncoderConfig, seed: u64, move_cap: u32) -> Simulator {
        let game = Stratego;
        let mut sim = Simulator {
            arenas: Vec::with_capacity(num_envs),
            cfg,
            rng: Rng::new(seed),
            seed_ctr: seed.wrapping_mul(0x9e3779b9).wrapping_add(1),
            move_cap,
            attack_clock: crate::rules::MAX_NUM_MOVES_BETWEEN_ATTACKS,
        };

        let period = num_envs.max(1);
        for i in 0..num_envs {
            let env_seed = sim.next_seed();
            let mut arena = Arena {
                state: game.initial_state(),
                rng: Rng::new(env_seed),
                plies: 0,
            };
            desync(&game, &mut arena, i % period);
            sim.arenas.push(arena);
        }
        sim
    }

    pub fn num_envs(&self) -> usize {
        self.arenas.len()
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// Current no-attack draw clock this simulator's own termination checks use
    /// (see `attack_clock` on [`Simulator`]).
    pub fn attack_clock(&self) -> u32 {
        self.attack_clock
    }

    /// Sets the no-attack draw clock for subsequent [`commit`](Simulator::commit)
    /// calls — a curriculum anneals this across training without reconstructing
    /// the simulator (which would drop every live arena). Takes effect from the
    /// next decision onward; in-flight games are unaffected mid-decision. Also
    /// updates the encoder's `max_num_moves_between_attacks` denominator (the
    /// obs's own "moves since last attack" plane) so what the net sees about
    /// the clock never drifts from the clock actually governing termination.
    pub fn set_attack_clock(&mut self, attack_clock: u32) {
        self.attack_clock = attack_clock;
        self.cfg.max_num_moves_between_attacks = attack_clock;
    }

    /// A clone of `env`'s current move-phase board and the player to act, or
    /// `None` if the env is mid-deployment (test-time search runs in the move
    /// phase only — the deploy phase is the setup net's domain). The clone is the
    /// search root: the caller determinizes it per belief sample and rolls it out
    /// via [`RolloutBatch`](crate::search::RolloutBatch).
    pub fn move_root(&self, env: usize) -> Option<(crate::board::Board, usize)> {
        match &self.arenas[env].state {
            State::Play { board, to_play, .. } => Some(((**board).clone(), *to_play)),
            State::Deploy { .. } => None,
        }
    }

    /// A human-readable render of `env`'s current state from `viewer`'s
    /// perspective (deploy prompt or board), reusing the exact [`GameUi::render`]
    /// every other front end (the `lab` CLI) plays through — including its
    /// hidden-info guarantee that an opponent's unrevealed pieces never leak.
    pub fn render(&self, env: usize, viewer: usize) -> String {
        use game_core::GameUi;
        Stratego.render(&self.arenas[env].state, viewer)
    }

    /// A fresh per-env seed, mixing the top-level RNG stream with the advancing
    /// counter so reseeds stay deterministic given the constructor `seed` and
    /// never collide with an in-flight env's stream.
    fn next_seed(&mut self) -> u64 {
        let s = self.seed_ctr ^ self.rng.next_u64();
        self.seed_ctr = self.seed_ctr.wrapping_add(SEED_STRIDE);
        s
    }

    /// Resets `env` to a fresh game with the next deterministic seed.
    fn reset_env(&mut self, env: usize) {
        let game = Stratego;
        let env_seed = self.next_seed();
        self.arenas[env].state = game.initial_state();
        self.arenas[env].rng = Rng::new(env_seed);
        self.arenas[env].plies = 0;
    }

    /// First half of a decision step: reset any already-terminal env so the batch
    /// is all live decisions, then batch-encode every env into a [`Collected`]
    /// batch for one evaluator/net forward. Resets are serial to keep seeds
    /// ordered. Pair with [`commit`](Simulator::commit), passing the evaluations
    /// in env order.
    pub fn collect(&mut self) -> Collected {
        // Uses the clocked check (not `Game::is_terminal`'s hardcoded constant)
        // so a reset-worthy state is judged by the same `attack_clock` that
        // `commit`/`sample_apply` used to produce it — otherwise an annealed
        // clock other than the reference-parity 100 would let this reset check
        // disagree with the one that actually ended the game.
        for env in 0..self.arenas.len() {
            let terminal = match &self.arenas[env].state {
                State::Play {
                    board,
                    to_play,
                    flag_captured,
                } => crate::rules::is_terminal_with_clock(
                    board,
                    *to_play,
                    *flag_captured,
                    self.attack_clock,
                ),
                State::Deploy { .. } => false,
            };
            if terminal {
                self.reset_env(env);
            }
        }

        let cfg = &self.cfg;
        let decisions: Vec<EnvDecision> = self
            .arenas
            .par_iter()
            .map(|arena| encode_decision(arena, cfg))
            .collect();
        Collected { decisions }
    }

    /// Second half of a decision step: take the per-env [`Evaluation`]s (in env
    /// order, parallel to the [`Collected`] batch), softmax-sample a legal action
    /// per env, apply it, record the transition into `buffer`, and auto-reset any
    /// env that just terminated. Returns the per-call stats delta.
    pub fn commit(
        &mut self,
        collected: &Collected,
        evals: &[Evaluation],
        buffer: &mut ReplayBuffer,
    ) -> CommitResult {
        let game = Stratego;
        assert_eq!(evals.len(), self.arenas.len(), "one evaluation per env");

        let move_cap = self.move_cap;
        let attack_clock = self.attack_clock;
        let outcomes: Vec<EnvOutcome> = self
            .arenas
            .par_iter()
            .zip(collected.decisions.par_iter())
            .zip(evals.par_iter())
            .map(|((arena, decision), evaluation)| {
                sample_apply(&game, arena, decision, evaluation, move_cap, attack_clock)
            })
            .collect();

        let mut result = CommitResult {
            completed: vec![None; self.arenas.len()],
            capped: vec![false; self.arenas.len()],
            ..CommitResult::default()
        };
        for (env, outcome) in outcomes.into_iter().enumerate() {
            result.stats.decision_steps += 1;
            if outcome.was_move {
                result.stats.move_decisions += 1;
                if outcome.was_attack {
                    result.stats.move_attacks += 1;
                }
            }
            buffer.record(env, outcome.transition);
            if let Some(reward_pl0) = outcome.completed {
                result.stats.games_completed += 1;
                result.stats.reward_pl0_sum += reward_pl0;
                result.completed[env] = Some(reward_pl0);
                result.capped[env] = outcome.capped;
                self.reset_env(env);
            } else {
                self.arenas[env].state = outcome.next_state;
                self.arenas[env].rng = outcome.rng;
                self.arenas[env].plies = outcome.plies;
            }
        }
        result
    }

    /// Advances every live env by one decision: [`collect`](Simulator::collect),
    /// one batched evaluator call, then [`commit`](Simulator::commit). The
    /// in-process wrapper around the collect/commit core. Returns the per-call
    /// stats delta.
    pub fn step(&mut self, eval: &dyn Evaluator, buffer: &mut ReplayBuffer) -> RunStats {
        let collected = self.collect();
        let evals = eval.evaluate_batch(&collected.requests());
        self.commit(&collected, &evals, buffer).stats
    }

    /// Runs `steps` decision steps, accumulating stats.
    pub fn run(
        &mut self,
        eval: &dyn Evaluator,
        buffer: &mut ReplayBuffer,
        steps: usize,
    ) -> RunStats {
        let mut total = RunStats::default();
        for _ in 0..steps {
            total.add(self.step(eval, buffer));
        }
        total
    }
}

/// Advances an arena by `steps` uniform-random legal decisions using its own
/// RNG, resetting if it terminates (rare this early). No buffer recording.
fn desync(game: &Stratego, arena: &mut Arena, steps: usize) {
    for _ in 0..steps {
        if game.is_terminal(&arena.state) {
            arena.state = game.initial_state();
            arena.plies = 0;
        }
        let actions = game.legal_actions(&arena.state);
        if actions.is_empty() {
            arena.state = game.initial_state();
            arena.plies = 0;
            continue;
        }
        let idx = arena.rng.below(actions.len());
        let is_move = matches!(arena.state, State::Play { .. });
        game.apply(&mut arena.state, actions[idx]);
        if is_move {
            arena.plies += 1;
        }
    }
}

/// Builds the per-env decision inputs (obs, legal set, phase, player) from the
/// arena's current live state.
fn encode_decision(arena: &Arena, cfg: &EncoderConfig) -> EnvDecision {
    match &arena.state {
        State::Play { board, to_play, .. } => {
            let mask = crate::rules::legal_mask(board, *to_play);
            let legal: Vec<u16> = (0..mask.len())
                .filter(|&i| mask[i])
                .map(|i| i as u16)
                .collect();
            let obs = encode_tokens(board, *to_play, cfg);
            EnvDecision {
                obs,
                legal,
                phase: Phase::Move,
                player: *to_play,
            }
        }
        State::Deploy { current, .. } => {
            let legal: Vec<u16> = current
                .legal_types()
                .into_iter()
                .map(|t| t as u16)
                .collect();
            EnvDecision {
                obs: crate::encode::deploy_obs(current),
                legal,
                phase: Phase::Deploy,
                player: current.player,
            }
        }
    }
}

/// Softmax-samples a legal option from `logits` using a cloned RNG, applies the
/// chosen action to a cloned state, and produces the recorded transition plus
/// the resulting state/RNG. All RNG use is on the clone, so the result is
/// independent of thread scheduling.
fn sample_apply(
    game: &Stratego,
    arena: &Arena,
    decision: &EnvDecision,
    evaluation: &Evaluation,
    move_cap: u32,
    attack_clock: u32,
) -> EnvOutcome {
    let log_probs = log_softmax(&evaluation.logits);
    let weights: Vec<f64> = log_probs.iter().map(|&lp| f64::from(lp).exp()).collect();

    let mut rng = arena.rng.clone();
    let chosen = pick_weighted(weights.iter().copied(), &mut rng);
    let action = decision.legal[chosen];

    let snapshot = snapshot_of(&arena.state);
    let num_moves = match &arena.state {
        State::Play { board, .. } => board.num_moves,
        State::Deploy { .. } => 0,
    };

    let game_action = match decision.phase {
        Phase::Deploy => Move::Place(PieceType::from_u8(action as u8)),
        Phase::Move => Move::Step(crate::action::Action(action)),
    };

    let was_move = decision.phase == Phase::Move;
    // Reads the PRE-move board: an attack is a move onto a cell occupied by an
    // enemy piece, exactly `rules::apply`'s own `dst_code != NO_ATTACK_DST_CODE`
    // (the Game trait's `apply` discards that `Applied` detail, so this
    // re-derives it independently rather than re-plumbing the whole return
    // value through the trait boundary).
    let was_attack = was_move
        && match &arena.state {
            State::Play { board, .. } => {
                let Move::Step(step_action) = game_action else {
                    unreachable!("was_move implies Move::Step")
                };
                let (_from_abs, to_abs) = step_action.to_abs(decision.player);
                board.pieces[to_abs].kind != PieceType::Empty
            }
            State::Deploy { .. } => false,
        };
    let mut next_state = arena.state.clone();
    game.apply(&mut next_state, game_action);
    let plies = arena.plies + u32::from(was_move);

    // Bypasses `Game::is_terminal`/`Game::returns` (which hardcode the
    // reference-parity clock) so this simulator's own `attack_clock` — possibly
    // annealed by a curriculum — governs its termination, matching the same
    // `rules::is_terminal`/`reward_pl0` logic those trait methods otherwise call.
    let (rules_terminal, reward_pl0_at_next) = match &next_state {
        State::Play {
            board,
            to_play,
            flag_captured,
        } => (
            crate::rules::is_terminal_with_clock(board, *to_play, *flag_captured, attack_clock),
            crate::rules::reward_pl0_with_clock(board, *to_play, *flag_captured, attack_clock),
        ),
        State::Deploy { .. } => (false, 0.0),
    };
    let capped = was_move && plies >= move_cap;

    // A capped game is force-reset with zero reward. It still has to break the
    // replay segment so lambda returns never bootstrap across the fresh game.
    let is_terminating_action = rules_terminal || capped;
    let returns_for = |player: usize| -> f64 {
        if player == 0 {
            reward_pl0_at_next
        } else {
            -reward_pl0_at_next
        }
    };
    let terminal_reward = if rules_terminal {
        returns_for(decision.player) as f32
    } else {
        0.0
    };
    let completed = if rules_terminal {
        Some(returns_for(0))
    } else if capped {
        Some(0.0)
    } else {
        None
    };

    let transition = Transition {
        snapshot,
        phase: decision.phase,
        player: decision.player,
        num_moves,
        action,
        legal: decision.legal.clone(),
        old_log_probs: log_probs,
        chosen,
        value: evaluation.value,
        value_probs: evaluation.value_probs,
        is_terminated_position: false,
        is_terminating_action,
        terminal_reward,
        truncated: capped && !rules_terminal,
        target_value: None,
        target_value_probs: None,
    };

    EnvOutcome {
        transition,
        next_state,
        rng,
        plies,
        completed,
        capped,
        was_move,
        was_attack,
    }
}

/// The snapshot needed to re-encode this state's observation on demand.
fn snapshot_of(state: &State) -> Snapshot {
    match state {
        State::Play { board, to_play, .. } => Snapshot::Play {
            board: Box::new((**board).clone()),
            to_play: *to_play,
        },
        State::Deploy { red, current } => Snapshot::Deploy {
            red: red.clone(),
            current: current.clone(),
        },
    }
}

/// Numerically-stable log-softmax over the legal-option logits.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = max + sum_exp.ln();
    logits.iter().map(|&l| l - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ReplayBuffer;
    use crate::evaluator::{Evaluation, UniformEvaluator};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An evaluator that wraps [`UniformEvaluator`] and counts its calls and the
    /// batch length it last saw, to assert the one-GPU-thread seam.
    struct CountingEvaluator {
        inner: UniformEvaluator,
        calls: AtomicUsize,
        last_batch: AtomicUsize,
    }

    impl CountingEvaluator {
        fn new() -> Self {
            CountingEvaluator {
                inner: UniformEvaluator,
                calls: AtomicUsize::new(0),
                last_batch: AtomicUsize::new(0),
            }
        }
    }

    impl Evaluator for CountingEvaluator {
        fn evaluate_batch(&self, batch: &[Decision]) -> Vec<Evaluation> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.last_batch.store(batch.len(), Ordering::SeqCst);
            self.inner.evaluate_batch(batch)
        }
    }

    fn buffer(num_envs: usize) -> ReplayBuffer {
        ReplayBuffer::new(num_envs, 256, EncoderConfig::default())
    }

    #[test]
    fn games_terminate_and_record_within_cap() {
        let num_envs = 8;
        let mut sim = Simulator::new(num_envs, EncoderConfig::default(), 12345, 4000);
        let mut buf = buffer(num_envs);
        let stats = sim.run(&UniformEvaluator, &mut buf, 2000);

        assert!(
            stats.games_completed > 0,
            "no games completed in 2000 steps over {num_envs} envs"
        );

        // Every completed game leaves a recorded terminating transition in some
        // env's ring; at least as many such transitions exist as games finished
        // minus the per-env trajectories the ring may have overwritten.
        let mut terminating = 0u64;
        for env in 0..num_envs {
            for slot in 0..buf.capacity() {
                if let Some(t) = buf.get(env, slot)
                    && t.is_terminating_action
                {
                    terminating += 1;
                }
            }
        }
        assert!(
            terminating > 0,
            "completed {} games but no terminating transition was recorded",
            stats.games_completed
        );
    }

    #[test]
    fn attack_telemetry_counts_only_move_decisions_that_land_on_an_occupied_cell() {
        let num_envs = 8;
        let mut sim = Simulator::new(num_envs, EncoderConfig::default(), 4242, 4000);
        let mut buf = buffer(num_envs);
        let stats = sim.run(&UniformEvaluator, &mut buf, 2000);

        assert!(stats.move_decisions > 0, "no move-phase decisions recorded");
        assert!(
            stats.move_attacks <= stats.move_decisions,
            "move_attacks {} exceeds move_decisions {}",
            stats.move_attacks,
            stats.move_decisions
        );
        // Uniform-random legal play attacks fairly often (both sides moving
        // freely toward each other) — some nonzero count over 2000 steps is the
        // sanity floor; the real discriminating rate comparison (HeuristicBot's
        // measured ~24.5/100 plies vs a passive RL net's ~0.6-1.7/100) lives in
        // train.py's `attacks_per_100_plies` metric, not this unit test.
        assert!(
            stats.move_attacks > 0,
            "expected at least one attack in {} move decisions",
            stats.move_decisions
        );
    }

    #[test]
    fn forced_cap_breaks_replay_without_real_reward() {
        let mut sim = Simulator::new(1, EncoderConfig::default(), 7, 1);
        let mut buf = buffer(1);

        for _ in 0..200 {
            let collected = sim.collect();
            let evals = UniformEvaluator.evaluate_batch(&collected.requests());
            let result = sim.commit(&collected, &evals, &mut buf);
            if result.completed[0].is_some() {
                assert!(
                    result.capped[0],
                    "first completion should be the one-ply cap"
                );
                let t = buf
                    .get(0, buf.head(0) - 1)
                    .expect("cap transition recorded");
                assert!(
                    t.is_terminating_action,
                    "cap must break replay bootstrapping"
                );
                assert_eq!(t.terminal_reward, 0.0, "cap carries no game reward");
                assert!(
                    t.truncated,
                    "cap is a truncation (value-bootstrapped target)"
                );
                return;
            }
        }
        panic!("cap did not fire");
    }

    #[test]
    fn same_seed_is_deterministic() {
        let num_envs = 6;
        let run = |seed: u64| {
            let mut sim = Simulator::new(num_envs, EncoderConfig::default(), seed, 4000);
            let mut buf = buffer(num_envs);
            let stats = sim.run(&UniformEvaluator, &mut buf, 400);
            let actions: Vec<u16> = (0..buf.capacity())
                .filter_map(|slot| buf.get(0, slot).map(|t| t.action))
                .collect();
            (stats, actions)
        };

        let (stats_a, actions_a) = run(777);
        let (stats_b, actions_b) = run(777);
        assert_eq!(stats_a, stats_b, "same seed must give identical stats");
        assert_eq!(
            actions_a, actions_b,
            "same seed must give identical env-0 action sequence"
        );

        let (stats_c, actions_c) = run(778);
        assert!(
            stats_c != stats_b || actions_c != actions_b,
            "different seeds must diverge"
        );
    }

    #[test]
    fn terminal_rewards_are_zero_sum() {
        let num_envs = 8;
        let mut sim = Simulator::new(num_envs, EncoderConfig::default(), 555, 4000);
        let mut buf = buffer(num_envs);
        let stats = sim.run(&UniformEvaluator, &mut buf, 2000);

        for env in 0..num_envs {
            for slot in 0..buf.capacity() {
                if let Some(t) = buf.get(env, slot)
                    && t.is_terminating_action
                {
                    assert!(
                        t.terminal_reward == 1.0
                            || t.terminal_reward == -1.0
                            || t.terminal_reward == 0.0,
                        "terminal reward {} is not in a zero-sum range",
                        t.terminal_reward
                    );
                }
            }
        }

        let bound = stats.games_completed as f64;
        assert!(
            stats.reward_pl0_sum.abs() <= bound,
            "player-0 reward sum {} exceeds completed-game bound {bound}",
            stats.reward_pl0_sum
        );
    }

    /// A terminal Stratego position is zero-sum between the two players.
    #[test]
    fn returns_are_zero_sum_at_terminals() {
        let game = Stratego;
        let mut sim = Simulator::new(4, EncoderConfig::default(), 99, 4000);
        let mut buf = buffer(4);
        sim.run(&UniformEvaluator, &mut buf, 1500);

        // Drive a few envs to terminal directly and check the returns identity.
        for env in 0..4 {
            let mut state = sim.arenas[env].state.clone();
            let mut rng = sim.arenas[env].rng.clone();
            for _ in 0..6000 {
                if game.is_terminal(&state) {
                    assert_eq!(
                        game.returns(&state, 0),
                        -game.returns(&state, 1),
                        "terminal returns are not zero-sum"
                    );
                    break;
                }
                let actions = game.legal_actions(&state);
                if actions.is_empty() {
                    break;
                }
                let idx = rng.below(actions.len());
                game.apply(&mut state, actions[idx]);
            }
        }
    }

    #[test]
    fn desync_spreads_phase_and_depth() {
        let num_envs = 16;
        let sim = Simulator::new(num_envs, EncoderConfig::default(), 2024, 4000);

        let mut deploy_progress = std::collections::HashSet::new();
        let mut play_depths = std::collections::HashSet::new();
        let mut any_play = false;
        let mut any_deploy = false;
        for arena in &sim.arenas {
            match &arena.state {
                State::Deploy { current, .. } => {
                    any_deploy = true;
                    deploy_progress.insert(current.placed.len());
                }
                State::Play { board, .. } => {
                    any_play = true;
                    play_depths.insert(board.num_moves);
                }
            }
        }

        let varied = (any_play && any_deploy) || deploy_progress.len() > 1 || play_depths.len() > 1;
        assert!(
            varied,
            "desync left all envs in the same phase/depth: \
             deploy slots {deploy_progress:?}, play depths {play_depths:?}"
        );
    }

    #[test]
    fn evaluator_called_once_per_step_with_full_batch() {
        let num_envs = 8;
        let mut sim = Simulator::new(num_envs, EncoderConfig::default(), 31, 4000);
        let mut buf = buffer(num_envs);
        let counter = CountingEvaluator::new();

        sim.step(&counter, &mut buf);
        assert_eq!(
            counter.calls.load(Ordering::SeqCst),
            1,
            "step must call evaluate_batch exactly once"
        );
        assert_eq!(
            counter.last_batch.load(Ordering::SeqCst),
            num_envs,
            "the batch must cover every env"
        );

        sim.run(&counter, &mut buf, 5);
        assert_eq!(
            counter.calls.load(Ordering::SeqCst),
            6,
            "each step contributes exactly one evaluate_batch call"
        );
    }

    /// The collect/commit core composed by hand reproduces `step` exactly: the
    /// Python bridge drives the sim this way, so the split must be transparent.
    #[test]
    fn collect_commit_matches_step() {
        let num_envs = 6;
        let eval = UniformEvaluator;

        let mut sim_a = Simulator::new(num_envs, EncoderConfig::default(), 4242, 4000);
        let mut buf_a = buffer(num_envs);
        let mut sim_b = Simulator::new(num_envs, EncoderConfig::default(), 4242, 4000);
        let mut buf_b = buffer(num_envs);

        for _ in 0..300 {
            let stats_a = sim_a.step(&eval, &mut buf_a);

            let collected = sim_b.collect();
            let evals = eval.evaluate_batch(&collected.requests());
            let result = sim_b.commit(&collected, &evals, &mut buf_b);

            assert_eq!(
                stats_a, result.stats,
                "collect/commit stats diverged from step"
            );
            let completed = result.completed.iter().filter(|c| c.is_some()).count() as u64;
            assert_eq!(
                completed, result.stats.games_completed,
                "per-env completed flags must match the aggregate count"
            );
        }

        for env in 0..num_envs {
            let actions_a: Vec<u16> = (0..buf_a.capacity())
                .filter_map(|slot| buf_a.get(env, slot).map(|t| t.action))
                .collect();
            let actions_b: Vec<u16> = (0..buf_b.capacity())
                .filter_map(|slot| buf_b.get(env, slot).map(|t| t.action))
                .collect();
            assert_eq!(
                actions_a, actions_b,
                "collect/commit env-{env} actions diverged from step"
            );
        }
    }
}
