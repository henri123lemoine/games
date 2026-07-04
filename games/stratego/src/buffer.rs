//! The replay buffer — a circular ring of cheap-clone board snapshots that *is*
//! the trajectory store (§1.6: "the replay buffer IS the circular board
//! buffer"). Per transition we keep exactly what the move-RL loop consumes
//! (mirroring the reference `core/buffer.py`): the snapshot needed to re-encode
//! the observation on demand, the chosen action, the old log-probs over the
//! legal set, the legal mask, the acting player, the phase, and the per-step
//! reward/terminal flags from which λ-returns, GAE advantages, and the two-ply
//! value targets are computed by [`ReplayBuffer::process_data`].
//!
//! ## Store-board, re-encode on demand
//! Following the reference (`buffer.py:sample` calls `env.infostate_tensor(step)`
//! to *recompute* the 355-channel tensor rather than storing it), we store the
//! [`Board`] snapshot — not the ~`92*643` float observation — and re-encode via
//! [`encode_tokens`](crate::encode::encode_tokens) when a transition is read
//! back. The board carries its own `action_history`, so the 32-move history
//! window the encoder needs reconstructs deterministically from the snapshot
//! with no external ring walk. [`Snapshot`] holds the board (move phase) or the
//! partial deployment (setup phase).
//!
//! ## Solipsistic terminal handling
//! Each player is trained as if its own action ended the game (`buffer.py`
//! `add_post_act`): when player P's move objectively terminates, the *previous*
//! player's transition is rewritten to be terminating with the sign-flipped
//! reward. [`ReplayBuffer::record`] performs this fix-up across the per-env
//! linked transitions.

use crate::arrangement::DeploymentState;
use crate::board::Board;
use crate::encode::EncoderConfig;
use crate::evaluator::{Phase, two_hot};

/// The minimal state needed to re-encode a transition's observation on demand.
#[derive(Debug, Clone)]
pub enum Snapshot {
    /// A move-phase board snapshot (carries its own action history).
    Play { board: Box<Board>, to_play: usize },
    /// A deployment-phase partial placement (red's finished arrangement, if any,
    /// plus the in-progress deployment).
    Deploy {
        red: Option<crate::arrangement::Arrangement>,
        current: DeploymentState,
    },
}

/// One recorded decision transition. Everything the trainer needs per step; the
/// observation itself is re-derived from `snapshot` on read (see module docs).
#[derive(Debug, Clone)]
pub struct Transition {
    /// The state snapshot, for on-demand re-encoding of obs / legal mask.
    pub snapshot: Snapshot,
    pub phase: Phase,
    /// Acting player (0 = red, 1 = blue).
    pub player: usize,
    /// `num_moves` at this position (move phase) — the temporal index the net
    /// also receives; `0` during deployment.
    pub num_moves: u32,
    /// Chosen action: a 1800-space index (move) or `PieceType as u16` (deploy).
    pub action: u16,
    /// The legal option indices at this state (parallel to `old_log_probs`).
    pub legal: Vec<u16>,
    /// Old log-probs over the legal set, from the evaluator at collection time
    /// (parallel to `legal`). The PPO ratio's denominator.
    pub old_log_probs: Vec<f32>,
    /// Index into `legal` of the chosen `action`.
    pub chosen: usize,
    /// Scalar value the evaluator gave this position (acting POV): `value_probs
    /// @ VALUE_CATEGORIES`.
    pub value: f32,
    /// The evaluator's categorical distribution over `VALUE_CATEGORIES` for this
    /// position (acting POV) — the move value head's raw softmax(W/L/D). Feeds
    /// the categorical λ-return in [`ReplayBuffer::process_data`]; `value` is
    /// only its aggregate.
    pub value_probs: [f32; 3],
    /// True if this position is itself terminal (no action was taken from it in
    /// the live game — a dummy reset transition).
    pub is_terminated_position: bool,
    /// True if the action taken here ends the replay segment (rules-terminal,
    /// forced cap reset, or solipsistic fix-up).
    pub is_terminating_action: bool,
    /// Terminal reward to the acting player (`is_terminating_action * reward`),
    /// 0 otherwise.
    pub terminal_reward: f32,
    /// True when this terminating action is a `move_cap` TRUNCATION (not a real
    /// rules-terminal). Its value target resolves to its own `value` (delta 0)
    /// instead of `terminal_reward`, so a timeout neither poisons the value head
    /// toward 0 nor rewards stalling, while still breaking the replay segment.
    pub truncated: bool,
    /// Two-ply-later value target for this position, filled by [`record`] once
    /// the same player's next position is known, or the one-hot terminal reward.
    /// `None` until resolved.
    pub target_value: Option<f32>,
    /// Categorical counterpart of `target_value`: the two-ply-later position's
    /// `value_probs`, or the terminal reward's one-hot, matching the reference
    /// `buffer.py`'s `target_values` (`softmax(values[next])`, not a two-hot
    /// projection of the scalar `target_value`). `None` until resolved.
    pub target_value_probs: Option<[f32; 3]>,
}

/// The processed per-transition learning targets, produced by
/// [`ReplayBuffer::process_data`] over each per-env trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Targets {
    /// Categorical λ-return (the value-CE target), `td_lambda` discounted (move
    /// default 0.8): the vector cumsum of `target_value_probs - value_probs`
    /// (reference `rl.py`'s `returns`, not a two-hot projection of a scalar
    /// return — the two differ whenever a bootstrapped value isn't a bare
    /// category anchor).
    pub ret: [f32; 3],
    /// GAE advantage, `gae_lambda` discounted (move default 0.5); scalar, from
    /// the aggregated (`value` / `target_value`) deltas.
    pub advantage: f32,
}

/// One completed deployment trajectory for the co-trained setup loop (§4.2): the
/// full 40-placement arrangement one player built, the per-slot data-policy
/// log-probs (the PPO ratio denominator / rev-KL-to-data target), and the
/// Monte-Carlo outcome of the game it seeded, in the *deploying player's* POV
/// (`+1` that player won). Mirrors the reference `ArrangementBuffer.add_rewards`
/// keying, but per-trajectory rather than deduped by combinatorial id.
#[derive(Debug, Clone, PartialEq)]
pub struct SetupGame {
    /// The deploying player (0 = red, 1 = blue).
    pub player: usize,
    /// The 40 placements in row-major home order (`PieceType as u8`), slot 0
    /// first. The setup net consumes this as the `(40, 14)` one-hot sequence.
    pub placements: [u8; 40],
    /// Per-slot data-policy log-prob of the placed type (parallel to
    /// `placements`): the log-prob the actor assigned the chosen type at that
    /// slot when it deployed.
    pub old_log_prob: [f32; 40],
    /// Monte-Carlo game outcome in the deploying player's POV (`-1`, `0`, `+1`).
    pub outcome: f32,
}

/// One re-encoded view of a stored transition: the observation the net consumes
/// and the legal mask, recomputed from the snapshot. Used to verify the buffer
/// round-trips and the history reconstruction matches the live encoder.
#[derive(Debug, Clone)]
pub struct EncodedView {
    pub obs: Vec<f32>,
    pub legal: Vec<u16>,
    pub phase: Phase,
    pub player: usize,
}

/// The circular replay buffer: a per-env ring of [`Transition`]s. `capacity` is
/// the per-env ring length (≥ the trajectory length so a full trajectory is
/// resident); transitions roll forward modulo `capacity`.
#[derive(Debug)]
pub struct ReplayBuffer {
    num_envs: usize,
    capacity: usize,
    cfg: EncoderConfig,
    rings: Vec<Vec<Option<Transition>>>,
    heads: Vec<usize>,
    counts: Vec<usize>,
}

impl ReplayBuffer {
    /// A buffer for `num_envs` envs, each a ring of `capacity` transitions.
    /// `capacity` must be ≥ 2 so the solipsistic two-ply fix-up has room.
    pub fn new(num_envs: usize, capacity: usize, cfg: EncoderConfig) -> ReplayBuffer {
        assert!(
            capacity >= 2,
            "ring needs room for the two-ply value target"
        );
        ReplayBuffer {
            num_envs,
            capacity,
            cfg,
            rings: (0..num_envs)
                .map(|_| (0..capacity).map(|_| None).collect())
                .collect(),
            heads: vec![0; num_envs],
            counts: vec![0; num_envs],
        }
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// The number of transitions ever recorded for `env` (the next write lands at
    /// `head % capacity`; the most recent resident is at `(head - 1) % capacity`).
    pub fn head(&self, env: usize) -> usize {
        self.heads[env]
    }

    /// Total transitions currently resident (capped at `num_envs * capacity`).
    pub fn len(&self) -> usize {
        self.counts.iter().map(|&c| c.min(self.capacity)).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }

    /// Records one transition for `env`, advancing its ring head. Resolves the
    /// previous same-player position's two-ply value target and applies the
    /// solipsistic terminal fix-up to the immediately-previous transition when
    /// `transition.is_terminating_action` first marks the boundary. Implemented
    /// by the buffer milestone.
    pub fn record(&mut self, env: usize, transition: Transition) {
        let cap = self.capacity;
        let idx = self.heads[env];
        let slot = idx % cap;

        let terminating = transition.is_terminating_action;
        let boundary = terminating && !transition.is_terminated_position;
        let terminal_reward = transition.terminal_reward;
        let value = transition.value;
        let value_probs = transition.value_probs;
        let truncated = transition.truncated;

        self.rings[env][slot] = Some(transition);

        if idx > 0 && boundary {
            let prev = (slot + cap - 1) % cap;
            if let Some(p) = self.rings[env][prev].as_mut() {
                p.terminal_reward -= terminal_reward;
                p.is_terminating_action = true;
                if truncated {
                    p.truncated = true;
                }
            }
        }

        if idx > 1 {
            let prev_prev = (slot + cap - 2) % cap;
            if let Some(pp) = self.rings[env][prev_prev].as_mut() {
                pp.target_value = Some(if pp.is_terminating_action {
                    if pp.truncated {
                        pp.value
                    } else {
                        pp.terminal_reward
                    }
                } else {
                    value
                });
                pp.target_value_probs = Some(if pp.is_terminating_action {
                    if pp.truncated {
                        pp.value_probs
                    } else {
                        two_hot(pp.terminal_reward)
                    }
                } else {
                    value_probs
                });
            }
        }

        self.heads[env] = idx + 1;
        self.counts[env] += 1;
    }

    /// The transition at ring slot `step` for `env`, if present.
    pub fn get(&self, env: usize, step: usize) -> Option<&Transition> {
        self.rings[env][step % self.capacity].as_ref()
    }

    /// Re-encodes the observation and legal mask for the transition at `(env,
    /// step)` from its stored snapshot — the on-demand reconstruction. Returns
    /// `None` if the slot is empty. Implemented by the buffer milestone.
    pub fn encode_view(&self, env: usize, step: usize) -> Option<EncodedView> {
        let transition = self.rings[env][step % self.capacity].as_ref()?;
        Some(match &transition.snapshot {
            Snapshot::Play { board, to_play } => {
                let obs = crate::encode::encode_tokens(board, *to_play, &self.cfg);
                let mask = crate::rules::legal_mask(board, *to_play);
                let legal = (0..crate::action::NUM_ACTIONS)
                    .filter(|&i| mask[i])
                    .map(|i| i as u16)
                    .collect();
                EncodedView {
                    obs,
                    legal,
                    phase: Phase::Move,
                    player: *to_play,
                }
            }
            Snapshot::Deploy { current, .. } => {
                let obs = crate::encode::deploy_obs(current);
                let legal = current.legal_types().iter().map(|&t| t as u16).collect();
                EncodedView {
                    obs,
                    legal,
                    phase: Phase::Deploy,
                    player: current.player,
                }
            }
        })
    }

    /// Computes the categorical λ-return and GAE advantage over the resident
    /// trajectory of one env, returning per-resident-slot [`Targets`] keyed by
    /// ring slot. Mirrors `buffer.py:process_data`'s `use_cat_vf` path:
    /// `delta = target_value_probs - value_probs` (per-category vectors);
    /// `returns = segmented_discounted_cumsum(delta, td_lambda * ~terminal) + value_probs`
    /// (vector); `scalar_delta = target_value - value`;
    /// `advantages = segmented_discounted_cumsum(scalar_delta, gae_lambda * ~terminal)`
    /// (scalar — advantages stay the aggregated form regardless of value-function
    /// flavor).
    pub fn process_data(
        &self,
        env: usize,
        td_lambda: f32,
        gae_lambda: f32,
    ) -> Vec<(usize, Targets)> {
        let cap = self.capacity;
        let resident = self.counts[env].min(cap);
        if resident == 0 {
            return Vec::new();
        }
        let start = self.heads[env] - resident;

        let mut slots = Vec::with_capacity(resident);
        let mut value_probs = Vec::with_capacity(resident);
        let mut delta = Vec::with_capacity(resident);
        let mut delta_probs = Vec::with_capacity(resident);
        let mut td_disc = Vec::with_capacity(resident);
        let mut gae_disc = Vec::with_capacity(resident);
        for i in 0..resident {
            let slot = (start + i) % cap;
            let Some(t) = self.rings[env][slot].as_ref() else {
                continue;
            };
            let v = t.value;
            let vp = t.value_probs;
            let target = t.target_value.unwrap_or(v);
            let target_p = t.target_value_probs.unwrap_or(vp);
            slots.push(slot);
            value_probs.push(vp);
            delta.push(target - v);
            delta_probs.push(sub3(target_p, vp));
            let cont = if t.is_terminating_action { 0.0 } else { 1.0 };
            td_disc.push(td_lambda * cont);
            gae_disc.push(gae_lambda * cont);
        }

        let adv = segmented_discounted_cumsum(&delta, &gae_disc);
        let ret_cumsum = segmented_discounted_cumsum_vec(&delta_probs, &td_disc);
        (0..slots.len())
            .map(|i| {
                (
                    slots[i],
                    Targets {
                        ret: add3(ret_cumsum[i], value_probs[i]),
                        advantage: adv[i],
                    },
                )
            })
            .collect()
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Vector counterpart of [`segmented_discounted_cumsum`]: the same recurrence
/// applied independently per category (the discount at each step is a shared
/// scalar, so the 3-category recursion decomposes exactly into three scalar
/// recursions).
fn segmented_discounted_cumsum_vec(x: &[[f32; 3]], disc: &[f32]) -> Vec<[f32; 3]> {
    let mut y = vec![[0.0f32; 3]; x.len()];
    if x.is_empty() {
        return y;
    }
    let last = x.len() - 1;
    y[last] = x[last];
    for t in (0..last).rev() {
        for c in 0..3 {
            y[t][c] = x[t][c] + disc[t] * y[t + 1][c];
        }
    }
    y
}

/// Reverse scan matching `buffer.py:segmented_discounted_cumsum`: `y[last] =
/// x[last]`; `y[t] = x[t] + disc[t] * y[t + 1]` for earlier `t`. The discount at
/// `t` gates the carry from `t + 1`, so a per-step terminal (disc 0) breaks the
/// segment there.
fn segmented_discounted_cumsum(x: &[f32], disc: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    if x.is_empty() {
        return y;
    }
    let last = x.len() - 1;
    y[last] = x[last];
    for t in (0..last).rev() {
        y[t] = x[t] + disc[t] * y[t + 1];
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::board::{Board, PieceType};
    use crate::game::{State, Stratego};
    use crate::rules;

    const ARR: &str = "KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD";

    fn play_board(red: &str, blue: &str) -> Board {
        match Stratego::from_arrangement_strings(red, blue).expect("valid arrangements") {
            State::Play { board, .. } => *board,
            _ => unreachable!(),
        }
    }

    fn live_obs(board: &Board, to_play: usize, cfg: &EncoderConfig) -> Vec<f32> {
        crate::encode::encode_tokens(board, to_play, cfg)
    }

    fn live_legal(board: &Board, to_play: usize) -> Vec<u16> {
        let mask = rules::legal_mask(board, to_play);
        (0..crate::action::NUM_ACTIONS)
            .filter(|&i| mask[i])
            .map(|i| i as u16)
            .collect()
    }

    fn play_snapshot(board: &Board, to_play: usize) -> Transition {
        let legal = live_legal(board, to_play);
        Transition {
            snapshot: Snapshot::Play {
                board: Box::new(board.clone()),
                to_play,
            },
            phase: Phase::Move,
            player: to_play,
            num_moves: board.num_moves,
            action: *legal.first().unwrap_or(&0),
            legal,
            old_log_probs: Vec::new(),
            chosen: 0,
            value: 0.0,
            value_probs: two_hot(0.0),
            is_terminated_position: false,
            is_terminating_action: false,
            terminal_reward: 0.0,
            truncated: false,
            target_value: None,
            target_value_probs: None,
        }
    }

    /// A bare move-phase transition carrying just the value/terminal fields the
    /// record/process logic reads, snapshotting a fixed board for completeness.
    /// `value_probs` is `two_hot(value)` — a stand-in categorical distribution
    /// whose expectation is the given scalar (exact whenever `value` is itself a
    /// bare category anchor, e.g. a terminal reward).
    fn synth(
        player: usize,
        value: f32,
        is_terminating_action: bool,
        is_terminated_position: bool,
        terminal_reward: f32,
    ) -> Transition {
        let board = play_board(ARR, ARR);
        Transition {
            snapshot: Snapshot::Play {
                board: Box::new(board),
                to_play: player,
            },
            phase: Phase::Move,
            player,
            num_moves: 0,
            action: 0,
            legal: vec![0],
            old_log_probs: Vec::new(),
            chosen: 0,
            value,
            value_probs: two_hot(value),
            is_terminated_position,
            is_terminating_action,
            terminal_reward,
            truncated: false,
            target_value: None,
            target_value_probs: None,
        }
    }

    #[test]
    fn round_trip_reencodes_live_board() {
        let cfg = EncoderConfig::default();
        let mut buf = ReplayBuffer::new(1, 64, cfg);
        let mut board = play_board(ARR, ARR);

        rules::apply(&mut board, Action::from_abs(20, 30, 0).unwrap(), 0);
        rules::apply(&mut board, Action::from_abs(99, 98, 1).unwrap(), 1);
        rules::apply(&mut board, Action::from_abs(30, 40, 0).unwrap(), 0);

        let to_play = 1;
        buf.record(0, play_snapshot(&board, to_play));

        let view = buf.encode_view(0, 0).expect("present");
        assert_eq!(view.phase, Phase::Move);
        assert_eq!(view.player, to_play);
        assert_eq!(view.obs, live_obs(&board, to_play, &cfg));
        assert_eq!(view.legal, live_legal(&board, to_play));
    }

    #[test]
    fn history_reconstructs_per_step() {
        let cfg = EncoderConfig::default();
        let mut buf = ReplayBuffer::new(1, 64, cfg);
        let mut board = play_board(ARR, ARR);

        let moves = [
            (0usize, 20usize, 30usize),
            (1, 99, 98),
            (0, 30, 40),
            (1, 98, 97),
            (0, 40, 50),
            (1, 97, 96),
            (0, 50, 51),
            (1, 96, 95),
        ];

        let mut step = 0usize;
        let mut expected: Vec<(Vec<f32>, Vec<u16>, usize)> = Vec::new();
        for &(p, s, d) in &moves {
            let to_play = p;
            expected.push((
                live_obs(&board, to_play, &cfg),
                live_legal(&board, to_play),
                to_play,
            ));
            buf.record(0, play_snapshot(&board, to_play));
            step += 1;
            rules::apply(&mut board, Action::from_abs(s, d, p).unwrap(), p);
        }
        assert_eq!(step, moves.len());

        for (i, (obs, legal, player)) in expected.into_iter().enumerate() {
            let view = buf.encode_view(0, i).expect("present");
            assert_eq!(view.player, player, "player at step {i}");
            assert_eq!(view.legal, legal, "legal at step {i}");
            assert_eq!(view.obs, obs, "obs at step {i}");
        }
    }

    #[test]
    fn deploy_view_encodes_placement_prefix() {
        let cfg = EncoderConfig::default();
        let mut buf = ReplayBuffer::new(1, 4, cfg);
        let mut current = DeploymentState::classic(0, true);
        current.place(PieceType::Scout);
        current.place(PieceType::Miner);

        let legal: Vec<u16> = current.legal_types().iter().map(|&t| t as u16).collect();
        buf.record(
            0,
            Transition {
                snapshot: Snapshot::Deploy {
                    red: None,
                    current: current.clone(),
                },
                phase: Phase::Deploy,
                player: 0,
                num_moves: 0,
                action: PieceType::Scout as u16,
                legal: legal.clone(),
                old_log_probs: Vec::new(),
                chosen: 0,
                value: 0.0,
                value_probs: two_hot(0.0),
                is_terminated_position: false,
                is_terminating_action: false,
                terminal_reward: 0.0,
                truncated: false,
                target_value: None,
                target_value_probs: None,
            },
        );

        let view = buf.encode_view(0, 0).expect("present");
        assert_eq!(view.phase, Phase::Deploy);
        assert_eq!(view.player, 0);
        assert_eq!(view.legal, legal);

        let mut want = vec![0.0f32; crate::board::HOME_CELLS * 14];
        want[PieceType::Scout as usize] = 1.0;
        want[14 + PieceType::Miner as usize] = 1.0;
        assert_eq!(view.obs, want);
    }

    #[test]
    fn two_ply_target_bootstrap_and_terminal() {
        let mut buf = ReplayBuffer::new(1, 16, EncoderConfig::default());
        // t0 red (v=0.2), t1 blue, t2 red (v=0.7): t0 must bootstrap to v[t2].
        buf.record(0, synth(0, 0.2, false, false, 0.0));
        buf.record(0, synth(1, 0.5, false, false, 0.0));
        buf.record(0, synth(0, 0.7, false, false, 0.0));
        assert_eq!(buf.get(0, 0).unwrap().target_value, Some(0.7));

        // A terminating position bootstraps to its own terminal reward, not v[t].
        let mut buf2 = ReplayBuffer::new(1, 16, EncoderConfig::default());
        buf2.record(0, synth(0, 0.2, true, false, 1.0)); // t0 terminating, reward +1
        buf2.record(0, synth(1, 0.5, false, false, 0.0)); // t1 opponent
        buf2.record(0, synth(0, 0.7, false, false, 0.0)); // t2 resolves t0
        assert_eq!(buf2.get(0, 0).unwrap().target_value, Some(1.0));
    }

    #[test]
    fn solipsistic_terminal_fixup_rewrites_previous() {
        let mut buf = ReplayBuffer::new(1, 16, EncoderConfig::default());
        buf.record(0, synth(0, 0.1, false, false, 0.0)); // t0 red
        buf.record(0, synth(1, 0.2, false, false, 0.0)); // t1 blue
        // t2 red's move objectively terminates with reward +1 (boundary).
        buf.record(0, synth(0, 0.9, true, false, 1.0));

        let prev = buf.get(0, 1).unwrap();
        assert!(prev.is_terminating_action, "t1 marked terminating");
        assert_eq!(prev.terminal_reward, -1.0, "t1 reward sign-flipped");

        // The terminated position itself must not be a boundary, so no fix-up.
        let mut buf2 = ReplayBuffer::new(1, 16, EncoderConfig::default());
        buf2.record(0, synth(0, 0.1, false, false, 0.0));
        buf2.record(0, synth(1, 0.2, true, true, 0.5)); // terminated position
        assert!(!buf2.get(0, 0).unwrap().is_terminating_action);
        assert_eq!(buf2.get(0, 0).unwrap().terminal_reward, 0.0);
    }

    #[test]
    fn process_data_matches_hand_computation() {
        let td = 0.8f32;
        let gae = 0.5f32;
        let mut buf = ReplayBuffer::new(1, 16, EncoderConfig::default());

        // Three-step trajectory; the last move terminates (reward +1).
        buf.record(0, synth(0, 0.2, false, false, 0.0));
        buf.record(0, synth(1, 0.4, false, false, 0.0));
        buf.record(0, synth(0, 0.6, true, false, 1.0));

        // Resolve the two-ply targets the live recorder leaves for the tail:
        // t0 bootstraps to v[t2]=0.6, t1 has no t+2 so stays None -> own value.
        // Replicate the reference: unresolved tail bootstraps to its own value.
        let value = [0.2f32, 0.4, 0.6];
        let target = [
            buf.get(0, 0).unwrap().target_value.unwrap_or(value[0]),
            buf.get(0, 1).unwrap().target_value.unwrap_or(value[1]),
            buf.get(0, 2).unwrap().target_value.unwrap_or(value[2]),
        ];
        let term = [false, false, true];

        let delta: Vec<f32> = (0..3).map(|i| target[i] - value[i]).collect();
        let cont = |t: bool| if t { 0.0 } else { 1.0 };
        let gae_disc: Vec<f32> = term.iter().map(|&t| gae * cont(t)).collect();

        let adv_cs = {
            let mut y = vec![0.0f32; 3];
            y[2] = delta[2];
            y[1] = delta[1] + gae_disc[1] * y[2];
            y[0] = delta[0] + gae_disc[0] * y[1];
            y
        };

        let out = buf.process_data(0, td, gae);
        assert_eq!(out.len(), 3);
        for (i, (slot, targets)) in out.iter().enumerate() {
            assert_eq!(*slot, i, "slot order");
            assert!(
                (targets.advantage - adv_cs[i]).abs() < 1e-6,
                "adv[{i}] got {} want {}",
                targets.advantage,
                adv_cs[i]
            );
        }
    }

    /// The categorical value-CE target (`Targets::ret`): a genuine per-category
    /// vector cumsum of `two_hot(target) - two_hot(value)` (reference
    /// `buffer.py`'s `use_cat_vf` path), NOT a two-hot projection of the
    /// aggregated scalar return. The two differ whenever a bootstrapped value
    /// isn't a bare category anchor — this test's values (0.2/0.4/0.6) are
    /// deliberately off-anchor to exercise that.
    #[test]
    fn process_data_categorical_return_is_a_vector_cumsum_not_a_scalar_two_hot() {
        let td = 0.8f32;
        let gae = 0.5f32;
        let mut buf = ReplayBuffer::new(1, 16, EncoderConfig::default());

        buf.record(0, synth(0, 0.2, false, false, 0.0));
        buf.record(0, synth(1, 0.4, false, false, 0.0));
        buf.record(0, synth(0, 0.6, true, false, 1.0));

        let value = [0.2f32, 0.4, 0.6];
        let value_probs: Vec<[f32; 3]> = value.iter().map(|&v| two_hot(v)).collect();
        // Same bootstrap structure as the scalar test: only t0 resolves (to
        // value_probs[2]); t1/t2 fall back to their own distribution.
        let target_probs = [value_probs[2], value_probs[1], value_probs[2]];
        let term = [false, false, true];

        let delta_probs: Vec<[f32; 3]> = (0..3)
            .map(|i| sub3(target_probs[i], value_probs[i]))
            .collect();
        let cont = |t: bool| if t { 0.0 } else { 1.0 };
        let td_disc: Vec<f32> = term.iter().map(|&t| td * cont(t)).collect();

        let mut y = [[0.0f32; 3]; 3];
        y[2] = delta_probs[2];
        for t in [1usize, 0] {
            for c in 0..3 {
                y[t][c] = delta_probs[t][c] + td_disc[t] * y[t + 1][c];
            }
        }
        let want_ret: Vec<[f32; 3]> = (0..3).map(|i| add3(y[i], value_probs[i])).collect();

        // Sanity: the vector return's expectation must still match the old
        // scalar-only cumsum (0.6, 0.4, 0.6) — proves the two formulations agree
        // in aggregate even though the distributions differ.
        const CATS: [f32; 3] = crate::evaluator::VALUE_CATEGORIES;
        let want_scalar = [0.6f32, 0.4, 0.6];
        for (i, r) in want_ret.iter().enumerate() {
            let agg = r[0] * CATS[0] + r[1] * CATS[1] + r[2] * CATS[2];
            assert!(
                (agg - want_scalar[i]).abs() < 1e-6,
                "ret[{i}] aggregate got {agg} want {}",
                want_scalar[i]
            );
        }

        let out = buf.process_data(0, td, gae);
        assert_eq!(out.len(), 3);
        for (i, (_, targets)) in out.iter().enumerate() {
            for (c, (&got, &want)) in targets.ret.iter().zip(want_ret[i].iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "ret[{i}][{c}] got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn ring_wrap_keeps_last_capacity_transitions() {
        let mut buf = ReplayBuffer::new(1, 3, EncoderConfig::default());
        for i in 0..7 {
            buf.record(0, synth(i % 2, i as f32, false, false, 0.0));
        }
        // capacity 3, 7 recorded: residents are records 4,5,6 (values 4,5,6).
        let out = buf.process_data(0, 0.8, 0.5);
        assert_eq!(out.len(), 3);
        let slots: Vec<usize> = out.iter().map(|(s, _)| *s).collect();
        // Oldest resident is head-3 = 4 -> slot 1, then slot 2, then slot 0.
        assert_eq!(slots, vec![1, 2, 0]);
        let values: Vec<f32> = out
            .iter()
            .map(|(s, _)| buf.get(0, *s).unwrap().value)
            .collect();
        assert_eq!(values, vec![4.0, 5.0, 6.0]);
    }
}
