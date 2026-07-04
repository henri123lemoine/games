//! The pyo3 bridge: a `BatchSim` Python class driving the verified Rust Stratego
//! self-play simulator (`games/stratego`) so the MLX trainer owns the loop —
//! `collect()` → MLX net forward → `commit()` → `drain_training_batch()`.
//!
//! ## The loop the trainer runs (EnvPool pattern)
//! ```text
//! batch = sim.collect()                 # ragged per-env decisions, split by phase
//! mlogits, mvals = move_net(batch["move_obs"], batch["move_legal"])   # (n_move, 1800), (n_move,)
//! dlogits, dvals = setup_net(batch["deploy_obs"], batch["deploy_legal"])
//! out = sim.commit(mlogits, mvals, dlogits, dvals)                    # samples + applies + records
//! ...                                   # every few hundred steps:
//! data = sim.drain_training_batch(0.8, 0.5)                          # the move-RL arrays
//! ```
//! The net forward is the throughput bottleneck, so the Python-loop overhead per
//! step is negligible. `commit` reuses the exact Rust softmax-sample/record path
//! ([`stratego::Simulator::commit`]) — there is no second sampling code path.
//!
//! ## Ragged-vs-padded representation
//! Decisions are split by phase and **padded** per phase (not ragged
//! per-env): move rows carry a dense `(1800,)` boolean legal mask, deploy rows a
//! dense `(14,)` one. The mask is exactly what the MLX loss needs to set illegal
//! logits to `-inf` before its categorical, so no separate ragged index list is
//! exposed. `move_env`/`deploy_env` map each row back to its env id.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use stratego::encode::{DEPLOY_TYPE_WIDTH, NUM_OCCUPIABLE_CELLS};
use stratego::evaluator::{Evaluation, Phase};
use stratego::search::{
    RolloutBatch, RolloutDecision, RootInfo, assign_hidden, marginal_posterior,
};
use stratego::{Collected, EncoderConfig, NUM_ACTIONS, ReplayBuffer, SetupGame, Simulator};

const N_PIECE_TYPE: usize = DEPLOY_TYPE_WIDTH; // 14

const N_ACTION: usize = NUM_ACTIONS; // 1800
const DEPLOY_SLOTS: usize = stratego::board::HOME_CELLS; // 40
const DEPLOY_WIDTH: usize = DEPLOY_TYPE_WIDTH; // 14
const MOVE_TOKENS: usize = NUM_OCCUPIABLE_CELLS; // 92

/// A per-env decision the bridge keeps resident between `collect` and `commit`:
/// which phase the env is in and its legal option indices (into the 1800 move
/// space or the 14 deploy-type space). Lets `commit` gather the net's dense
/// logits back down to each env's ragged legal set for the verified sampler.
struct PendingEnv {
    phase: Phase,
    /// Row of this env within its phase's padded batch (move or deploy).
    row: usize,
    legal: Vec<u16>,
}

/// The batch state carried across one `collect`/`commit` pair.
struct Pending {
    collected: Collected,
    envs: Vec<PendingEnv>,
    n_move: usize,
    n_deploy: usize,
}

/// One side's in-progress deployment for an env, filled placement-by-placement
/// as deploy transitions are committed. Becomes a [`SetupGame`] once the game it
/// seeds terminates (the MC outcome is known then) — see [`SetupAccumulator`].
#[derive(Clone)]
struct PartialSetup {
    player: usize,
    placements: Vec<u8>,
    old_log_prob: Vec<f32>,
}

impl PartialSetup {
    fn new(player: usize) -> Self {
        PartialSetup {
            player,
            placements: Vec::with_capacity(DEPLOY_SLOTS),
            old_log_prob: Vec::with_capacity(DEPLOY_SLOTS),
        }
    }
}

/// Per-env accumulator that turns the interleaved deploy/move stream into
/// completed setup trajectories with their final-game returns (§4.2). Deploy
/// transitions carry no outcome (a deployment is never rules-terminal), so we
/// buffer each side's 40 placements as they happen and stamp the player-0 game
/// outcome onto both sides the moment the seeded game terminates. This is
/// independent of the replay ring, so it is robust to ring eviction across the
/// long move phase. The completed games queue drains into `drain_setup_batch`.
struct SetupAccumulator {
    /// In-progress placements per env, in deploy order: red (player 0) fills
    /// first, then blue (player 1). Two complete entries are pending per env at
    /// the moment its game starts the move phase.
    pending: Vec<Vec<PartialSetup>>,
    completed: Vec<SetupGame>,
}

impl SetupAccumulator {
    fn new(num_envs: usize) -> Self {
        SetupAccumulator {
            pending: (0..num_envs).map(|_| Vec::new()).collect(),
            completed: Vec::new(),
        }
    }

    /// Record one committed deploy placement for `env`.
    fn record_deploy(&mut self, env: usize, player: usize, piece_type: u8, old_log_prob: f32) {
        let slot = &mut self.pending[env];
        // Start a fresh side whenever the deploying player changes (red->blue)
        // or no side is in progress.
        let need_new = match slot.last() {
            None => true,
            Some(last) => last.player != player || last.placements.len() >= DEPLOY_SLOTS,
        };
        if need_new {
            slot.push(PartialSetup::new(player));
        }
        let side = slot.last_mut().expect("just ensured");
        side.placements.push(piece_type);
        side.old_log_prob.push(old_log_prob);
    }

    /// The seeded game for `env` terminated with player-0 reward `reward_pl0`.
    /// Flush every complete (40-placement) side as a [`SetupGame`] with the
    /// outcome in that side's POV, then clear the env's pending sides.
    fn record_terminal(&mut self, env: usize, reward_pl0: f32) {
        for side in self.pending[env].drain(..) {
            if side.placements.len() != DEPLOY_SLOTS {
                continue; // a partial side (ring-start truncation) — skip.
            }
            let outcome = if side.player == 0 {
                reward_pl0
            } else {
                -reward_pl0
            };
            let mut placements = [0u8; DEPLOY_SLOTS];
            let mut old_log_prob = [0f32; DEPLOY_SLOTS];
            placements.copy_from_slice(&side.placements);
            old_log_prob.copy_from_slice(&side.old_log_prob);
            self.completed.push(SetupGame {
                player: side.player,
                placements,
                old_log_prob,
                outcome,
            });
        }
    }

    /// A force-reset that carried no genuine outcome (e.g. a ply cap) — discard
    /// the env's in-progress sides so they do not attach to the next game.
    fn discard(&mut self, env: usize) {
        self.pending[env].clear();
    }

    fn take_completed(&mut self) -> Vec<SetupGame> {
        std::mem::take(&mut self.completed)
    }
}

/// The high-throughput Stratego self-play simulator, exposed to Python.
///
/// `BatchSim(num_envs, move_cap, seed, history_len=32)`.
#[pyclass]
struct BatchSim {
    sim: Simulator,
    buffer: ReplayBuffer,
    move_feat: usize,
    pending: Option<Pending>,
    setup: SetupAccumulator,
    /// Constructor seed, kept so the heuristic-opponent eval can derive a
    /// per-env [`HeuristicBot`] RNG deterministically (the live sampler path
    /// never reads it).
    seed: u64,
}

#[pymethods]
impl BatchSim {
    #[new]
    #[pyo3(signature = (num_envs, move_cap, seed, history_len = 32, buffer_capacity = 256))]
    fn new(
        num_envs: usize,
        move_cap: u32,
        seed: u64,
        history_len: usize,
        buffer_capacity: usize,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be > 0"));
        }
        if buffer_capacity < 2 {
            return Err(PyValueError::new_err("buffer_capacity must be >= 2"));
        }
        let cfg = EncoderConfig {
            history_len,
            ..EncoderConfig::default()
        };
        let sim = Simulator::new(num_envs, cfg, seed, move_cap);
        let buffer = ReplayBuffer::new(num_envs, buffer_capacity, cfg);
        Ok(BatchSim {
            sim,
            buffer,
            move_feat: cfg.num_token_features(),
            pending: None,
            setup: SetupAccumulator::new(num_envs),
            seed,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.sim.num_envs()
    }

    /// Per-token move-net feature width (`355 + history_len + 256`); 643 at the
    /// default `history_len = 32`.
    #[getter]
    fn move_feat(&self) -> usize {
        self.move_feat
    }

    #[getter]
    fn n_action(&self) -> usize {
        N_ACTION
    }

    #[getter]
    fn deploy_slots(&self) -> usize {
        DEPLOY_SLOTS
    }

    #[getter]
    fn deploy_width(&self) -> usize {
        DEPLOY_WIDTH
    }

    #[getter]
    fn move_tokens(&self) -> usize {
        MOVE_TOKENS
    }

    /// Snapshot every live env and encode the batch, split by phase. Returns a
    /// dict of numpy arrays (see module docs for the ragged-vs-padded layout):
    ///
    /// * `move_obs` `(n_move, 92, move_feat) f32`, `move_legal` `(n_move, 1800)` bool,
    ///   `move_env` `(n_move,) i64`, `move_player` `(n_move,) i64`.
    /// * `deploy_obs` `(n_deploy, 40, 14) f32`, `deploy_legal` `(n_deploy, 14)` bool,
    ///   `deploy_env` `(n_deploy,) i64`, `deploy_player` `(n_deploy,) i64`.
    ///
    /// Must be paired with one [`commit`](BatchSim::commit) before the next
    /// `collect`.
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let collected = self.sim.collect();
        let decisions = collected.decisions();

        let mut move_idx = Vec::new();
        let mut deploy_idx = Vec::new();
        let mut envs = Vec::with_capacity(decisions.len());
        for d in decisions {
            let (phase, row) = match d.phase {
                Phase::Move => {
                    let row = move_idx.len();
                    move_idx.push(envs.len());
                    (Phase::Move, row)
                }
                Phase::Deploy => {
                    let row = deploy_idx.len();
                    deploy_idx.push(envs.len());
                    (Phase::Deploy, row)
                }
            };
            envs.push(PendingEnv {
                phase,
                row,
                legal: d.legal.clone(),
            });
        }

        let n_move = move_idx.len();
        let n_deploy = deploy_idx.len();
        let feat = self.move_feat;

        let mut move_obs = Array3::<f32>::zeros((n_move, MOVE_TOKENS, feat));
        let mut move_legal = Array2::<bool>::default((n_move, N_ACTION));
        let mut move_env = Array1::<i64>::zeros(n_move);
        let mut move_player = Array1::<i64>::zeros(n_move);
        for (row, &env) in move_idx.iter().enumerate() {
            let d = &decisions[env];
            move_env[row] = env as i64;
            move_player[row] = d.player as i64;
            move_obs
                .slice_mut(numpy::ndarray::s![row, .., ..])
                .as_slice_mut()
                .expect("contiguous row")
                .copy_from_slice(&d.obs);
            for &a in &d.legal {
                move_legal[(row, a as usize)] = true;
            }
        }

        let mut deploy_obs = Array3::<f32>::zeros((n_deploy, DEPLOY_SLOTS, DEPLOY_WIDTH));
        let mut deploy_legal = Array2::<bool>::default((n_deploy, DEPLOY_WIDTH));
        let mut deploy_env = Array1::<i64>::zeros(n_deploy);
        let mut deploy_player = Array1::<i64>::zeros(n_deploy);
        for (row, &env) in deploy_idx.iter().enumerate() {
            let d = &decisions[env];
            deploy_env[row] = env as i64;
            deploy_player[row] = d.player as i64;
            deploy_obs
                .slice_mut(numpy::ndarray::s![row, .., ..])
                .as_slice_mut()
                .expect("contiguous row")
                .copy_from_slice(&d.obs);
            for &t in &d.legal {
                deploy_legal[(row, t as usize)] = true;
            }
        }

        self.pending = Some(Pending {
            collected,
            envs,
            n_move,
            n_deploy,
        });

        let out = PyDict::new(py);
        out.set_item("move_obs", move_obs.into_pyarray(py))?;
        out.set_item("move_legal", move_legal.into_pyarray(py))?;
        out.set_item("move_env", move_env.into_pyarray(py))?;
        out.set_item("move_player", move_player.into_pyarray(py))?;
        out.set_item("deploy_obs", deploy_obs.into_pyarray(py))?;
        out.set_item("deploy_legal", deploy_legal.into_pyarray(py))?;
        out.set_item("deploy_env", deploy_env.into_pyarray(py))?;
        out.set_item("deploy_player", deploy_player.into_pyarray(py))?;
        Ok(out)
    }

    /// Take the net's per-phase logits + scalar values, gather them down to each
    /// env's legal set, then softmax-sample / apply / record via the verified
    /// Rust core, auto-resetting terminals.
    ///
    /// Shapes (parallel to the last [`collect`](BatchSim::collect)):
    /// * `move_logits` `(n_move, 1800) f32` — full env-action logits; illegal
    ///   slots are ignored (the legal mask selects).
    /// * `move_values` `(n_move,) f32` — scalar value in `[-1, 1]` (the net's
    ///   `softmax(W/L/D) @ [-1, 0, 1]`), acting-player POV.
    /// * `move_value_probs` `(n_move, 3) f32` — the move net's raw `softmax(W/L/D)`
    ///   the scalar above aggregates (`[P(loss), P(tie), P(win)]`); feeds the
    ///   replay buffer's categorical λ-return (ATARAXOS_SPEC §4.1).
    /// * `deploy_logits` `(n_deploy, 14) f32`, `deploy_values` `(n_deploy,) f32`,
    ///   `deploy_value_probs` `(n_deploy, 3) f32` — same contract for the setup
    ///   net (its distribution can still be the bootstrap target for a move
    ///   transition two plies before a deploy-phase boundary).
    ///
    /// Returns a dict: `terminal` `(num_envs,)` bool (env completed a game this
    /// step) and `reward_pl0` `(num_envs,) f32` (player-0 terminal reward, 0 if
    /// not terminal).
    #[allow(clippy::too_many_arguments)]
    fn commit<'py>(
        &mut self,
        py: Python<'py>,
        move_logits: PyReadonlyArray2<'py, f32>,
        move_values: PyReadonlyArray1<'py, f32>,
        move_value_probs: PyReadonlyArray2<'py, f32>,
        deploy_logits: PyReadonlyArray2<'py, f32>,
        deploy_values: PyReadonlyArray1<'py, f32>,
        deploy_value_probs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("commit() called before collect()"))?;

        let mlogits = move_logits.as_array();
        let mvalues = move_values.as_array();
        let mvalue_probs = move_value_probs.as_array();
        let dlogits = deploy_logits.as_array();
        let dvalues = deploy_values.as_array();
        let dvalue_probs = deploy_value_probs.as_array();

        check_shape("move_logits", mlogits.shape(), &[pending.n_move, N_ACTION])?;
        check_len("move_values", mvalues.len(), pending.n_move)?;
        check_shape(
            "move_value_probs",
            mvalue_probs.shape(),
            &[pending.n_move, 3],
        )?;
        check_shape(
            "deploy_logits",
            dlogits.shape(),
            &[pending.n_deploy, DEPLOY_WIDTH],
        )?;
        check_len("deploy_values", dvalues.len(), pending.n_deploy)?;
        check_shape(
            "deploy_value_probs",
            dvalue_probs.shape(),
            &[pending.n_deploy, 3],
        )?;

        // Gather the dense net logits down to each env's ragged legal set, in env
        // order, building exactly the `Evaluation` the Rust sampler consumes.
        let evals: Vec<Evaluation> = pending
            .envs
            .iter()
            .map(|e| {
                let (logits, value, value_probs) = match e.phase {
                    Phase::Move => {
                        let row = mlogits.row(e.row);
                        let logits = e.legal.iter().map(|&a| row[a as usize]).collect();
                        let vp = mvalue_probs.row(e.row);
                        (logits, mvalues[e.row], [vp[0], vp[1], vp[2]])
                    }
                    Phase::Deploy => {
                        let row = dlogits.row(e.row);
                        let logits = e.legal.iter().map(|&t| row[t as usize]).collect();
                        let vp = dvalue_probs.row(e.row);
                        (logits, dvalues[e.row], [vp[0], vp[1], vp[2]])
                    }
                };
                Evaluation {
                    logits,
                    value,
                    value_probs,
                }
            })
            .collect();

        let n_envs = self.sim.num_envs();
        let result = self
            .sim
            .commit(&pending.collected, &evals, &mut self.buffer);

        let mut terminal = Array1::<bool>::default(n_envs);
        let mut reward_pl0 = Array1::<f32>::zeros(n_envs);
        let mut capped = Array1::<bool>::default(n_envs);
        for (env, completed) in result.completed.iter().enumerate() {
            // Feed the setup accumulator from the transition this commit just
            // recorded for `env` (most recent ring slot). A deploy placement
            // extends the env's in-progress side; a genuine rules-terminal move
            // flushes both sides with the game outcome.
            let head = self.buffer.head(env);
            if head > 0
                && let Some(t) = self.buffer.get(env, head - 1)
                && t.phase == Phase::Deploy
            {
                let lp = t.old_log_probs.get(t.chosen).copied().unwrap_or(0.0);
                self.setup.record_deploy(env, t.player, t.action as u8, lp);
            }

            if let Some(r) = completed {
                terminal[env] = true;
                reward_pl0[env] = *r as f32;
                capped[env] = result.capped[env];
                // A genuine rules-terminal stamps the outcome onto the deploy
                // sides; a pure ply-cap (no terminating action recorded)
                // carries no real result and is discarded.
                let genuine = head > 0
                    && self
                        .buffer
                        .get(env, head - 1)
                        .is_some_and(|t| t.is_terminating_action);
                if genuine && !result.capped[env] {
                    self.setup.record_terminal(env, *r as f32);
                } else {
                    self.setup.discard(env);
                }
            }
        }

        let out = PyDict::new(py);
        out.set_item("terminal", terminal.into_pyarray(py))?;
        out.set_item("reward_pl0", reward_pl0.into_pyarray(py))?;
        // `capped[env]` is only meaningful where `terminal[env]` — a ply-cap
        // force-reset with zero reward, distinct from a genuine rules-terminal
        // draw (rare, e.g. a chase/twosquare repetition ruling), so trainers can
        // tell "the game timed out" from "the game was actually drawn."
        out.set_item("capped", capped.into_pyarray(py))?;
        Ok(out)
    }

    /// Drain the resident move-phase transitions into the move-RL training arrays
    /// (per §4.1 of `ATARAXOS_SPEC.md`), running λ-returns / GAE over each env's
    /// trajectory. `N` = total resident move-phase transitions across all envs.
    ///
    /// Returns a dict of numpy arrays, all parallel along `N`:
    /// * `obs` `(N, 92, move_feat) f32` — re-encoded from each snapshot on demand.
    /// * `action` `(N,) i64` — the chosen 1800-space env-action index.
    /// * `legal_mask` `(N, 1800)` bool.
    /// * `old_log_prob` `(N,) f32` — data-policy log-prob of the chosen action.
    /// * `data_log_prob` `(N, 1800) f32` — data-policy log-probs on legal slots,
    ///   `-inf` off-legal (the rev-KL-to-data target distribution).
    /// * `value` `(N,) f32`, `target_value` `(N,) f32`, `advantage` `(N,) f32`,
    ///   `ret` `(N, 3) f32` (the categorical λ-return value-CE target,
    ///   `[P(loss), P(tie), P(win)]` — ATARAXOS_SPEC §4.1's `use_cat_vf` path,
    ///   NOT a two-hot projection of a scalar return).
    /// * `player` `(N,) i64`, `num_moves` `(N,) i64`, `is_terminating` `(N,)` bool.
    fn drain_training_batch<'py>(
        &self,
        py: Python<'py>,
        td_lambda: f32,
        gae_lambda: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        // First pass: count resident move-phase transitions and stash targets.
        let mut rows: Vec<(usize, usize, stratego::Targets)> = Vec::new();
        for env in 0..self.buffer.num_envs() {
            for (slot, targets) in self.buffer.process_data(env, td_lambda, gae_lambda) {
                if let Some(t) = self.buffer.get(env, slot)
                    && t.phase == Phase::Move
                {
                    rows.push((env, slot, targets));
                }
            }
        }
        let n = rows.len();

        let mut env_arr = Array1::<i64>::zeros(n);
        let mut slot_arr = Array1::<i64>::zeros(n);
        let mut action = Array1::<i64>::zeros(n);
        let mut legal_mask = Array2::<bool>::default((n, N_ACTION));
        let mut old_log_prob = Array1::<f32>::zeros(n);
        let mut data_log_prob = Array2::<f32>::from_elem((n, N_ACTION), f32::NEG_INFINITY);
        let mut value = Array1::<f32>::zeros(n);
        let mut target_value = Array1::<f32>::zeros(n);
        let mut advantage = Array1::<f32>::zeros(n);
        let mut ret = Array2::<f32>::zeros((n, 3));
        let mut player = Array1::<i64>::zeros(n);
        let mut num_moves = Array1::<i64>::zeros(n);
        let mut is_terminating = Array1::<bool>::default(n);

        for (i, &(env, slot, targets)) in rows.iter().enumerate() {
            let t = self.buffer.get(env, slot).expect("resident move slot");
            env_arr[i] = env as i64;
            slot_arr[i] = slot as i64;
            action[i] = t.action as i64;
            for (&a, &lp) in t.legal.iter().zip(t.old_log_probs.iter()) {
                legal_mask[(i, a as usize)] = true;
                data_log_prob[(i, a as usize)] = lp;
            }
            old_log_prob[i] = t.old_log_probs.get(t.chosen).copied().unwrap_or(0.0);
            value[i] = t.value;
            target_value[i] = t.target_value.unwrap_or(t.value);
            advantage[i] = targets.advantage;
            ret[(i, 0)] = targets.ret[0];
            ret[(i, 1)] = targets.ret[1];
            ret[(i, 2)] = targets.ret[2];
            player[i] = t.player as i64;
            num_moves[i] = t.num_moves as i64;
            is_terminating[i] = t.is_terminating_action;
        }

        let out = PyDict::new(py);
        out.set_item("env", env_arr.into_pyarray(py))?;
        out.set_item("slot", slot_arr.into_pyarray(py))?;
        out.set_item("action", action.into_pyarray(py))?;
        out.set_item("legal_mask", legal_mask.into_pyarray(py))?;
        out.set_item("old_log_prob", old_log_prob.into_pyarray(py))?;
        out.set_item("data_log_prob", data_log_prob.into_pyarray(py))?;
        out.set_item("value", value.into_pyarray(py))?;
        out.set_item("target_value", target_value.into_pyarray(py))?;
        out.set_item("advantage", advantage.into_pyarray(py))?;
        out.set_item("ret", ret.into_pyarray(py))?;
        out.set_item("player", player.into_pyarray(py))?;
        out.set_item("num_moves", num_moves.into_pyarray(py))?;
        out.set_item("is_terminating", is_terminating.into_pyarray(py))?;
        Ok(out)
    }

    /// Encode the move-net obs for a specific set of (env, slot) rows. The move
    /// pass trains on only the advantage-filtered subset (<= max_train_batch), so
    /// `drain_training_batch` returns env/slot but NOT obs, and we encode obs here
    /// for just those rows — skipping the ~94% of resident transitions the filter
    /// discards (the dominant per-iter encode cost).
    fn encode_move_obs<'py>(
        &self,
        py: Python<'py>,
        envs: PyReadonlyArray1<'py, i64>,
        slots: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
        let envs = envs.as_array();
        let slots = slots.as_array();
        let n = envs.len();
        let mut obs = Array3::<f32>::zeros((n, MOVE_TOKENS, self.move_feat));
        for i in 0..n {
            let env = envs[i] as usize;
            let slot = slots[i] as usize;
            let view = self
                .buffer
                .encode_view(env, slot)
                .expect("resident move slot");
            obs.slice_mut(numpy::ndarray::s![i, .., ..])
                .as_slice_mut()
                .expect("contiguous row")
                .copy_from_slice(&view.obs);
        }
        Ok(obs.into_pyarray(py))
    }

    /// Drain completed setup (deployment) trajectories into the co-trained
    /// setup-loop arrays (§4.2 of `ATARAXOS_SPEC.md`). Each row is one player's
    /// full 40-placement deployment with the Monte-Carlo outcome of the game it
    /// seeded, in that player's POV. Setup is pure MC (no λ-bootstrapping, no
    /// filtering), so the game's terminal return is the whole value target — no
    /// per-step processing is needed here, unlike `drain_training_batch`.
    ///
    /// `M` = number of completed setup trajectories since the last drain.
    /// Returns a dict of numpy arrays, parallel along `M`:
    /// * `seq` `(M, 40, 14) f32` — one-hot placement sequence (slot 0 first); the
    ///   exact input the setup net consumes (a zero start token is prepended
    ///   net-side, each slot predicting the next).
    /// * `action` `(M, 40) i64` — the chosen `PieceType` per slot.
    /// * `old_log_prob` `(M, 40) f32` — data-policy log-prob of each placement
    ///   (the PPO ratio denominator / rev-KL-to-data baseline).
    /// * `outcome` `(M,) f32` — MC game result in the deploying player's POV
    ///   (`-1`, `0`, `+1`).
    /// * `player` `(M,) i64` — deploying player (0 = red, 1 = blue).
    fn drain_setup_batch<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let games = self.setup.take_completed();
        let m = games.len();

        let mut seq = Array3::<f32>::zeros((m, DEPLOY_SLOTS, DEPLOY_WIDTH));
        let mut action = Array2::<i64>::zeros((m, DEPLOY_SLOTS));
        let mut old_log_prob = Array2::<f32>::zeros((m, DEPLOY_SLOTS));
        let mut outcome = Array1::<f32>::zeros(m);
        let mut player = Array1::<i64>::zeros(m);

        for (i, g) in games.iter().enumerate() {
            outcome[i] = g.outcome;
            player[i] = g.player as i64;
            for slot in 0..DEPLOY_SLOTS {
                let t = g.placements[slot];
                action[(i, slot)] = i64::from(t);
                old_log_prob[(i, slot)] = g.old_log_prob[slot];
                seq[(i, slot, t as usize)] = 1.0;
            }
        }

        let out = PyDict::new(py);
        out.set_item("seq", seq.into_pyarray(py))?;
        out.set_item("action", action.into_pyarray(py))?;
        out.set_item("old_log_prob", old_log_prob.into_pyarray(py))?;
        out.set_item("outcome", outcome.into_pyarray(py))?;
        out.set_item("player", player.into_pyarray(py))?;
        Ok(out)
    }

    /// Open a test-time **search root** at `env`'s current move-phase position
    /// (§5). Returns a [`Searcher`] session pre-loaded with the cloned root board
    /// plus the hidden-opponent inventory and the analytic posterior the belief
    /// sampler needs. Errors if `env` is mid-deployment (search is move-phase
    /// only). The session is independent of the live sim — driving it does not
    /// perturb self-play.
    fn search_root(&self, env: usize) -> PyResult<Searcher> {
        let (board, to_play) = self.sim.move_root(env).ok_or_else(|| {
            PyValueError::new_err("env is mid-deployment; search is move-phase only")
        })?;
        let info = RootInfo::from_board(&board, to_play);
        Ok(Searcher {
            board,
            info,
            cfg: *self.sim.config(),
            rollouts: None,
            pending: None,
        })
    }

    /// Run the codebase [`HeuristicBot`] on each requested env's live position
    /// and return its chosen move in the 1800-space, so an eval can use the
    /// heuristic as a non-net opponent (build a near-one-hot logit row at the
    /// returned index). Reads the same live arena `collect` snapshots via
    /// [`Simulator::move_root`], independent of the resident `commit` path.
    ///
    /// A mid-deployment, out-of-range, or no-legal-move env yields `-1` (the
    /// heuristic plays the move phase only; deployment stays uniform, matching
    /// [`HeuristicBot`]'s own random legal fill). The bot RNG is seeded
    /// deterministically from the bridge seed and env id, so the opponent is
    /// reproducible across eval runs.
    fn heuristic_move_actions(&self, envs: Vec<i64>) -> Vec<i64> {
        use game_core::{Agent, Game};
        let game = stratego::Stratego;
        envs.iter()
            .map(|&e| {
                if e < 0 || e as usize >= self.sim.num_envs() {
                    return -1;
                }
                let env = e as usize;
                let Some((board, to_play)) = self.sim.move_root(env) else {
                    return -1;
                };
                let state = stratego::State::Play {
                    board: Box::new(board),
                    to_play,
                    flag_captured: None,
                };
                let actions = game.legal_actions(&state);
                if actions.is_empty() {
                    return -1;
                }
                let mut rng = game_core::Rng::new(
                    self.seed
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(env as u64),
                );
                let idx = stratego::HeuristicBot.act(&game, &state, to_play, &mut rng);
                match actions[idx] {
                    stratego::Move::Step(a) => i64::from(a.0),
                    stratego::Move::Place(_) => -1,
                }
            })
            .collect()
    }

    /// A human-readable render of `env`'s current state from `viewer`'s seat (the
    /// deployment prompt, or the board with the opponent's unrevealed pieces
    /// hidden) — the play CLI's board display. Delegates to
    /// [`Simulator::render`], the same render every other front end drives
    /// through.
    fn render(&self, env: usize, viewer: usize) -> PyResult<String> {
        if env >= self.sim.num_envs() {
            return Err(PyValueError::new_err(format!("env {env} out of range")));
        }
        Ok(self.sim.render(env, viewer))
    }
}

/// Decodes a move-phase action index to absolute `(src, dst)` board cells for
/// `player`'s POV — the play CLI's "which action index did the human's typed
/// move correspond to" and "what did the net actually just play" lookups.
#[pyfunction]
fn action_to_srcdst(action: u16, player: usize) -> (usize, usize) {
    stratego::action::Action(action).to_abs(player)
}

/// Encodes an absolute `(src, dst)` orthogonal slide for `player` back into the
/// 1800-slot action space, or `None` if it is not a legal-shaped straight-line
/// move (the play CLI validates the human's typed move against the engine's own
/// legal set separately; this only handles the coordinate encoding).
#[pyfunction]
fn srcdst_to_action(src: usize, dst: usize, player: usize) -> Option<u16> {
    stratego::action::Action::from_abs(src, dst, player).map(|a| a.0)
}

/// A resident test-time search session over one root position. Owns the cloned
/// root board and its hidden-opponent inventory; exposes the belief-sampling
/// inputs (the analytic posterior + count/movability constraints) and drives the
/// belief-determinized depth-`D` rollout batch under a Python move-net policy.
///
/// Search flow (`stratego_trainer/search.py`):
/// 1. read [`root`](Searcher::root) for the legal action set + belief inputs;
/// 2. sample `n_sample` belief assignments + a per-world root action in Python;
/// 3. [`begin`](Searcher::begin) the rollout batch with those worlds;
/// 4. loop [`collect`](Searcher::collect) → move-net forward →
///    [`commit`](Searcher::commit) until [`is_done`](Searcher::is_done);
/// 5. [`finish`](Searcher::finish) → per-world (root action, leaf value) → the
///    MMD closed form and the final sample, in Python.
#[pyclass]
struct Searcher {
    board: stratego::board::Board,
    info: RootInfo,
    cfg: EncoderConfig,
    rollouts: Option<RolloutBatch>,
    /// The decisions from the last [`collect`](Searcher::collect), kept resident
    /// for the matching [`commit`](Searcher::commit) to gather the dense net
    /// logits back down to each world's ragged legal set.
    pending: Option<Vec<RolloutDecision>>,
}

#[pymethods]
impl Searcher {
    /// The root's search inputs:
    /// * `to_play` `int` — the search player.
    /// * `legal` `(1800,)` bool — the root legal-action mask.
    /// * `n_hidden` `int` — number of hidden opponent pieces.
    /// * `hidden_counts` `(12,)` i64 — per-type hidden-opponent supply (the
    ///   combinatorial-uniform / count-mask budget).
    /// * `hidden_has_moved` `(n_hidden,)` bool — movability constraint per piece.
    /// * `hidden_pos_onehot` `(n_hidden, 100)` bool — each hidden piece's
    ///   absolute board cell (row-major POV rank order).
    /// * `marginal` `(n_hidden, 14)` f32 — the analytic opponent-type posterior
    ///   per hidden piece (the MARGINALIZED_UNIFORM sampling marginals).
    fn root<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n_hidden = self.info.n_hidden();
        let mut legal = Array1::<bool>::default(N_ACTION);
        let mask = stratego::rules::legal_mask(&self.board, self.info.to_play);
        for (i, slot) in legal.iter_mut().enumerate() {
            *slot = mask[i];
        }

        let mut hidden_counts = Array1::<i64>::zeros(12);
        for (t, c) in self.info.hidden_counts.iter().enumerate() {
            hidden_counts[t] = i64::from(*c);
        }
        let mut has_moved = Array1::<bool>::default(n_hidden);
        let mut pos_onehot = Array2::<bool>::default((n_hidden, 100));
        for (i, &cell) in self.info.hidden_cells.iter().enumerate() {
            has_moved[i] = self.info.hidden_has_moved[i];
            pos_onehot[(i, cell)] = true;
        }
        let marg = marginal_posterior(&self.board, &self.info);
        let mut marginal = Array2::<f32>::zeros((n_hidden, N_PIECE_TYPE));
        for (i, row) in marg.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                marginal[(i, t)] = v;
            }
        }

        let out = PyDict::new(py);
        out.set_item("to_play", self.info.to_play)?;
        out.set_item("legal", legal.into_pyarray(py))?;
        out.set_item("n_hidden", n_hidden)?;
        out.set_item("hidden_counts", hidden_counts.into_pyarray(py))?;
        out.set_item("hidden_has_moved", has_moved.into_pyarray(py))?;
        out.set_item("hidden_pos_onehot", pos_onehot.into_pyarray(py))?;
        out.set_item("marginal", marginal.into_pyarray(py))?;
        Ok(out)
    }

    /// The move-net token observation of the **true** root (the actual infostate
    /// the acting player sees — opponent pieces stay hidden). `(92, move_feat)`
    /// f32. This is the input for the root behavior policy `log π_bp` of the MMD
    /// closed form (which is over the real position, not a determinization).
    fn root_obs<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f32>> {
        let obs = stratego::encode_tokens(&self.board, self.info.to_play, &self.cfg);
        let feat = self.cfg.num_token_features();
        Array2::from_shape_vec((MOVE_TOKENS, feat), obs)
            .expect("token obs shape")
            .into_pyarray(py)
    }

    /// The ground-truth hidden opponent ranks at the root (`PieceType as u8`), in
    /// the same row-major POV rank order as [`root`](Searcher::root)'s belief
    /// inputs. The `belief = None` "perfect search" ablation determinizes every
    /// world with this exact assignment.
    fn true_hidden<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        let mut out = Array1::<u8>::zeros(self.info.n_hidden());
        for (i, &cell) in self.info.hidden_cells.iter().enumerate() {
            out[i] = self.board.pieces[cell].kind as u8;
        }
        out.into_pyarray(py)
    }

    /// Begin the depth-`D` rollout batch. `assignments` `(num_worlds, n_hidden)`
    /// `uint8` are the per-world sampled opponent type assignments (one
    /// [`PieceType`] value per hidden piece, in [`root`](Searcher::root)'s POV
    /// rank order); `root_actions` `(num_worlds,)` `i64` is the 1800-space root
    /// action each world is seeded with; `seed` deterministically seeds each
    /// world's rollout RNG. `depth` must be even and ≥ 2.
    ///
    /// Each world clones the root, determinizes it with its assignment
    /// ([`assign_opponent_hidden_pieces`](assign_hidden) equivalent), and applies
    /// the seeded root action; the batch is then ready for
    /// [`collect`](Searcher::collect).
    #[pyo3(signature = (assignments, root_actions, depth, seed))]
    fn begin(
        &mut self,
        assignments: PyReadonlyArray2<'_, u8>,
        root_actions: PyReadonlyArray1<'_, i64>,
        depth: usize,
        seed: u64,
    ) -> PyResult<()> {
        let a = assignments.as_array();
        let acts = root_actions.as_array();
        let n_hidden = self.info.n_hidden();
        let num_worlds = a.shape()[0];
        if a.shape()[1] != n_hidden {
            return Err(PyValueError::new_err(format!(
                "assignments: expected (num_worlds, {n_hidden}), got {:?}",
                a.shape()
            )));
        }
        check_len("root_actions", acts.len(), num_worlds)?;
        if depth < 2 || !depth.is_multiple_of(2) {
            return Err(PyValueError::new_err("depth must be even and ≥ 2"));
        }

        let roots: Vec<(stratego::board::Board, u16, u64)> = (0..num_worlds)
            .map(|w| {
                let assignment: Vec<u8> = (0..n_hidden).map(|i| a[(w, i)]).collect();
                let board = assign_hidden(&self.board, &self.info, &assignment);
                let root_action = acts[w] as u16;
                let world_seed = seed
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(w as u64)
                    .wrapping_add(1);
                (board, root_action, world_seed)
            })
            .collect();

        self.rollouts = Some(RolloutBatch::new(
            &roots,
            self.info.to_play,
            depth,
            self.cfg,
        ));
        Ok(())
    }

    /// Whether every rollout forward (the `depth - 1` move forwards plus the leaf
    /// forward) has been committed.
    fn is_done(&self) -> PyResult<bool> {
        Ok(self
            .rollouts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("begin() not called"))?
            .is_done())
    }

    /// Encode every live rollout world's move decision for one batched move-net
    /// forward. Returns a dict:
    /// * `obs` `(num_worlds, 92, move_feat)` f32 — the move-net token input.
    /// * `legal` `(num_worlds, 1800)` bool — per-world legal mask.
    /// * `player` `(num_worlds,)` i64 — acting player.
    /// * `live` `(num_worlds,)` bool — whether this row is a real decision (a
    ///   terminal world yields an inert all-false row the commit pass skips).
    ///
    /// On the leaf forward every live row is the search player's leaf position;
    /// the net's value head supplies the bootstrap.
    fn collect<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let batch = self
            .rollouts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("begin() not called"))?;
        let decisions = batch.collect();
        let n = decisions.len();
        let feat = self.cfg.num_token_features();

        let mut obs = Array3::<f32>::zeros((n, MOVE_TOKENS, feat));
        let mut legal = Array2::<bool>::default((n, N_ACTION));
        let mut player = Array1::<i64>::zeros(n);
        let mut live = Array1::<bool>::default(n);
        for (row, d) in decisions.iter().enumerate() {
            player[row] = d.player as i64;
            live[row] = d.live;
            if !d.live {
                continue;
            }
            obs.slice_mut(numpy::ndarray::s![row, .., ..])
                .as_slice_mut()
                .expect("contiguous row")
                .copy_from_slice(&d.obs);
            for &a in &d.legal {
                legal[(row, a as usize)] = true;
            }
        }

        // Stash the decisions for the matching commit (the legal sets gather the
        // dense net logits back down to each world's ragged option list).
        self.pending = Some(decisions);

        let out = PyDict::new(py);
        out.set_item("obs", obs.into_pyarray(py))?;
        out.set_item("legal", legal.into_pyarray(py))?;
        out.set_item("player", player.into_pyarray(py))?;
        out.set_item("live", live.into_pyarray(py))?;
        Ok(out)
    }

    /// Apply one move-net forward to the live rollout worlds. `logits`
    /// `(num_worlds, 1800)` f32 are the move-net action logits (illegal slots
    /// ignored — the legal mask selects); `values` `(num_worlds,)` f32 are the
    /// scalar value-head outputs (search-player POV `softmax(W/L/D) @ [-1,0,1]`).
    /// Each live world softmax-samples a legal move and advances (or, on the leaf
    /// forward, only latches the bootstrap value). Pair with the last
    /// [`collect`](Searcher::collect).
    fn commit(
        &mut self,
        logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let decisions = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("commit() called before collect()"))?;
        let batch = self
            .rollouts
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("begin() not called"))?;

        let lg = logits.as_array();
        let vals = values.as_array();
        check_shape("logits", lg.shape(), &[decisions.len(), N_ACTION])?;
        check_len("values", vals.len(), decisions.len())?;

        let evals: Vec<Evaluation> = decisions
            .iter()
            .enumerate()
            .map(|(row, d)| {
                let logits = if d.live {
                    let r = lg.row(row);
                    d.legal.iter().map(|&a| r[a as usize]).collect()
                } else {
                    Vec::new()
                };
                Evaluation {
                    logits,
                    value: vals[row],
                    // Search rollouts only ever read `value` (the scalar leaf
                    // bootstrap); `value_probs` is unused here, so a two-hot
                    // stand-in keeps the struct's invariant (`value ==
                    // value_probs @ VALUE_CATEGORIES`) without a real forward.
                    value_probs: stratego::evaluator::two_hot(vals[row]),
                }
            })
            .collect();

        batch.commit(&decisions, &evals);
        Ok(())
    }

    /// The completed rollout's per-world `(root_action, leaf_value)`. Returns a
    /// dict: `root_action` `(num_worlds,)` i64 and `leaf` `(num_worlds,)` f32 —
    /// each world's λ-return leaf value (terminal reward or value-head bootstrap)
    /// in the search player's POV. Feed these to the per-action scatter +
    /// scalar_q + MMD closed form (in Python).
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let batch = self
            .rollouts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("begin() not called"))?;
        let leaves = batch.finish();
        let n = leaves.len();
        let mut root_action = Array1::<i64>::zeros(n);
        let mut leaf = Array1::<f32>::zeros(n);
        for (i, (a, v)) in leaves.iter().enumerate() {
            root_action[i] = i64::from(*a);
            leaf[i] = *v;
        }
        let out = PyDict::new(py);
        out.set_item("root_action", root_action.into_pyarray(py))?;
        out.set_item("leaf", leaf.into_pyarray(py))?;
        Ok(out)
    }
}

fn check_shape(name: &str, got: &[usize], want: &[usize]) -> PyResult<()> {
    if got != want {
        return Err(PyValueError::new_err(format!(
            "{name}: expected shape {want:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn check_len(name: &str, got: usize, want: usize) -> PyResult<()> {
    if got != want {
        return Err(PyValueError::new_err(format!(
            "{name}: expected length {want}, got {got}"
        )));
    }
    Ok(())
}

/// Re-encode a single stored move-phase board to its `(92, move_feat)` obs — a
/// parity hook for the trainer to cross-check the bridge's obs against the Rust
/// encoder. Returns `None` for non-resident or deploy-phase slots.
#[pyfunction]
fn encode_view_obs<'py>(
    py: Python<'py>,
    sim: &BatchSim,
    env: usize,
    slot: usize,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
    let Some(view) = sim.buffer.encode_view(env, slot) else {
        return Ok(None);
    };
    let Some(t) = sim.buffer.get(env, slot) else {
        return Ok(None);
    };
    if t.phase != Phase::Move {
        return Ok(None);
    }
    let arr = Array2::from_shape_vec((MOVE_TOKENS, sim.move_feat), view.obs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Some(arr.into_pyarray(py)))
}

/// Re-encode the *most recently recorded* transition for `env` via the buffer's
/// independent `encode_view` path. Returns `(obs, is_move)` — `obs` is the
/// `(92, move_feat)` matrix for a move transition (and `None` for a deploy one,
/// whose `(40,14)` deploy obs has a different shape). The parity hook: the obs
/// the live `collect()` produced for this env must byte-match this re-encode.
#[pyfunction]
fn last_move_obs<'py>(
    py: Python<'py>,
    sim: &BatchSim,
    env: usize,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
    let head = sim.buffer.head(env);
    if head == 0 {
        return Ok(None);
    }
    let slot = (head - 1) % sim.buffer.capacity();
    encode_view_obs(py, sim, env, slot)
}

#[pymodule]
fn stratego_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchSim>()?;
    m.add_class::<Searcher>()?;
    m.add_function(wrap_pyfunction!(encode_view_obs, m)?)?;
    m.add_function(wrap_pyfunction!(last_move_obs, m)?)?;
    m.add_function(wrap_pyfunction!(action_to_srcdst, m)?)?;
    m.add_function(wrap_pyfunction!(srcdst_to_action, m)?)?;
    m.add("N_ACTION", N_ACTION)?;
    m.add("DEPLOY_SLOTS", DEPLOY_SLOTS)?;
    m.add("DEPLOY_WIDTH", DEPLOY_WIDTH)?;
    m.add("MOVE_TOKENS", MOVE_TOKENS)?;
    m.add("N_PIECE_TYPE", N_PIECE_TYPE)?;
    Ok(())
}
