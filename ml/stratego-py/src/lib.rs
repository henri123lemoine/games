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
use stratego::{Collected, EncoderConfig, NUM_ACTIONS, ReplayBuffer, Simulator};

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

/// The high-throughput Stratego self-play simulator, exposed to Python.
///
/// `BatchSim(num_envs, move_cap, seed, history_len=32)`.
#[pyclass]
struct BatchSim {
    sim: Simulator,
    buffer: ReplayBuffer,
    move_feat: usize,
    pending: Option<Pending>,
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
    /// * `deploy_logits` `(n_deploy, 14) f32`, `deploy_values` `(n_deploy,) f32`.
    ///
    /// Returns a dict: `terminal` `(num_envs,)` bool (env completed a game this
    /// step) and `reward_pl0` `(num_envs,) f32` (player-0 terminal reward, 0 if
    /// not terminal).
    fn commit<'py>(
        &mut self,
        py: Python<'py>,
        move_logits: PyReadonlyArray2<'py, f32>,
        move_values: PyReadonlyArray1<'py, f32>,
        deploy_logits: PyReadonlyArray2<'py, f32>,
        deploy_values: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("commit() called before collect()"))?;

        let mlogits = move_logits.as_array();
        let mvalues = move_values.as_array();
        let dlogits = deploy_logits.as_array();
        let dvalues = deploy_values.as_array();

        check_shape("move_logits", mlogits.shape(), &[pending.n_move, N_ACTION])?;
        check_len("move_values", mvalues.len(), pending.n_move)?;
        check_shape(
            "deploy_logits",
            dlogits.shape(),
            &[pending.n_deploy, DEPLOY_WIDTH],
        )?;
        check_len("deploy_values", dvalues.len(), pending.n_deploy)?;

        // Gather the dense net logits down to each env's ragged legal set, in env
        // order, building exactly the `Evaluation` the Rust sampler consumes.
        let evals: Vec<Evaluation> = pending
            .envs
            .iter()
            .map(|e| {
                let (logits, value) = match e.phase {
                    Phase::Move => {
                        let row = mlogits.row(e.row);
                        let logits = e.legal.iter().map(|&a| row[a as usize]).collect();
                        (logits, mvalues[e.row])
                    }
                    Phase::Deploy => {
                        let row = dlogits.row(e.row);
                        let logits = e.legal.iter().map(|&t| row[t as usize]).collect();
                        (logits, dvalues[e.row])
                    }
                };
                Evaluation { logits, value }
            })
            .collect();

        let n_envs = self.sim.num_envs();
        let result = self
            .sim
            .commit(&pending.collected, &evals, &mut self.buffer);

        let mut terminal = Array1::<bool>::default(n_envs);
        let mut reward_pl0 = Array1::<f32>::zeros(n_envs);
        for (env, completed) in result.completed.iter().enumerate() {
            if let Some(r) = completed {
                terminal[env] = true;
                reward_pl0[env] = *r as f32;
            }
        }

        let out = PyDict::new(py);
        out.set_item("terminal", terminal.into_pyarray(py))?;
        out.set_item("reward_pl0", reward_pl0.into_pyarray(py))?;
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
    ///   `ret` `(N,) f32` (the λ-return value target).
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
        let feat = self.move_feat;

        let mut obs = Array3::<f32>::zeros((n, MOVE_TOKENS, feat));
        let mut action = Array1::<i64>::zeros(n);
        let mut legal_mask = Array2::<bool>::default((n, N_ACTION));
        let mut old_log_prob = Array1::<f32>::zeros(n);
        let mut data_log_prob = Array2::<f32>::from_elem((n, N_ACTION), f32::NEG_INFINITY);
        let mut value = Array1::<f32>::zeros(n);
        let mut target_value = Array1::<f32>::zeros(n);
        let mut advantage = Array1::<f32>::zeros(n);
        let mut ret = Array1::<f32>::zeros(n);
        let mut player = Array1::<i64>::zeros(n);
        let mut num_moves = Array1::<i64>::zeros(n);
        let mut is_terminating = Array1::<bool>::default(n);

        for (i, &(env, slot, targets)) in rows.iter().enumerate() {
            let t = self.buffer.get(env, slot).expect("resident move slot");
            let view = self.buffer.encode_view(env, slot).expect("resident slot");

            obs.slice_mut(numpy::ndarray::s![i, .., ..])
                .as_slice_mut()
                .expect("contiguous row")
                .copy_from_slice(&view.obs);
            action[i] = t.action as i64;
            for (&a, &lp) in t.legal.iter().zip(t.old_log_probs.iter()) {
                legal_mask[(i, a as usize)] = true;
                data_log_prob[(i, a as usize)] = lp;
            }
            old_log_prob[i] = t.old_log_probs.get(t.chosen).copied().unwrap_or(0.0);
            value[i] = t.value;
            target_value[i] = t.target_value.unwrap_or(t.value);
            advantage[i] = targets.advantage;
            ret[i] = targets.ret;
            player[i] = t.player as i64;
            num_moves[i] = t.num_moves as i64;
            is_terminating[i] = t.is_terminating_action;
        }

        let out = PyDict::new(py);
        out.set_item("obs", obs.into_pyarray(py))?;
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
    m.add_function(wrap_pyfunction!(encode_view_obs, m)?)?;
    m.add_function(wrap_pyfunction!(last_move_obs, m)?)?;
    m.add("N_ACTION", N_ACTION)?;
    m.add("DEPLOY_SLOTS", DEPLOY_SLOTS)?;
    m.add("DEPLOY_WIDTH", DEPLOY_WIDTH)?;
    m.add("MOVE_TOKENS", MOVE_TOKENS)?;
    Ok(())
}
