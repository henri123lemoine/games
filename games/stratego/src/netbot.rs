//! The trained net as a playing agent — ataraxios, the pure_r4 champion,
//! running through `nn_infer`'s torch-free [`StrategoNet`] forward.
//!
//! Both phases are the raw net with softmax sampling, exactly the trainer's
//! eval-time play (`stratego_trainer/eval.py`): no search. During deployment
//! the setup net scores the next placement type over the placed-so-far prefix
//! (the game's [`DeploymentState`](crate::arrangement::DeploymentState)
//! legality — supply budget and flag handedness — is the same mask the net was
//! trained under). During the move phase the move net scores the legal actions
//! through the key-query grid, gathered per action by this module's reduced
//! (lake-free) cell indexing — the inverse of the trainer's
//! `action_map.create_srcdst_to_env_action_index` scatter.
//!
//! Sampling reproduces the eval pipeline's numerical hardening: legal logits
//! clamped to `1e4`, scaled by `1/temperature` (eval default `0.25`), clamped
//! to `60`, then softmax-sampled.

use game_core::rand::pick_weighted;
use game_core::{Agent, Rng};
use nn_infer::StrategoNet;

use crate::action::{Action, NUM_ACTIONS};
use crate::board::{HOME_CELLS, LAKES, NUM_CELLS};
use crate::encode::{
    DEPLOY_TYPE_WIDTH, EncoderConfig, NUM_OCCUPIABLE_CELLS, deploy_obs, encode_tokens,
};
use crate::game::{Move, State, Stratego};

/// The trainer's eval-time sampling temperature (`config.py eval_temperature`).
pub const EVAL_TEMPERATURE: f32 = 0.25;

/// Legal-logit ceiling applied before temperature scaling (`eval.py pre_ceil`).
const PRE_CEIL: f32 = 1e4;
/// Scaled-logit ceiling applied after temperature scaling (`eval.py post_ceil`).
const POST_CEIL: f32 = 60.0;

/// Reduced (lake-free) index per POV cell; `usize::MAX` marks a lake.
fn reduced_cells() -> [usize; NUM_CELLS] {
    let mut table = [usize::MAX; NUM_CELLS];
    let mut next = 0;
    for (cell, slot) in table.iter_mut().enumerate() {
        if !LAKES.contains(&cell) {
            *slot = next;
            next += 1;
        }
    }
    table
}

/// Scatters a `(92, 92)` src-dst policy grid into the 1800-slot env action
/// space; lake-touching slots read `f32::MIN` (the trainer's
/// `ActionLogitMap.apply` fill). Public for the export parity test.
pub fn scatter_grid(grid: &[f32]) -> Vec<f32> {
    let reduced = reduced_cells();
    let mut out = vec![f32::MIN; NUM_ACTIONS];
    for (env, slot) in out.iter_mut().enumerate() {
        // Player 0's POV is the identity, so `to_abs(0)` decodes the env slot
        // straight to POV coordinates.
        let (src, dst) = Action(env as u16).to_abs(0);
        let (rs, rd) = (reduced[src], reduced[dst]);
        if rs != usize::MAX && rd != usize::MAX {
            *slot = grid[rs * NUM_OCCUPIABLE_CELLS + rd];
        }
    }
    out
}

/// The trained-net agent.
pub struct NetBot {
    net: StrategoNet,
    temperature: f32,
    reduced: [usize; NUM_CELLS],
}

impl NetBot {
    /// Parses an `ATRX1` export and checks it matches this game's encoding
    /// widths. Plays at [`EVAL_TEMPERATURE`].
    pub fn from_bytes(bytes: &[u8]) -> Result<NetBot, String> {
        let net = StrategoNet::parse(bytes)?;
        let arch = net.arch();
        let want_in = EncoderConfig::default().num_token_features();
        if arch.mv.in_dim != want_in || arch.mv.tokens != NUM_OCCUPIABLE_CELLS {
            return Err(format!(
                "move net shape {}x{} does not match the encoder's {}x{}",
                arch.mv.tokens, arch.mv.in_dim, NUM_OCCUPIABLE_CELLS, want_in
            ));
        }
        if arch.setup.in_dim != DEPLOY_TYPE_WIDTH || arch.setup.tokens != HOME_CELLS {
            return Err(format!(
                "setup net shape {}x{} does not match deployment's {}x{}",
                arch.setup.tokens, arch.setup.in_dim, HOME_CELLS, DEPLOY_TYPE_WIDTH
            ));
        }
        Ok(NetBot {
            net,
            temperature: EVAL_TEMPERATURE,
            reduced: reduced_cells(),
        })
    }

    pub fn with_temperature(mut self, temperature: f32) -> NetBot {
        self.temperature = temperature;
        self
    }

    /// Access to the parsed net (the parity test and any future search drive
    /// the forward directly).
    pub fn net(&self) -> &StrategoNet {
        &self.net
    }

    /// Clamp, temperature-scale, softmax, sample — the eval pipeline's move
    /// selection over the legal logits.
    fn sample(&self, logits: &mut [f32], rng: &mut Rng) -> usize {
        for l in logits.iter_mut() {
            *l = (*l).min(PRE_CEIL) / self.temperature;
            *l = (*l).min(POST_CEIL);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        pick_weighted(logits.iter().map(|l| f64::from(l - max).exp()), rng)
    }
}

impl Agent<Stratego> for NetBot {
    fn act(&self, game: &Stratego, state: &State, player: usize, rng: &mut Rng) -> usize {
        use game_core::Game;
        let actions = game.legal_actions(state);
        match state {
            State::Deploy { current, .. } => {
                let obs = deploy_obs(current);
                let placed = current.placed.len();
                let type_logits = self.net.setup_forward(&obs[..placed * DEPLOY_TYPE_WIDTH]);
                let mut logits: Vec<f32> = actions
                    .iter()
                    .map(|m| match m {
                        Move::Place(t) => type_logits[*t as usize],
                        Move::Step(_) => unreachable!("deploy phase"),
                    })
                    .collect();
                self.sample(&mut logits, rng)
            }
            State::Play { board, to_play, .. } => {
                debug_assert_eq!(*to_play, player);
                let tokens = encode_tokens(board, player, &EncoderConfig::default());
                let out = self.net.move_forward(&tokens);
                let mut logits: Vec<f32> = actions
                    .iter()
                    .map(|m| match m {
                        Move::Step(a) => {
                            let (src, dst) = a.to_abs(player);
                            let (src, dst) = if player == 1 {
                                (99 - src, 99 - dst)
                            } else {
                                (src, dst)
                            };
                            out.grid[self.reduced[src] * NUM_OCCUPIABLE_CELLS + self.reduced[dst]]
                        }
                        Move::Place(_) => unreachable!("move phase"),
                    })
                    .collect();
                self.sample(&mut logits, rng)
            }
        }
    }
}
