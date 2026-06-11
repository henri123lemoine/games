//! The AlphaZero loop: rayon-parallel self-play with PUCT + root Dirichlet
//! noise, a replay buffer of (encoding, visit distribution, outcome) triples,
//! and minibatch SGD on policy cross-entropy + value MSE + L2.

use std::collections::VecDeque;

use game_core::{Game, Rng, Turn};
use rayon::prelude::*;
use web_time::Instant;

use super::mlp::{Mlp, Sample, SgdMomentum};
use super::puct::{PolicyValueEncoder, Puct, argmax, sample_chance};
use super::rand::mix;

pub struct AzeroConfig {
    pub hidden: usize,
    pub sims: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub root_noise: f32,
    /// Plies played proportionally to visit counts before switching to argmax.
    pub temp_moves: usize,
    /// Self-play games longer than this are cut off and scored as draws.
    pub max_game_len: usize,
    pub games_per_iter: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub batches_per_iter: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
}

impl Default for AzeroConfig {
    fn default() -> Self {
        Self {
            hidden: 256,
            sims: 96,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            root_noise: 0.25,
            temp_moves: 20,
            max_game_len: 200,
            games_per_iter: 30,
            replay_capacity: 60_000,
            batch_size: 64,
            batches_per_iter: 150,
            lr: 0.02,
            momentum: 0.9,
            l2: 1e-4,
        }
    }
}

pub struct IterStats {
    pub games: usize,
    pub decisive: usize,
    pub avg_plies: f32,
    /// Replay buffer size after this iteration.
    pub samples: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub self_play_secs: f32,
    pub train_secs: f32,
}

impl IterStats {
    pub fn total_loss(&self) -> f32 {
        self.policy_loss + self.value_loss
    }
}

/// Self-play + train. Two-player zero-sum games only (see [`Puct`]).
pub struct SelfPlayTrainer<'a, G: Game, E: PolicyValueEncoder<G>> {
    game: &'a G,
    enc: &'a E,
    cfg: AzeroConfig,
    net: Mlp,
    opt: SgdMomentum,
    buffer: VecDeque<Sample>,
}

impl<'a, G: Game, E: PolicyValueEncoder<G>> SelfPlayTrainer<'a, G, E> {
    pub fn new(game: &'a G, enc: &'a E, cfg: AzeroConfig, seed: u64) -> Self {
        let net = Mlp::new(enc.input_len(), cfg.hidden, enc.policy_len(), seed);
        Self::with_net(game, enc, cfg, net)
    }

    /// Resumes from an existing net (e.g. a loaded checkpoint). Optimizer
    /// momentum and the replay buffer start empty.
    pub fn with_net(game: &'a G, enc: &'a E, cfg: AzeroConfig, net: Mlp) -> Self {
        assert_eq!(net.input_len(), enc.input_len());
        assert_eq!(net.policy_len(), enc.policy_len());
        let opt = SgdMomentum::new(cfg.lr, cfg.momentum, cfg.l2);
        Self {
            game,
            enc,
            cfg,
            net,
            opt,
            buffer: VecDeque::new(),
        }
    }

    pub fn net(&self) -> &Mlp {
        &self.net
    }

    pub fn into_net(self) -> Mlp {
        self.net
    }

    /// One iteration: `games_per_iter` self-play games in parallel, then
    /// `batches_per_iter` SGD steps on minibatches from the replay buffer.
    pub fn iterate(&mut self, seed: u64) -> IterStats {
        let sp_start = Instant::now();
        let games = {
            let this: &Self = self;
            (0..this.cfg.games_per_iter)
                .into_par_iter()
                .map(|g| this.self_play(mix(seed, g as u64 + 1)))
                .collect::<Vec<_>>()
        };
        let self_play_secs = sp_start.elapsed().as_secs_f32();

        let n_games = games.len();
        let mut decisive = 0;
        let mut plies = 0usize;
        for (samples, dec) in games {
            decisive += usize::from(dec);
            plies += samples.len();
            self.buffer.extend(samples);
        }
        while self.buffer.len() > self.cfg.replay_capacity {
            self.buffer.pop_front();
        }

        let train_start = Instant::now();
        let mut policy_loss = 0.0f64;
        let mut value_loss = 0.0f64;
        let mut batches = 0usize;
        if !self.buffer.is_empty() {
            let mut rng = Rng::new(mix(seed, 0x7E57_DA7A));
            let mut grad = Vec::new();
            for _ in 0..self.cfg.batches_per_iter {
                let batch: Vec<&Sample> = (0..self.cfg.batch_size)
                    .map(|_| {
                        let i = (rng.unit() * self.buffer.len() as f64) as usize;
                        &self.buffer[i.min(self.buffer.len() - 1)]
                    })
                    .collect();
                let (pl, vl) = self.net.grad_par(&batch, &mut grad);
                self.opt.step(&mut self.net, &grad);
                policy_loss += f64::from(pl);
                value_loss += f64::from(vl);
                batches += 1;
            }
        }
        let nb = batches.max(1) as f64;

        IterStats {
            games: n_games,
            decisive,
            avg_plies: if n_games > 0 {
                plies as f32 / n_games as f32
            } else {
                0.0
            },
            samples: self.buffer.len(),
            policy_loss: (policy_loss / nb) as f32,
            value_loss: (value_loss / nb) as f32,
            self_play_secs,
            train_secs: train_start.elapsed().as_secs_f32(),
        }
    }

    fn self_play(&self, seed: u64) -> (Vec<Sample>, bool) {
        let g = self.game;
        let mut rng = Rng::new(seed);
        let mut puct = Puct::new(g, self.enc, &self.net, self.cfg.sims);
        puct.c_puct = self.cfg.c_puct;
        puct.dirichlet_alpha = self.cfg.dirichlet_alpha;
        puct.root_noise = self.cfg.root_noise;

        let mut s = g.initial_state();
        sample_chance(g, &mut s, &mut rng);
        let mut recs: Vec<Record> = Vec::new();
        while !g.is_terminal(&s) && recs.len() < self.cfg.max_game_len {
            let Turn::Player(player) = g.turn(&s) else {
                unreachable!("chance was sampled");
            };
            let visits = puct.search(&s, &mut rng);
            let actions = g.legal_actions(&s);
            let total: u32 = visits.iter().sum();
            let target = actions
                .iter()
                .zip(&visits)
                .map(|(&a, &n)| (self.enc.action_index(g, &s, a), n as f32 / total as f32))
                .collect();
            recs.push((self.enc.encode_state(g, &s), target, player));
            let choice = if recs.len() <= self.cfg.temp_moves {
                sample_proportional(&visits, &mut rng)
            } else {
                argmax(&visits)
            };
            g.apply(&mut s, actions[choice]);
            sample_chance(g, &mut s, &mut rng);
        }

        let z0 = if g.is_terminal(&s) {
            g.returns(&s, 0)
        } else {
            0.0
        };
        let samples = recs
            .into_iter()
            .map(|(x, policy, player)| Sample {
                x,
                policy,
                z: if player == 0 { z0 as f32 } else { -z0 as f32 },
            })
            .collect();
        (samples, z0 != 0.0)
    }
}

type Record = (Vec<f32>, Vec<(usize, f32)>, usize);

fn sample_proportional(visits: &[u32], rng: &mut Rng) -> usize {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return 0;
    }
    let mut r = rng.unit() * f64::from(total);
    for (i, &n) in visits.iter().enumerate() {
        r -= f64::from(n);
        if r < 0.0 {
            return i;
        }
    }
    visits.len() - 1
}
