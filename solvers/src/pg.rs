//! REINFORCE policy gradient with a learned value baseline, reusing the
//! azero pieces: the [`Mlp`] (policy head over legal actions + tanh value
//! head) and the [`PolicyValueEncoder`] game knowledge.
//!
//! Episodes are played by sampling the current policy — self-play for
//! two-player zero-sum games (every seat is the learner), plain episodes for
//! single-player games. Each decision is credited with the undiscounted
//! terminal return to the player who moved; the advantage is that return
//! minus the value head's prediction, and the value head itself regresses
//! toward the return (the Monte-Carlo baseline). An entropy bonus keeps the
//! policy stochastic.
//!
//! The [`Mlp`] only knows cross-entropy training, whose logit gradient is
//! `probs − target`. REINFORCE needs the logit gradient
//! `A·(probs − onehot) + β·∂(−H)/∂logits`, so each decision becomes a
//! pseudo-[`Sample`] with `target = probs − desired_gradient` (it still sums
//! to one) — the exact policy gradient drops out of the unmodified backprop,
//! and `z = return` trains the baseline for free.

use game_core::{Agent, Game, Rng, Turn};

use crate::azero::{Mlp, PolicyValueEncoder, Sample, SgdMomentum};

pub struct PgConfig {
    /// Width of the two hidden layers.
    pub hidden: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    /// Entropy-bonus coefficient β (0 disables it).
    pub entropy: f32,
    /// Episodes whose decisions are pooled into one SGD step.
    pub episodes_per_batch: usize,
    /// Episodes longer than this are cut off and scored as zero return.
    pub max_episode_len: usize,
}

impl Default for PgConfig {
    fn default() -> Self {
        Self {
            hidden: 64,
            lr: 0.02,
            momentum: 0.9,
            l2: 1e-4,
            entropy: 0.1,
            episodes_per_batch: 16,
            max_episode_len: 1000,
        }
    }
}

/// Aggregates over one [`Reinforce::train_episodes`] call.
#[derive(Debug, Clone, Copy)]
pub struct PgStats {
    pub episodes: usize,
    /// SGD steps taken.
    pub batches: usize,
    /// Mean terminal return to player 0 per episode.
    pub mean_return: f64,
    /// Mean policy entropy (nats) per decision.
    pub mean_entropy: f64,
    /// Mean value-head (baseline) MSE across batches.
    pub value_mse: f32,
}

struct Step {
    x: Vec<f32>,
    support: Vec<usize>,
    /// Index into `support` of the sampled action.
    chosen: usize,
    player: usize,
    /// Terminal return to `player`, filled in once the episode ends.
    ret: f32,
}

struct Episode {
    steps: Vec<Step>,
    return0: f64,
    entropy_sum: f64,
}

/// REINFORCE trainer. The policy/value net is an azero [`Mlp`]; episodes are
/// sequential and fully reproducible from the seed.
pub struct Reinforce<'a, G: Game, E: PolicyValueEncoder<G>> {
    game: &'a G,
    enc: &'a E,
    cfg: PgConfig,
    net: Mlp,
    opt: SgdMomentum,
    rng: Rng,
    grad: Vec<f32>,
}

impl<'a, G: Game, E: PolicyValueEncoder<G>> Reinforce<'a, G, E> {
    pub fn new(game: &'a G, enc: &'a E, cfg: PgConfig, seed: u64) -> Self {
        let net = Mlp::new(enc.input_len(), cfg.hidden, enc.policy_len(), seed);
        let opt = SgdMomentum::new(cfg.lr, cfg.momentum, cfg.l2);
        Self {
            game,
            enc,
            cfg,
            net,
            opt,
            rng: Rng::new(seed ^ 0x5EED_F00D_CAFE_BABE),
            grad: Vec::new(),
        }
    }

    pub fn net(&self) -> &Mlp {
        &self.net
    }

    pub fn into_net(self) -> Mlp {
        self.net
    }

    /// The current policy as an arena agent, sampling from the softmax.
    pub fn agent(&self) -> PgAgent<'_, E> {
        PgAgent::new(self.enc, &self.net, false)
    }

    /// The current policy as an arena agent, playing the argmax action.
    pub fn greedy_agent(&self) -> PgAgent<'_, E> {
        PgAgent::new(self.enc, &self.net, true)
    }

    /// Plays `n` episodes with the evolving policy, taking one SGD step per
    /// [`PgConfig::episodes_per_batch`] episodes.
    pub fn train_episodes(&mut self, n: usize) -> PgStats {
        let mut pending: Vec<Step> = Vec::new();
        let mut in_batch = 0usize;
        let mut batches = 0usize;
        let mut ret_sum = 0.0f64;
        let mut ent_sum = 0.0f64;
        let mut decisions = 0usize;
        let mut mse_sum = 0.0f64;
        for _ in 0..n {
            let ep = self.play_episode();
            ret_sum += ep.return0;
            ent_sum += ep.entropy_sum;
            decisions += ep.steps.len();
            pending.extend(ep.steps);
            in_batch += 1;
            if in_batch == self.cfg.episodes_per_batch {
                mse_sum += f64::from(self.step_on(&pending));
                pending.clear();
                in_batch = 0;
                batches += 1;
            }
        }
        if !pending.is_empty() {
            mse_sum += f64::from(self.step_on(&pending));
            batches += 1;
        }
        PgStats {
            episodes: n,
            batches,
            mean_return: ret_sum / n.max(1) as f64,
            mean_entropy: if decisions > 0 {
                ent_sum / decisions as f64
            } else {
                0.0
            },
            value_mse: (mse_sum / batches.max(1) as f64) as f32,
        }
    }

    fn play_episode(&mut self) -> Episode {
        let g = self.game;
        let mut s = g.initial_state();
        let mut steps: Vec<Step> = Vec::new();
        let mut entropy_sum = 0.0f64;
        while !g.is_terminal(&s) && steps.len() < self.cfg.max_episode_len {
            match g.turn(&s) {
                Turn::Chance => {
                    let outs = g.chance_outcomes(&s);
                    let mut r = self.rng.unit();
                    let mut chosen = outs[outs.len() - 1].0;
                    for &(a, p) in &outs {
                        if r < p {
                            chosen = a;
                            break;
                        }
                        r -= p;
                    }
                    g.apply(&mut s, chosen);
                }
                Turn::Player(player) => {
                    let actions = g.legal_actions(&s);
                    let x = self.enc.encode_state(g, &s);
                    let support: Vec<usize> = actions
                        .iter()
                        .map(|&a| self.enc.action_index(g, &s, a))
                        .collect();
                    let (q, _) = self.net.policy_value(&x, &support);
                    entropy_sum += f64::from(entropy(&q));
                    let probs: Vec<f64> = q.iter().map(|&p| f64::from(p)).collect();
                    let chosen = self.rng.pick(&probs);
                    g.apply(&mut s, actions[chosen]);
                    steps.push(Step {
                        x,
                        support,
                        chosen,
                        player,
                        ret: 0.0,
                    });
                }
            }
        }
        let terminal = g.is_terminal(&s);
        let z: Vec<f32> = (0..g.num_players())
            .map(|p| {
                if terminal {
                    g.returns(&s, p) as f32
                } else {
                    0.0
                }
            })
            .collect();
        for st in &mut steps {
            st.ret = z[st.player];
        }
        Episode {
            steps,
            return0: f64::from(z.first().copied().unwrap_or(0.0)),
            entropy_sum,
        }
    }

    fn step_on(&mut self, steps: &[Step]) -> f32 {
        let samples: Vec<Sample> = steps.iter().map(|st| self.pseudo_sample(st)).collect();
        let batch: Vec<&Sample> = samples.iter().collect();
        let (_, value_mse) = self.net.grad(&batch, &mut self.grad);
        self.opt.step(&mut self.net, &self.grad);
        value_mse
    }

    /// Encodes one decision as a [`Sample`] whose cross-entropy gradient under
    /// the current net equals the REINFORCE + entropy-bonus logit gradient
    /// (see the module docs), with `z = return` training the baseline.
    fn pseudo_sample(&self, st: &Step) -> Sample {
        let (q, v) = self.net.policy_value(&st.x, &st.support);
        let adv = st.ret - v;
        let h = entropy(&q);
        let policy = st
            .support
            .iter()
            .zip(&q)
            .enumerate()
            .map(|(i, (&k, &qi))| {
                let onehot = if i == st.chosen { 1.0 } else { 0.0 };
                let dl = adv * (qi - onehot) + self.cfg.entropy * qi * (qi.max(1e-12).ln() + h);
                (k, qi - dl)
            })
            .collect();
        Sample {
            x: st.x.clone(),
            policy,
            z: st.ret,
        }
    }
}

fn entropy(q: &[f32]) -> f32 {
    q.iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

/// A policy net as an arena [`Agent`]: softmax over the legal actions, either
/// sampled with the arena's randomness or played greedily (argmax).
pub struct PgAgent<'a, E> {
    enc: &'a E,
    net: &'a Mlp,
    greedy: bool,
}

impl<'a, E> PgAgent<'a, E> {
    pub fn new(enc: &'a E, net: &'a Mlp, greedy: bool) -> Self {
        Self { enc, net, greedy }
    }
}

impl<G: Game, E: PolicyValueEncoder<G>> Agent<G> for PgAgent<'_, E> {
    fn act(&self, game: &G, state: &G::State, _player: usize, rng: &mut Rng) -> usize {
        let actions = game.legal_actions(state);
        let x = self.enc.encode_state(game, state);
        let support: Vec<usize> = actions
            .iter()
            .map(|&a| self.enc.action_index(game, state, a))
            .collect();
        let (q, _) = self.net.policy_value(&x, &support);
        if self.greedy {
            q.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        } else {
            let probs: Vec<f64> = q.iter().map(|&p| f64::from(p)).collect();
            rng.pick(&probs)
        }
    }
}
