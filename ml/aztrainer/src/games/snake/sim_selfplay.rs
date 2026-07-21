//! Batched simultaneous self-play with fixed-depth joint-action backups.
//!
//! This deliberately does not reuse sequential PUCT: every player's strategy
//! is computed from the same public root. The primary method is an Albatross-
//! style Logit/QRE backup over the full joint-action table. A two-player
//! entropic maximin solver and a policy-only ablation share the same pipeline.

use std::str::FromStr;
use std::time::Instant;

use game_core::rand::dirichlet;
use game_core::{Rng, SimultaneousGame, SimultaneousPolicyValueEncoder, SimultaneousTurn};
use snake::battlesnake::{Battlesnake, BoardState, Direction, Rules};
use snake::battlesnake_encode::BattlesnakeEncoder;

use super::sample::Sample;
use crate::net::{EvalRequest, EvalResult, Infer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackupMethod {
    Logit,
    Maximin,
    Policy,
}

impl BackupMethod {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Logit => "logit",
            Self::Maximin => "maximin",
            Self::Policy => "policy",
        }
    }
}

impl FromStr for BackupMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "logit" => Ok(Self::Logit),
            "maximin" => Ok(Self::Maximin),
            "policy" => Ok(Self::Policy),
            _ => Err(format!("unknown backup method {value:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub concurrent: usize,
    pub method: BackupMethod,
    /// Inverse temperature for Logit responses / maximin mirror descent.
    pub rationality: f32,
    pub solve_iters: usize,
    pub damping: f32,
    pub root_noise: f32,
    pub dirichlet_alpha: f64,
    pub sample_turns: u16,
    pub gamma: f32,
    pub max_turns: u16,
    pub safety_mask: bool,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            concurrent: 128,
            method: BackupMethod::Logit,
            rationality: 8.0,
            solve_iters: 32,
            damping: 0.5,
            root_noise: 0.15,
            dirichlet_alpha: 1.0,
            sample_turns: 20,
            gamma: 0.997,
            max_turns: 750,
            safety_mask: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SelfPlayStats {
    pub games: u32,
    pub turns: u64,
    pub draws: u32,
    pub capped: u32,
    pub root_evals: u64,
    pub leaf_evals: u64,
    pub inference_secs: f32,
    pub cpu_secs: f32,
}

impl SelfPlayStats {
    pub fn avg_turns(self) -> f32 {
        if self.games == 0 {
            0.0
        } else {
            self.turns as f32 / self.games as f32
        }
    }
}

struct Record {
    planes: Vec<f32>,
    policy: [f32; 4],
    player: usize,
    q: f32,
    turn: u16,
}

struct Worker<const N: usize> {
    game: Battlesnake<N>,
    state: BoardState<N>,
    rng: Rng,
    records: Vec<Record>,
    game_seed: u64,
}

impl<const N: usize> Worker<N> {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let game_seed = rng.next_u64();
        let game = Battlesnake::new(Rules {
            seed: game_seed,
            ..Rules::default()
        });
        let state = game.initial_state();
        Self {
            game,
            state,
            rng,
            records: Vec::new(),
            game_seed,
        }
    }

    fn reset(&mut self) {
        self.game_seed = self.rng.next_u64();
        self.game = Battlesnake::new(Rules {
            seed: self.game_seed,
            ..Rules::default()
        });
        self.state = self.game.initial_state();
        self.records.clear();
    }
}

struct RootData<const N: usize> {
    planes: [Option<Vec<f32>>; N],
    priors: [[f32; 4]; N],
    values: [f32; N],
    support: [[bool; 4]; N],
}

impl<const N: usize> RootData<N> {
    fn new() -> Self {
        Self {
            planes: std::array::from_fn(|_| None),
            priors: [[0.0; 4]; N],
            values: [0.0; N],
            support: [[false; 4]; N],
        }
    }
}

struct PayoffTable<const N: usize> {
    joints: Vec<[Direction; N]>,
    values: Vec<[f32; N]>,
    active: [bool; N],
    support: [[bool; 4]; N],
}

type LeafBatch<const N: usize> = (
    Vec<PayoffTable<N>>,
    Vec<EvalRequest>,
    Vec<(usize, usize, usize)>,
);

pub struct SelfPlay<const N: usize> {
    cfg: SelfPlayConfig,
    workers: Vec<Worker<N>>,
    encoder: BattlesnakeEncoder,
}

#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    pub method: BackupMethod,
    pub rationality: f32,
    pub solve_iters: usize,
    pub damping: f32,
}

impl From<SelfPlayConfig> for SolveConfig {
    fn from(config: SelfPlayConfig) -> Self {
        Self {
            method: config.method,
            rationality: config.rationality,
            solve_iters: config.solve_iters,
            damping: config.damping,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EquilibriumResult<const N: usize> {
    pub strategies: [[f32; 4]; N],
    pub values: [f32; N],
}

/// Batched no-noise root solve used by evaluation and deployed neural agents.
pub fn evaluate_states<const N: usize>(
    infer: &Infer,
    games: &[Battlesnake<N>],
    states: &[BoardState<N>],
    config: SolveConfig,
) -> Vec<EquilibriumResult<N>> {
    assert_eq!(games.len(), states.len());
    assert!(config.method != BackupMethod::Maximin || N == 2);
    let encoder = BattlesnakeEncoder;
    let mut roots: Vec<_> = states.iter().map(|_| RootData::new()).collect();
    let mut root_requests = Vec::new();
    let mut root_map = Vec::new();
    for (index, (game, state)) in games.iter().zip(states).enumerate() {
        for player in 0..N {
            if game.is_active(state, player) {
                let planes = encoder.encode_state(game, state, player);
                let support = effective_action_mask(game, state, player, true);
                roots[index].planes[player] = Some(planes.clone());
                roots[index].support[player] = support;
                root_requests.push(EvalRequest {
                    features: planes,
                    support: support_indices(&support),
                });
                root_map.push((index, player));
            } else {
                roots[index].priors[player][Direction::Up as usize] = 1.0;
                roots[index].values[player] = -1.0;
                roots[index].support[player][Direction::Up as usize] = true;
            }
        }
    }
    assign_roots(&mut roots, &root_map, infer.forward_batch(&root_requests));
    if config.method == BackupMethod::Policy {
        return roots
            .into_iter()
            .map(|root| EquilibriumResult {
                strategies: root.priors,
                values: root.values,
            })
            .collect();
    }

    let mut tables = Vec::with_capacity(states.len());
    let mut leaf_requests = Vec::new();
    let mut leaf_map = Vec::new();
    for (index, (game, state)) in games.iter().zip(states).enumerate() {
        let joints = joint_actions(game, state, true);
        let mut values = vec![[0.0; N]; joints.len()];
        for (joint_index, joint) in joints.iter().enumerate() {
            let mut child = *state;
            game.apply_joint(&mut child, joint);
            // Common random numbers: every action in this normal-form table
            // sees the same chance stream. Action-index-dependent food noise
            // would bias the equilibrium toward whichever joint happened to
            // receive the luckier rollout.
            let mut rng = Rng::new(mix(
                game.rules().seed,
                game.state_key(state).expect("Battlesnake state key"),
            ));
            resolve_chance(game, &mut child, &mut rng);
            for (player, value) in values[joint_index].iter_mut().enumerate() {
                if game.is_terminal(&child) {
                    *value = game.returns(&child, player) as f32;
                } else if !game.is_active(&child, player) {
                    *value = -1.0;
                } else {
                    leaf_requests.push(EvalRequest {
                        features: encoder.encode_state(game, &child, player),
                        support: vec![0],
                    });
                    leaf_map.push((index, joint_index, player));
                }
            }
        }
        tables.push(PayoffTable {
            joints,
            values,
            active: std::array::from_fn(|player| game.is_active(state, player)),
            support: std::array::from_fn(|player| effective_action_mask(game, state, player, true)),
        });
    }
    assign_leaves(&mut tables, &leaf_map, infer.forward_batch(&leaf_requests));
    roots
        .into_iter()
        .zip(&tables)
        .map(|(root, table)| {
            let (strategies, values) = match config.method {
                BackupMethod::Logit => logit_equilibrium(
                    table,
                    root.priors,
                    config.rationality,
                    config.solve_iters,
                    config.damping,
                ),
                BackupMethod::Maximin => {
                    maximin_equilibrium(table, root.priors, config.rationality, config.solve_iters)
                }
                BackupMethod::Policy => unreachable!(),
            };
            EquilibriumResult { strategies, values }
        })
        .collect()
}

impl<const N: usize> SelfPlay<N> {
    pub fn new(cfg: SelfPlayConfig, seed: u64) -> Self {
        assert!((2..=4).contains(&N));
        assert!(cfg.concurrent > 0);
        assert!(cfg.solve_iters > 0);
        assert!((0.0..=1.0).contains(&cfg.damping));
        assert!((0.0..=1.0).contains(&cfg.root_noise));
        assert!(
            cfg.method != BackupMethod::Maximin || N == 2,
            "maximin is duel-only"
        );
        Self {
            cfg,
            workers: (0..cfg.concurrent)
                .map(|index| Worker::new(mix(seed, index as u64)))
                .collect(),
            encoder: BattlesnakeEncoder,
        }
    }

    pub fn collect(&mut self, infer: &Infer, target: usize) -> (Vec<Sample>, SelfPlayStats) {
        let mut samples = Vec::with_capacity(target + self.cfg.concurrent * N * 100);
        let mut stats = SelfPlayStats::default();
        while samples.len() < target {
            let cycle = Instant::now();
            let (mut roots, root_requests, root_map) = self.root_requests();
            let infer_start = Instant::now();
            let root_results = infer.forward_batch(&root_requests);
            stats.inference_secs += infer_start.elapsed().as_secs_f32();
            stats.root_evals += root_requests.len() as u64;
            assign_roots(&mut roots, &root_map, root_results);
            for (worker, root) in self.workers.iter_mut().zip(&mut roots) {
                add_root_noise(self.cfg, worker, root);
            }

            let tables = if self.cfg.method == BackupMethod::Policy {
                Vec::new()
            } else {
                let (mut tables, leaf_requests, leaf_map) = self.leaf_requests();
                let infer_start = Instant::now();
                let leaf_results = infer.forward_batch(&leaf_requests);
                stats.inference_secs += infer_start.elapsed().as_secs_f32();
                stats.leaf_evals += leaf_requests.len() as u64;
                assign_leaves(&mut tables, &leaf_map, leaf_results);
                tables
            };

            let finished = self.play_roots(&mut roots, &tables);
            for index in finished.into_iter().rev() {
                let worker = &mut self.workers[index];
                let capped = worker.state.turn_number() >= self.cfg.max_turns
                    && !worker.game.is_terminal(&worker.state);
                let terminal = worker.game.is_terminal(&worker.state);
                let draw = terminal && worker.state.alive_count() == 0;
                let end_turn = worker.state.turn_number();
                for record in worker.records.drain(..) {
                    let raw = if capped {
                        capped_value(&worker.state, record.player)
                    } else {
                        worker.game.returns(&worker.state, record.player) as f32
                    };
                    let distance = i32::from(end_turn.saturating_sub(record.turn));
                    samples.push(Sample {
                        planes: record.planes,
                        policy: record
                            .policy
                            .into_iter()
                            .enumerate()
                            .map(|(index, probability)| (index as u16, probability))
                            .collect(),
                        z: raw * self.cfg.gamma.powi(distance),
                        q: record.q,
                    });
                }
                stats.games += 1;
                stats.turns += u64::from(end_turn);
                stats.draws += u32::from(draw);
                stats.capped += u32::from(capped);
                worker.reset();
            }
            stats.cpu_secs += cycle.elapsed().as_secs_f32();
        }
        (samples, stats)
    }

    fn root_requests(&self) -> (Vec<RootData<N>>, Vec<EvalRequest>, Vec<(usize, usize)>) {
        let mut roots: Vec<_> = self.workers.iter().map(|_| RootData::new()).collect();
        let mut requests = Vec::new();
        let mut map = Vec::new();
        for (worker_index, worker) in self.workers.iter().enumerate() {
            debug_assert_eq!(worker.game.turn(&worker.state), SimultaneousTurn::Players);
            for player in 0..N {
                if !worker.game.is_active(&worker.state, player) {
                    roots[worker_index].priors[player][Direction::Up as usize] = 1.0;
                    roots[worker_index].values[player] = -1.0;
                    roots[worker_index].support[player][Direction::Up as usize] = true;
                    continue;
                }
                let planes = self
                    .encoder
                    .encode_state(&worker.game, &worker.state, player);
                let support = effective_action_mask(
                    &worker.game,
                    &worker.state,
                    player,
                    self.cfg.safety_mask,
                );
                roots[worker_index].planes[player] = Some(planes.clone());
                roots[worker_index].support[player] = support;
                requests.push(EvalRequest {
                    features: planes,
                    support: support_indices(&support),
                });
                map.push((worker_index, player));
            }
        }
        (roots, requests, map)
    }

    fn leaf_requests(&self) -> LeafBatch<N> {
        let mut tables = Vec::with_capacity(self.workers.len());
        let mut requests = Vec::new();
        let mut map = Vec::new();
        for (worker_index, worker) in self.workers.iter().enumerate() {
            let joints = joint_actions(&worker.game, &worker.state, self.cfg.safety_mask);
            let mut values = vec![[0.0; N]; joints.len()];
            for (joint_index, joint) in joints.iter().enumerate() {
                let mut child = worker.state;
                worker.game.apply_joint(&mut child, joint);
                let mut chance_rng = Rng::new(mix(
                    worker.game_seed,
                    worker
                        .game
                        .state_key(&worker.state)
                        .expect("Battlesnake state key"),
                ));
                resolve_chance(&worker.game, &mut child, &mut chance_rng);
                for (player, value) in values[joint_index].iter_mut().enumerate() {
                    if worker.game.is_terminal(&child) {
                        *value = worker.game.returns(&child, player) as f32;
                    } else if !worker.game.is_active(&child, player) {
                        *value = -1.0;
                    } else {
                        requests.push(EvalRequest {
                            features: self.encoder.encode_state(&worker.game, &child, player),
                            support: vec![0],
                        });
                        map.push((worker_index, joint_index, player));
                    }
                }
            }
            tables.push(PayoffTable {
                joints,
                values,
                active: std::array::from_fn(|player| worker.game.is_active(&worker.state, player)),
                support: std::array::from_fn(|player| {
                    effective_action_mask(&worker.game, &worker.state, player, self.cfg.safety_mask)
                }),
            });
        }
        (tables, requests, map)
    }

    fn play_roots(&mut self, roots: &mut [RootData<N>], tables: &[PayoffTable<N>]) -> Vec<usize> {
        let mut finished = Vec::new();
        for worker_index in 0..self.workers.len() {
            let worker = &mut self.workers[worker_index];
            let root = &mut roots[worker_index];
            let (strategies, values) = match self.cfg.method {
                BackupMethod::Policy => (root.priors, root.values),
                BackupMethod::Logit => logit_equilibrium(
                    &tables[worker_index],
                    root.priors,
                    self.cfg.rationality,
                    self.cfg.solve_iters,
                    self.cfg.damping,
                ),
                BackupMethod::Maximin => maximin_equilibrium(
                    &tables[worker_index],
                    root.priors,
                    self.cfg.rationality,
                    self.cfg.solve_iters,
                ),
            };
            let mut joint = [Direction::Up; N];
            for player in 0..N {
                if !worker.game.is_active(&worker.state, player) {
                    continue;
                }
                worker.records.push(Record {
                    planes: root.planes[player].take().expect("active player planes"),
                    policy: strategies[player],
                    player,
                    q: values[player],
                    turn: worker.state.turn_number(),
                });
                let action = if worker.state.turn_number() < self.cfg.sample_turns {
                    worker.rng.pick(&strategies[player].map(f64::from))
                } else {
                    argmax(&strategies[player])
                };
                joint[player] = Direction::ALL[action];
            }
            worker.game.apply_joint(&mut worker.state, &joint);
            resolve_chance(&worker.game, &mut worker.state, &mut worker.rng);
            if worker.game.is_terminal(&worker.state)
                || worker.state.turn_number() >= self.cfg.max_turns
            {
                finished.push(worker_index);
            }
        }
        finished
    }
}

fn assign_roots<const N: usize>(
    roots: &mut [RootData<N>],
    map: &[(usize, usize)],
    results: Vec<EvalResult>,
) {
    assert_eq!(map.len(), results.len());
    for (&(worker, player), result) in map.iter().zip(results) {
        let support = roots[worker].support[player];
        assert_eq!(
            support.iter().filter(|&&allowed| allowed).count(),
            result.priors.len()
        );
        for (action, probability) in support
            .into_iter()
            .enumerate()
            .filter(|&(_, allowed)| allowed)
            .zip(result.priors)
        {
            roots[worker].priors[player][action.0] = probability;
        }
        roots[worker].values[player] = result.value;
    }
}

fn assign_leaves<const N: usize>(
    tables: &mut [PayoffTable<N>],
    map: &[(usize, usize, usize)],
    results: Vec<EvalResult>,
) {
    assert_eq!(map.len(), results.len());
    for (&(worker, joint, player), result) in map.iter().zip(results) {
        tables[worker].values[joint][player] = result.value;
    }
}

fn add_root_noise<const N: usize>(
    cfg: SelfPlayConfig,
    worker: &mut Worker<N>,
    root: &mut RootData<N>,
) {
    for player in 0..N {
        if !worker.game.is_active(&worker.state, player) {
            continue;
        }
        let supported: Vec<_> = root.support[player]
            .iter()
            .enumerate()
            .filter_map(|(action, &allowed)| allowed.then_some(action))
            .collect();
        let noise = dirichlet(cfg.dirichlet_alpha, supported.len(), &mut worker.rng);
        for (&action, &sample) in supported.iter().zip(&noise) {
            root.priors[player][action] = (1.0 - cfg.root_noise) * root.priors[player][action]
                + cfg.root_noise * sample as f32;
        }
    }
}

fn resolve_chance<const N: usize>(game: &Battlesnake<N>, state: &mut BoardState<N>, rng: &mut Rng) {
    while !game.is_terminal(state) && game.turn(state) == SimultaneousTurn::Chance {
        let action = game.sample_chance_action(state, rng);
        game.apply_chance(state, action);
    }
}

fn joint_actions<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    safety_mask: bool,
) -> Vec<[Direction; N]> {
    let mut joints = vec![[Direction::Up; N]];
    for player in 0..N {
        if !game.is_active(state, player) {
            continue;
        }
        let prior = joints;
        let support = effective_action_mask(game, state, player, safety_mask);
        joints = Vec::with_capacity(prior.len() * support.iter().filter(|&&safe| safe).count());
        for base in prior {
            for action in Direction::ALL
                .into_iter()
                .filter(|action| support[*action as usize])
            {
                let mut joint = base;
                joint[player] = action;
                joints.push(joint);
            }
        }
    }
    joints
}

fn effective_action_mask<const N: usize>(
    game: &Battlesnake<N>,
    state: &BoardState<N>,
    player: usize,
    safety_mask: bool,
) -> [bool; 4] {
    if !game.is_active(state, player) {
        return [true, false, false, false];
    }
    if !safety_mask {
        return [true; 4];
    }
    let mask = game.nonfatal_action_mask(state, player);
    if mask.iter().any(|&allowed| allowed) {
        mask
    } else {
        // When death is forced, action choice can still decide whether every
        // snake dies on the joint move. Preserve all draw-saving tactics.
        [true; 4]
    }
}

fn support_indices(mask: &[bool; 4]) -> Vec<u16> {
    mask.iter()
        .enumerate()
        .filter_map(|(action, &allowed)| allowed.then_some(action as u16))
        .collect()
}

fn logit_equilibrium<const N: usize>(
    table: &PayoffTable<N>,
    priors: [[f32; 4]; N],
    rationality: f32,
    iterations: usize,
    damping: f32,
) -> ([[f32; 4]; N], [f32; N]) {
    let mut strategy = priors;
    for _ in 0..iterations {
        let previous = strategy;
        for player in 0..N {
            if !table.active[player] {
                strategy[player] = [1.0, 0.0, 0.0, 0.0];
                continue;
            }
            let mut q = [0.0; 4];
            for (joint, payoff) in table.joints.iter().zip(&table.values) {
                let mut probability = 1.0;
                for other in 0..N {
                    if other != player {
                        probability *= previous[other][joint[other] as usize];
                    }
                }
                q[joint[player] as usize] += probability * payoff[player];
            }
            let logits = std::array::from_fn(|action| {
                priors[player][action].max(1e-6).ln() + rationality * q[action]
            });
            let response = masked_softmax(logits, &table.support[player]);
            for action in 0..4 {
                strategy[player][action] =
                    damping * previous[player][action] + (1.0 - damping) * response[action];
            }
            normalize(&mut strategy[player]);
        }
    }
    let value = expected_values(table, &strategy);
    (strategy, value)
}

fn maximin_equilibrium<const N: usize>(
    table: &PayoffTable<N>,
    priors: [[f32; 4]; N],
    rationality: f32,
    iterations: usize,
) -> ([[f32; 4]; N], [f32; N]) {
    assert_eq!(N, 2, "maximin is defined here for zero-sum duels");
    let mut matrix = [[0.0f32; 4]; 4];
    for (joint, payoff) in table.joints.iter().zip(&table.values) {
        matrix[joint[0] as usize][joint[1] as usize] = (payoff[0] - payoff[1]) * 0.5;
    }
    let mut row = priors[0];
    let mut col = priors[1];
    let mut row_average = [0.0; 4];
    let mut col_average = [0.0; 4];
    let mut row_sum = [0.0; 4];
    let mut col_sum = [0.0; 4];
    let eta = rationality / (iterations as f32).sqrt().max(1.0);
    for _ in 0..iterations {
        for action in 0..4 {
            row_sum[action] += (0..4)
                .map(|reply| col[reply] * matrix[action][reply])
                .sum::<f32>();
            col_sum[action] += (0..4)
                .map(|reply| row[reply] * matrix[reply][action])
                .sum::<f32>();
        }
        row = masked_softmax(
            std::array::from_fn(|action| priors[0][action].max(1e-6).ln() + eta * row_sum[action]),
            &table.support[0],
        );
        col = masked_softmax(
            std::array::from_fn(|action| priors[1][action].max(1e-6).ln() - eta * col_sum[action]),
            &table.support[1],
        );
        for action in 0..4 {
            row_average[action] += row[action];
            col_average[action] += col[action];
        }
    }
    normalize(&mut row_average);
    normalize(&mut col_average);
    let mut strategies = priors;
    strategies[0] = row_average;
    strategies[1] = col_average;
    let values = expected_values(table, &strategies);
    (strategies, values)
}

fn expected_values<const N: usize>(table: &PayoffTable<N>, strategies: &[[f32; 4]; N]) -> [f32; N] {
    let mut values = [0.0; N];
    for (joint, payoff) in table.joints.iter().zip(&table.values) {
        let probability: f32 = (0..N)
            .map(|player| strategies[player][joint[player] as usize])
            .product();
        for player in 0..N {
            values[player] += probability * payoff[player];
        }
    }
    values
}

fn softmax(logits: [f32; 4]) -> [f32; 4] {
    let maximum = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits.map(|value| (value - maximum).exp());
    normalize(&mut probabilities);
    probabilities
}

fn masked_softmax(mut logits: [f32; 4], support: &[bool; 4]) -> [f32; 4] {
    for (logit, &allowed) in logits.iter_mut().zip(support) {
        if !allowed {
            *logit = f32::NEG_INFINITY;
        }
    }
    softmax(logits)
}

fn normalize(probabilities: &mut [f32; 4]) {
    let total = probabilities.iter().sum::<f32>().max(1e-12);
    for probability in probabilities {
        *probability /= total;
    }
}

fn capped_value<const N: usize>(state: &BoardState<N>, player: usize) -> f32 {
    if !state.snake(player).is_alive() {
        return -1.0;
    }
    let my_length = state.snake(player).len() as f32;
    let mut enemy_total = 0.0;
    let mut enemies = 0;
    for opponent in 0..N {
        if opponent != player && state.snake(opponent).is_alive() {
            enemy_total += state.snake(opponent).len() as f32;
            enemies += 1;
        }
    }
    let mean = if enemies == 0 {
        0.0
    } else {
        enemy_total / enemies as f32
    };
    ((my_length - mean) / 8.0).tanh() * 0.25
}

fn argmax(values: &[f32; 4]) -> usize {
    (1..4).fold(0, |best, index| {
        if values[index] > values[best] {
            index
        } else {
            best
        }
    })
}

pub fn mix(seed: u64, stream: u64) -> u64 {
    game_core::hash::splitmix64(seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(matrix: [[f32; 4]; 4]) -> PayoffTable<2> {
        let mut joints = Vec::new();
        let mut values = Vec::new();
        for (row, matrix_row) in matrix.iter().enumerate() {
            for (col, &payoff) in matrix_row.iter().enumerate() {
                joints.push([Direction::ALL[row], Direction::ALL[col]]);
                values.push([payoff, -payoff]);
            }
        }
        PayoffTable {
            joints,
            values,
            active: [true; 2],
            support: [[true; 4]; 2],
        }
    }

    #[test]
    fn logit_prefers_a_strictly_dominant_action() {
        let matrix = std::array::from_fn(|row| std::array::from_fn(|_| row as f32));
        let (strategy, _) = logit_equilibrium(&table(matrix), [[0.25; 4]; 2], 8.0, 64, 0.5);
        assert!(strategy[0][3] > 0.95, "{:?}", strategy[0]);
    }

    #[test]
    fn maximin_mixes_matching_pennies() {
        let matrix = [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [-2.0, -2.0, -2.0, -2.0],
            [-2.0, -2.0, -2.0, -2.0],
        ];
        let priors = [[0.49, 0.49, 0.01, 0.01]; 2];
        let (strategy, value) = maximin_equilibrium(&table(matrix), priors, 8.0, 256);
        assert!(
            (strategy[0][0] - strategy[0][1]).abs() < 0.08,
            "{:?}",
            strategy[0]
        );
        assert!(strategy[0][2] + strategy[0][3] < 0.05);
        assert!(value[0].abs() < 0.1, "{value:?}");
    }

    #[test]
    fn joint_enumeration_is_full_and_simultaneous() {
        let game = Battlesnake::<4>::new(Rules::default());
        let state = game.initial_state();
        let joints = joint_actions(&game, &state, true);
        assert_eq!(joints.len(), 256);
        for player in 0..4 {
            for direction in Direction::ALL {
                assert!(joints.iter().any(|joint| joint[player] == direction));
            }
        }
    }
}
