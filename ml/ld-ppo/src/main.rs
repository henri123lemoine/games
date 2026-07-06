use std::convert::TryFrom as _;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use game_core::{Agent, Game, RandomAgent, Rng, Turn, winrate_vs_field};
use liars_dice::features::{MAX_DICE_PER, encode, history_encode};
use liars_dice::{
    BidConditioned, DiceShareValue, HistoryNetAgent, LdState, LiarsDice, MAX_FACES, MAX_PLAYERS,
    NetAgent, ProbabilisticAgent, RoundSubgame, feature_len, history_feature_len,
    history_net_policy, legal_actions_and_support, net_policy, policy_len,
};
use ppo_core::{AdvNorm, Minibatch, PpoConfig, Step, UpdateStats};
use solvers::azero::Mlp;
use solvers::{Rollout, nash_conv};
use tch::nn::OptimizerConfig;
use tch::{Device, Kind, Tensor, nn};

#[derive(Clone)]
struct Args {
    players: u8,
    dice: u8,
    faces: u8,
    mixed: bool,
    min_players: u8,
    max_players: u8,
    min_dice: u8,
    max_dice: u8,
    min_faces: u8,
    max_faces: u8,
    iters: usize,
    actors: usize,
    steps: usize,
    max_episode_len: usize,
    hidden: usize,
    lr: f64,
    gamma: f32,
    lambda: f32,
    clip: f64,
    value_coef: f64,
    entropy_coef: f64,
    max_grad_norm: f64,
    epochs: usize,
    minibatches: usize,
    val_every: usize,
    eval_games: u32,
    eval_rollouts: u32,
    eval_exploitability: bool,
    keep_checkpoints: bool,
    device: Device,
    outdir: PathBuf,
    seed: u64,
    /// Feature encoding fed to the net: the flat per-round summary ([`encode`])
    /// or the C13 bid-history-attention variant ([`history_encode`]). History
    /// carries the multi-round bid line the flat encoding collapses away — the
    /// signal an opponent-belief head needs to catch a bluffer, not just an
    /// honest bidder.
    input: InputMode,
    /// Raw `(spec, weight)` opponent-pool entries as given on the CLI; empty
    /// means pure self-play (the historical behavior). Resolving a `Checkpoint`
    /// entry into a loaded net is fallible I/O, so it happens once in
    /// [`build_pool`], not here.
    opponents: Vec<(OpponentSpec, f64)>,
    /// Coefficient for the opponent-hand belief auxiliary loss ([`PpoAdapter::aux_term`]).
    belief_coef: f64,
    /// The resolved pool actors sample opponent seats from each episode.
    /// Starts as pure self-play (matching `opponents` empty); [`main`] replaces
    /// it with [`build_pool`]'s output once `opponents` is parsed.
    pool: Rc<Vec<PoolEntry>>,
    /// Warm-start checkpoint for the trunk/policy/value heads (the belief head
    /// is always freshly initialized — it isn't part of the `Mlp` container).
    /// `None` (default) trains from a fresh random net, matching the historical
    /// behavior. Must match this run's `input` mode and `hidden` width exactly;
    /// a mismatch is a config error, not a silent width/mode coercion.
    init: Option<PathBuf>,
}

/// Which feature encoding — and therefore which net input width — a run uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InputMode {
    Flat,
    History,
}

/// One `opponents=` CLI entry before resolution (parsing is infallible and I/O-free;
/// loading a `Checkpoint` happens in [`build_pool`]).
#[derive(Clone, Debug)]
enum OpponentSpec {
    SelfPlay,
    Belief,
    Rollout(u32),
    Checkpoint(PathBuf),
}

impl OpponentSpec {
    fn label(&self) -> String {
        match self {
            OpponentSpec::SelfPlay => "self".to_string(),
            OpponentSpec::Belief => "belief".to_string(),
            OpponentSpec::Rollout(n) => format!("rollout:{n}"),
            OpponentSpec::Checkpoint(path) => format!("ckpt:{}", path.display()),
        }
    }
}

/// A resolved, ready-to-play pool member and the weight it's sampled with.
struct PoolEntry {
    label: String,
    weight: f64,
    kind: OpponentKind,
}

enum OpponentKind {
    /// Play the live, training net (the historical, only behavior).
    SelfPlay,
    /// Play a fixed agent: a rollout bot, the belief agent, or a frozen
    /// checkpoint — never the net being trained.
    Static(Box<dyn Agent<LiarsDice>>),
}

/// Resolve `cfg.opponents` into ready-to-play pool entries, loading any
/// checkpoint files. Empty `opponents` (the default) resolves to pure
/// self-play, identical to [`Args::default`]'s own pool.
fn build_pool(cfg: &Args) -> io::Result<Vec<PoolEntry>> {
    if cfg.opponents.is_empty() {
        return Ok(self_play_pool());
    }
    let mut pool = Vec::with_capacity(cfg.opponents.len());
    for (spec, weight) in &cfg.opponents {
        let kind = match spec {
            OpponentSpec::SelfPlay => OpponentKind::SelfPlay,
            OpponentSpec::Belief => {
                OpponentKind::Static(Box::new(ProbabilisticAgent::default_agent()))
            }
            OpponentSpec::Rollout(n) => OpponentKind::Static(Box::new(Rollout::new(
                *n,
                ProbabilisticAgent::default_agent(),
                BidConditioned::default(),
            ))),
            OpponentSpec::Checkpoint(path) => {
                let agent: Box<dyn Agent<LiarsDice>> = match cfg.input {
                    InputMode::Flat => Box::new(NetAgent::load(path).map_err(|e| {
                        invalid_input(format!("failed to load ckpt '{}': {e}", path.display()))
                    })?),
                    InputMode::History => Box::new(HistoryNetAgent::load(path).map_err(|e| {
                        invalid_input(format!("failed to load ckpt '{}': {e}", path.display()))
                    })?),
                };
                OpponentKind::Static(agent)
            }
        };
        pool.push(PoolEntry {
            label: spec.label(),
            weight: *weight,
            kind,
        });
    }
    Ok(pool)
}

fn self_play_pool() -> Vec<PoolEntry> {
    vec![PoolEntry {
        label: "self".to_string(),
        weight: 1.0,
        kind: OpponentKind::SelfPlay,
    }]
}

impl Default for Args {
    fn default() -> Self {
        Self {
            players: 5,
            dice: 5,
            faces: 6,
            mixed: false,
            min_players: 2,
            max_players: 5,
            min_dice: 2,
            max_dice: MAX_DICE_PER as u8,
            min_faces: 2,
            max_faces: MAX_FACES as u8,
            iters: 200,
            actors: 32,
            steps: 64,
            max_episode_len: 2048,
            hidden: 256,
            lr: 2.5e-4,
            gamma: 0.995,
            lambda: 0.95,
            clip: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 0.5,
            epochs: 4,
            minibatches: 8,
            val_every: 5,
            eval_games: 200,
            eval_rollouts: 48,
            eval_exploitability: true,
            keep_checkpoints: true,
            device: default_device(),
            outdir: PathBuf::from("runs/ld_ppo"),
            seed: 0xAA55_9900,
            input: InputMode::Flat,
            opponents: Vec::new(),
            belief_coef: 0.1,
            pool: Rc::new(self_play_pool()),
            init: None,
        }
    }
}

struct Transition {
    x: Vec<f32>,
    mask: Vec<f32>,
    action: i64,
    player: usize,
    players: u8,
    dice: u8,
    faces: u8,
    reward: f32,
    value: f32,
    terminal: bool,
    truncated: bool,
    log_prob: f32,
    /// Ground-truth opponent-hand belief targets computed at collection time
    /// (the collector has full state; the deployed net never sees it). Flat
    /// `(MAX_PLAYERS, MAX_FACES)`, rotated so slot `k` is `history_encode`'s
    /// seat `k`; self's own slot (`k == 0`) is zero and unsupervised.
    belief_target: Vec<f32>,
    /// `(MAX_PLAYERS,)`, `1.0` for opponent seats to supervise (alive, `k != 0`).
    belief_seat_mask: Vec<f32>,
    /// `(MAX_FACES,)` additive mask, `0.0` for `f < faces` else `-1e9` — masks
    /// out face bins the game config doesn't use, shared across all seats.
    belief_face_mask: Vec<f32>,
}

/// Ground-truth opponent-hand targets for the belief head, computed from the
/// collector's full state (never seen by the deployed policy). Per-seat
/// targets are the seat's own face histogram normalized by its dice count —
/// a distribution a per-seat softmax cross-entropy can train against directly.
fn belief_targets(
    game: &LiarsDice,
    state: &LdState,
    player: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let p = game.players as usize;
    let dice_left = state.dice_left();
    let mut target = vec![0.0f32; MAX_PLAYERS * MAX_FACES];
    let mut seat_mask = vec![0.0f32; MAX_PLAYERS];
    for k in 1..p.min(MAX_PLAYERS) {
        let seat = (player + k) % p;
        let d = dice_left[seat];
        if d == 0 {
            continue;
        }
        seat_mask[k] = 1.0;
        for f in 0..game.faces as usize {
            target[k * MAX_FACES + f] = f32::from(state.my_count(seat, f as u8 + 1)) / f32::from(d);
        }
    }
    let mut face_mask = vec![0.0f32; MAX_FACES];
    for slot in face_mask.iter_mut().skip(game.faces as usize) {
        *slot = -1.0e9;
    }
    (target, seat_mask, face_mask)
}

#[derive(Clone, Copy, Default)]
struct RolloutStats {
    transitions: usize,
    done_count: usize,
    truncated_count: usize,
    reward_sum: f64,
    players_sum: u64,
    dice_sum: u64,
    faces_sum: u64,
}

struct Actor {
    game: LiarsDice,
    state: LdState,
    perspective: usize,
    decisions: usize,
    /// Per-seat index into `cfg.pool`, sampled fresh each episode: which
    /// opponent (or the live net, for the perspective seat's own — unused —
    /// slot) plays each seat this episode.
    seat_pool: [usize; MAX_PLAYERS],
}

impl Actor {
    fn new(cfg: &Args, rng: &mut Rng) -> Self {
        let game = sample_game(cfg, rng);
        let state = game.initial_state();
        let perspective = rng.below(game.players as usize);
        let seat_pool = sample_seat_pool(cfg, &game, perspective, rng);
        Self {
            game,
            state,
            perspective,
            decisions: 0,
            seat_pool,
        }
    }

    fn reset(&mut self, cfg: &Args, rng: &mut Rng) {
        self.game = sample_game(cfg, rng);
        self.state = self.game.initial_state();
        self.perspective = rng.below(self.game.players as usize);
        self.seat_pool = sample_seat_pool(cfg, &self.game, self.perspective, rng);
        self.decisions = 0;
    }

    fn skip_chance(&mut self, rng: &mut Rng) {
        while !self.game.is_terminal(&self.state)
            && matches!(self.game.turn(&self.state), Turn::Chance)
        {
            let action = self.game.sample_chance_action(&self.state, rng);
            self.game.apply(&mut self.state, action);
        }
    }
}

struct PolicyNet {
    input: usize,
    hidden: usize,
    policy_len: usize,
    fc1: nn::Linear,
    fc2: nn::Linear,
    policy: nn::Linear,
    value: nn::Linear,
    /// Opponent-hand belief head: `(MAX_PLAYERS, MAX_FACES)` logits off the
    /// same trunk. Training-only — never part of the exported [`Mlp`] (the
    /// deployed policy only ever plays off `policy`/`value`), so it's *not*
    /// warm-started by [`Self::load_mlp`]: every run initializes it fresh.
    belief: nn::Linear,
    device: Device,
}

impl PolicyNet {
    fn new(
        path: &nn::Path<'_>,
        input: usize,
        hidden: usize,
        policy_len: usize,
        device: Device,
    ) -> Self {
        let fc1 = nn::linear(
            path / "fc1",
            input as i64,
            hidden as i64,
            Default::default(),
        );
        let fc2 = nn::linear(
            path / "fc2",
            hidden as i64,
            hidden as i64,
            Default::default(),
        );
        let policy = nn::linear(
            path / "policy",
            hidden as i64,
            policy_len as i64,
            Default::default(),
        );
        let value = nn::linear(path / "value", hidden as i64, 1, Default::default());
        let belief = nn::linear(
            path / "belief",
            hidden as i64,
            (MAX_PLAYERS * MAX_FACES) as i64,
            Default::default(),
        );
        Self {
            input,
            hidden,
            policy_len,
            fc1,
            fc2,
            policy,
            value,
            belief,
            device,
        }
    }

    fn from_mlp(path: &nn::Path<'_>, mlp: &Mlp, device: Device) -> Self {
        let mut net = Self::new(
            path,
            mlp.input_len(),
            mlp.hidden_len(),
            mlp.policy_len(),
            device,
        );
        net.load_mlp(mlp);
        net
    }

    fn trunk(&self, x: &Tensor) -> Tensor {
        x.apply(&self.fc1).relu().apply(&self.fc2).relu()
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let h = self.trunk(x);
        let logits = h.apply(&self.policy);
        let value = h.apply(&self.value).tanh().squeeze_dim(-1);
        (logits, value)
    }

    /// Belief-head logits, flat `(B, MAX_PLAYERS * MAX_FACES)` — the caller
    /// reshapes and masks (see [`PpoAdapter::aux_term`]). A second trunk
    /// forward from `evaluate`'s, matching the existing `bc_term` hook's
    /// already-independent-forward design rather than fusing the two.
    fn forward_belief(&self, x: &Tensor) -> Tensor {
        self.trunk(x).apply(&self.belief)
    }

    fn act_one(&self, x: &[f32], mask: &[f32]) -> (Vec<f32>, f32) {
        tch::no_grad(|| {
            let xs = Tensor::from_slice(x)
                .view([1, self.input as i64])
                .to_device(self.device);
            let ms = Tensor::from_slice(mask)
                .view([1, self.policy_len as i64])
                .to_device(self.device);
            let (logits, value) = self.forward(&xs);
            let probs = (logits + ms)
                .softmax(-1, Kind::Float)
                .to_device(Device::Cpu);
            let probs = Vec::<f32>::try_from(probs.flatten(0, -1)).expect("policy probs");
            let value = f32::try_from(value.to_device(Device::Cpu).squeeze()).expect("value");
            (probs, value)
        })
    }

    fn value_one(&self, x: &[f32]) -> f32 {
        tch::no_grad(|| {
            let xs = Tensor::from_slice(x)
                .view([1, self.input as i64])
                .to_device(self.device);
            let (_, value) = self.forward(&xs);
            f32::try_from(value.to_device(Device::Cpu).squeeze()).expect("value")
        })
    }

    fn load_mlp(&mut self, mlp: &Mlp) {
        assert_eq!(mlp.input_len(), self.input);
        assert_eq!(mlp.hidden_len(), self.hidden);
        assert_eq!(mlp.policy_len(), self.policy_len);
        let p = mlp.params();
        let l = layout(self.input, self.hidden, self.policy_len);
        load_linear(
            &mut self.fc1,
            &p[l.w1..l.b1],
            &p[l.b1..l.w2],
            self.hidden,
            self.input,
            self.device,
        );
        load_linear(
            &mut self.fc2,
            &p[l.w2..l.b2],
            &p[l.b2..l.wp],
            self.hidden,
            self.hidden,
            self.device,
        );
        load_linear(
            &mut self.policy,
            &p[l.wp..l.bp],
            &p[l.bp..l.wv],
            self.policy_len,
            self.hidden,
            self.device,
        );
        load_linear(
            &mut self.value,
            &p[l.wv..l.bv],
            &p[l.bv..l.total],
            1,
            self.hidden,
            self.device,
        );
    }

    fn export_mlp(&self) -> Mlp {
        let mut mlp = Mlp::new(self.input, self.hidden, self.policy_len, 0);
        let l = layout(self.input, self.hidden, self.policy_len);
        let p = mlp.params_mut();
        export_linear_into(&self.fc1, p, l.w1, l.b1, l.w2);
        export_linear_into(&self.fc2, p, l.w2, l.b2, l.wp);
        export_linear_into(&self.policy, p, l.wp, l.bp, l.wv);
        export_linear_into(&self.value, p, l.wv, l.bv, l.total);
        mlp
    }
}

struct PpoAdapter<'a> {
    policy: &'a PolicyNet,
    x_all: Tensor,
    mask_all: Tensor,
    action_all: Tensor,
    /// `(N, MAX_PLAYERS * MAX_FACES)` — see [`Transition::belief_target`].
    belief_target: Tensor,
    /// `(N, MAX_PLAYERS)` — see [`Transition::belief_seat_mask`].
    belief_seat_mask: Tensor,
    /// `(N, MAX_FACES)` additive — see [`Transition::belief_face_mask`].
    belief_face_mask: Tensor,
}

impl ppo_core::Policy for PpoAdapter<'_> {
    fn evaluate(&self, idx: &Tensor) -> ppo_core::Eval {
        let x = self.x_all.index_select(0, idx);
        let mask = self.mask_all.index_select(0, idx);
        let actions = self.action_all.index_select(0, idx);
        let (logits, value) = self.policy.forward(&x);
        let log_prob_all = (logits + mask).log_softmax(-1, Kind::Float);
        let probs = log_prob_all.exp();
        let log_prob = log_prob_all
            .gather(1, &actions.unsqueeze(1), false)
            .squeeze_dim(1);
        let entropy = -(&probs * &log_prob_all).sum_dim_intlist(&[1i64][..], false, Kind::Float);
        ppo_core::Eval {
            log_prob,
            entropy,
            value,
        }
    }

    /// Per-seat softmax cross-entropy between the belief head's predicted
    /// face distribution and the ground-truth (opponent-hand) target,
    /// averaged over the seats [`Transition::belief_seat_mask`] marks
    /// supervised, then over the minibatch.
    fn aux_term(&self, idx: &Tensor) -> Option<Tensor> {
        let x = self.x_all.index_select(0, idx);
        let belief_logits =
            self.policy
                .forward_belief(&x)
                .view([-1, MAX_PLAYERS as i64, MAX_FACES as i64]);
        let face_mask = self.belief_face_mask.index_select(0, idx).unsqueeze(1);
        let log_probs = (belief_logits + face_mask).log_softmax(-1, Kind::Float);
        let target = self.belief_target.index_select(0, idx).view([
            -1,
            MAX_PLAYERS as i64,
            MAX_FACES as i64,
        ]);
        let seat_mask = self.belief_seat_mask.index_select(0, idx);
        let ce = -(target * log_probs).sum_dim_intlist(&[2i64][..], false, Kind::Float);
        let seat_count = seat_mask
            .sum_dim_intlist(&[1i64][..], false, Kind::Float)
            .clamp_min(1.0);
        let per_sample =
            (ce * &seat_mask).sum_dim_intlist(&[1i64][..], false, Kind::Float) / seat_count;
        Some(per_sample.mean(Kind::Float))
    }
}

#[derive(Clone, Copy)]
struct Layout {
    w1: usize,
    b1: usize,
    w2: usize,
    b2: usize,
    wp: usize,
    bp: usize,
    wv: usize,
    bv: usize,
    total: usize,
}

fn main() -> io::Result<()> {
    let mut cfg = Args::parse()?;
    validate_config(&cfg)?;
    cfg.pool = Rc::new(build_pool(&cfg)?);
    train(&cfg)
}

/// The starting net for `train()`: a fresh random `Mlp` (the historical
/// default), or a warm-start checkpoint from `cfg.init` — validated to match
/// this run's input width, hidden width, and policy width exactly (a
/// mismatched warm-start is a config error, not something to silently coerce
/// or fall back from). The belief head is never part of the `Mlp` container,
/// so a warm-started run still gets a fresh belief head either way.
fn load_or_init_mlp(cfg: &Args) -> io::Result<Mlp> {
    let Some(path) = &cfg.init else {
        return Ok(Mlp::new(input_len(cfg), cfg.hidden, policy_len(), cfg.seed));
    };
    let loaded = Mlp::load(path).map_err(|e| {
        invalid_input(format!(
            "failed to load init checkpoint '{}': {e}",
            path.display()
        ))
    })?;
    let expected_input = input_len(cfg);
    if loaded.input_len() != expected_input {
        return Err(invalid_input(format!(
            "init checkpoint '{}' has input width {} but this run's input={:?} expects {}",
            path.display(),
            loaded.input_len(),
            cfg.input,
            expected_input
        )));
    }
    if loaded.hidden_len() != cfg.hidden {
        return Err(invalid_input(format!(
            "init checkpoint '{}' has hidden width {} but hidden={} was requested",
            path.display(),
            loaded.hidden_len(),
            cfg.hidden
        )));
    }
    if loaded.policy_len() != policy_len() {
        return Err(invalid_input(format!(
            "init checkpoint '{}' has policy width {} but expected {}",
            path.display(),
            loaded.policy_len(),
            policy_len()
        )));
    }
    Ok(loaded)
}

fn train(cfg: &Args) -> io::Result<()> {
    tch::manual_seed(cfg.seed as i64);
    std::fs::create_dir_all(&cfg.outdir)?;
    let mut metrics = std::fs::File::create(cfg.outdir.join("metrics.jsonl"))?;
    let mut log = std::fs::File::create(cfg.outdir.join("train.log"))?;
    write_ppo_config(&mut metrics, cfg)?;

    let vs = nn::VarStore::new(cfg.device);
    let init = load_or_init_mlp(cfg)?;
    let policy = PolicyNet::from_mlp(&vs.root(), &init, cfg.device);
    let mut opt = nn::Adam::default().build(&vs, cfg.lr).expect("optimizer");
    let mut rng = Rng::new(cfg.seed ^ 0x5050_5050);
    let mut actors: Vec<Actor> = (0..cfg.actors).map(|_| Actor::new(cfg, &mut rng)).collect();

    let pool_desc = pool_desc(cfg);
    println!(
        "ld-ppo: target {}p{}d{}f train={} iters={} actors={} steps={} hidden={} input={:?} pool=[{pool_desc}] belief_coef={} device={:?} eval={}x{} expl={} -> {}",
        cfg.players,
        cfg.dice,
        cfg.faces,
        train_range(cfg),
        cfg.iters,
        cfg.actors,
        cfg.steps,
        cfg.hidden,
        cfg.input,
        cfg.belief_coef,
        cfg.device,
        cfg.eval_rollouts,
        cfg.eval_games,
        cfg.eval_exploitability,
        cfg.outdir.display()
    );

    let run_start = Instant::now();
    let mut best_score = f64::NEG_INFINITY;
    let mut total_decisions = 0u64;
    for iter in 0..cfg.iters {
        let iter_start = Instant::now();
        let rollout_start = Instant::now();
        let (buf, rollout_stats) = collect_rollout(cfg, &policy, &mut actors, &mut rng);
        let rollout_s = rollout_start.elapsed().as_secs_f64();
        total_decisions += buf.len() as u64;
        let boot = bootstrap_values(cfg, &policy, &mut actors, &mut rng);

        let update_start = Instant::now();
        let stats = ppo_update(&policy, &mut opt, cfg, &buf, &boot);
        let update_s = update_start.elapsed().as_secs_f64();

        let exported = policy.export_mlp();
        let iter_done = iter + 1;
        let ckpt_path = cfg.outdir.join("ckpt.bin");
        exported.save(&ckpt_path)?;
        let durable_ckpt = cfg
            .keep_checkpoints
            .then(|| cfg.outdir.join(format!("ckpt_{iter_done}.bin")));
        if let Some(path) = durable_ckpt.as_deref() {
            exported.save(path)?;
        }

        let val_now = iter % cfg.val_every == 0 || iter + 1 == cfg.iters;
        let mut winshares = Vec::new();
        if val_now && cfg.eval_games > 0 {
            winshares = validate_winshares(&exported, cfg, cfg.seed ^ iter as u64);
        }
        let exploitability = if val_now && cfg.eval_exploitability {
            Some(validate_exploitability(&exported, cfg))
        } else {
            None
        };
        let score = if winshares.is_empty() {
            None
        } else {
            Some(winshares.iter().map(|(_, s)| *s).sum::<f64>() / winshares.len() as f64)
        };
        let is_best = if let Some(score) = score {
            if score > best_score {
                best_score = score;
                true
            } else {
                false
            }
        } else {
            iter + 1 == cfg.iters
        };
        if is_best {
            exported.save(&cfg.outdir.join("best.bin"))?;
        }

        let iter_s = iter_start.elapsed().as_secs_f64();
        let mut line = format!(
            "iter {iter:4}  trans {:6}  done {:4}  cfg {:.1}p{:.1}d{:.1}f  rew {:+.3}  ploss {:+.4}  vloss {:.4}  bloss {:.4}  ent {:.3}  kl {:.5}  {iter_s:5.1}s",
            rollout_stats.transitions,
            rollout_stats.done_count,
            rollout_stats.mean_players(),
            rollout_stats.mean_dice(),
            rollout_stats.mean_faces(),
            rollout_stats.mean_reward(),
            stats.policy_loss,
            stats.value_loss,
            stats.aux_loss,
            stats.entropy,
            stats.approx_kl,
        );
        if let Some(score) = score {
            line.push_str(&format!("  eval {score:.3}"));
        }
        if let Some(expl) = exploitability {
            line.push_str(&format!("  expl {expl:.4}"));
        }
        if is_best {
            line.push_str(" *best");
        }
        println!("{line}");
        writeln!(log, "{line}")?;
        log.flush()?;

        write_ppo_iter(
            &mut metrics,
            cfg,
            iter_done,
            total_decisions,
            &rollout_stats,
            &stats,
            rollout_s,
            update_s,
            iter_s,
            run_start.elapsed().as_secs_f64(),
            score,
            exploitability,
            best_score,
            is_best,
            durable_ckpt.as_deref().unwrap_or(&ckpt_path),
            &winshares,
        )?;
        metrics.flush()?;
    }
    Ok(())
}

impl Args {
    fn parse() -> io::Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    fn parse_from<I, S>(raw_args: I) -> io::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cfg = Args::default();
        for raw in raw_args {
            let arg = raw.as_ref();
            let Some((key, value)) = arg.split_once('=') else {
                return Err(invalid_input(format!(
                    "argument must be key=value, got '{arg}'"
                )));
            };
            match key {
                "players" => cfg.players = parse_num(value, key)?,
                "dice" => cfg.dice = parse_num(value, key)?,
                "faces" => cfg.faces = parse_num(value, key)?,
                "mixed" => cfg.mixed = parse_bool(value)?,
                "min_players" => cfg.min_players = parse_num(value, key)?,
                "max_players" => cfg.max_players = parse_num(value, key)?,
                "min_dice" => cfg.min_dice = parse_num(value, key)?,
                "max_dice" => cfg.max_dice = parse_num(value, key)?,
                "min_faces" => cfg.min_faces = parse_num(value, key)?,
                "max_faces" => cfg.max_faces = parse_num(value, key)?,
                "iters" => cfg.iters = parse_num(value, key)?,
                "actors" | "arenas" => cfg.actors = parse_num(value, key)?,
                "steps" => cfg.steps = parse_num(value, key)?,
                "max_episode_len" => cfg.max_episode_len = parse_num(value, key)?,
                "hidden" => cfg.hidden = parse_num(value, key)?,
                "lr" => cfg.lr = parse_num(value, key)?,
                "gamma" => cfg.gamma = parse_num(value, key)?,
                "lambda" => cfg.lambda = parse_num(value, key)?,
                "clip" => cfg.clip = parse_num(value, key)?,
                "value_coef" => cfg.value_coef = parse_num(value, key)?,
                "entropy_coef" => cfg.entropy_coef = parse_num(value, key)?,
                "max_grad_norm" => cfg.max_grad_norm = parse_num(value, key)?,
                "epochs" => cfg.epochs = parse_num(value, key)?,
                "minibatches" => cfg.minibatches = parse_num(value, key)?,
                "val_every" => cfg.val_every = parse_num(value, key)?,
                "eval_games" => cfg.eval_games = parse_num(value, key)?,
                "eval_rollouts" => cfg.eval_rollouts = parse_num(value, key)?,
                "eval_exploitability" => cfg.eval_exploitability = parse_bool(value)?,
                "keep_checkpoints" => cfg.keep_checkpoints = parse_bool(value)?,
                "device" => cfg.device = parse_device(value)?,
                "outdir" | "out" => cfg.outdir = PathBuf::from(value),
                "seed" => cfg.seed = parse_num(value, key)?,
                "input" => cfg.input = parse_input_mode(value)?,
                "init" => {
                    cfg.init = if value == "none" {
                        None
                    } else {
                        Some(PathBuf::from(value))
                    }
                }
                "opponents" => cfg.opponents = parse_opponents(value)?,
                "belief_coef" => cfg.belief_coef = parse_num(value, key)?,
                other => return Err(invalid_input(format!("unknown argument '{other}'"))),
            }
        }
        Ok(cfg)
    }
}

fn validate_config(cfg: &Args) -> io::Result<()> {
    let err = |msg: String| io::Error::new(io::ErrorKind::InvalidInput, msg);
    if cfg.players < 2 {
        return Err(err("players must be at least 2".to_string()));
    }
    if cfg.faces < 2 {
        return Err(err("faces must be at least 2".to_string()));
    }
    if cfg.players as usize > MAX_PLAYERS {
        return Err(err(format!("players must be <= {MAX_PLAYERS}")));
    }
    if cfg.dice as usize > MAX_DICE_PER {
        return Err(err(format!("dice must be <= {MAX_DICE_PER}")));
    }
    if cfg.faces as usize > MAX_FACES {
        return Err(err(format!("faces must be <= {MAX_FACES}")));
    }
    if cfg.iters == 0 || cfg.actors == 0 || cfg.steps == 0 {
        return Err(err("iters, actors, and steps must be positive".to_string()));
    }
    if cfg.epochs == 0 || cfg.minibatches == 0 {
        return Err(err("epochs and minibatches must be positive".to_string()));
    }
    if cfg.minibatches > cfg.actors * cfg.steps {
        return Err(err("minibatches must be <= actors * steps".to_string()));
    }
    if cfg.min_players < 2 || cfg.min_players > cfg.max_players {
        return Err(err("expected 2 <= min_players <= max_players".to_string()));
    }
    if cfg.max_players as usize > MAX_PLAYERS {
        return Err(err(format!("max_players must be <= {MAX_PLAYERS}")));
    }
    if cfg.min_dice == 0 || cfg.min_dice > cfg.max_dice {
        return Err(err("expected 1 <= min_dice <= max_dice".to_string()));
    }
    if cfg.max_dice as usize > MAX_DICE_PER {
        return Err(err(format!("max_dice must be <= {MAX_DICE_PER}")));
    }
    if cfg.min_faces < 2 || cfg.min_faces > cfg.max_faces {
        return Err(err("expected 2 <= min_faces <= max_faces".to_string()));
    }
    if cfg.max_faces as usize > MAX_FACES {
        return Err(err(format!("max_faces must be <= {MAX_FACES}")));
    }
    if cfg.belief_coef < 0.0 {
        return Err(err("belief_coef must be >= 0".to_string()));
    }
    Ok(())
}

fn collect_rollout(
    cfg: &Args,
    policy: &PolicyNet,
    actors: &mut [Actor],
    rng: &mut Rng,
) -> (Vec<Transition>, RolloutStats) {
    let mut buf = Vec::with_capacity(cfg.steps * cfg.actors);
    let mut stats = RolloutStats::default();
    for _ in 0..cfg.steps {
        for actor in actors.iter_mut() {
            let tr = actor_step(cfg, policy, actor, rng);
            debug_assert!(tr.player < tr.players as usize);
            stats.transitions += 1;
            stats.reward_sum += f64::from(tr.reward);
            stats.done_count += usize::from(tr.terminal);
            stats.truncated_count += usize::from(tr.truncated);
            stats.players_sum += u64::from(tr.players);
            stats.dice_sum += u64::from(tr.dice);
            stats.faces_sum += u64::from(tr.faces);
            buf.push(tr);
        }
    }
    (buf, stats)
}

fn actor_step(cfg: &Args, policy: &PolicyNet, actor: &mut Actor, rng: &mut Rng) -> Transition {
    loop {
        advance_to_perspective(cfg, policy, actor, rng);
        if !actor.game.is_terminal(&actor.state) && actor.decisions < cfg.max_episode_len {
            break;
        }
        actor.reset(cfg, rng);
    }
    let Turn::Player(player) = actor.game.turn(&actor.state) else {
        unreachable!("actor must be at its perspective decision")
    };
    debug_assert_eq!(player, actor.perspective);
    let (actions, support) = legal_actions_and_support(&actor.game, &actor.state);
    let x = encode_for(cfg, &actor.game, &actor.state, player);
    let (belief_target, belief_seat_mask, belief_face_mask) =
        belief_targets(&actor.game, &actor.state, player);
    let mask = support_mask(&support);
    let players = actor.game.players;
    let dice = actor.game.dice;
    let faces = actor.game.faces;
    let (probs, value) = policy.act_one(&x, &mask);
    let weights: Vec<f64> = support.iter().map(|&idx| f64::from(probs[idx])).collect();
    let chosen = rng.pick(&weights);
    let action_idx = support[chosen];
    let log_prob = probs[action_idx].max(1e-12).ln();
    actor.game.apply(&mut actor.state, actions[chosen]);
    actor.decisions += 1;

    advance_to_perspective(cfg, policy, actor, rng);
    let terminal = actor.game.is_terminal(&actor.state);
    let truncated = !terminal && actor.decisions >= cfg.max_episode_len;
    let reward = if terminal {
        actor.game.returns(&actor.state, player) as f32
    } else {
        0.0
    };
    if terminal || truncated {
        actor.reset(cfg, rng);
    }
    Transition {
        x,
        mask,
        action: action_idx as i64,
        player,
        players,
        dice,
        faces,
        reward,
        value,
        terminal,
        truncated,
        log_prob,
        belief_target,
        belief_seat_mask,
        belief_face_mask,
    }
}

fn advance_to_perspective(cfg: &Args, policy: &PolicyNet, actor: &mut Actor, rng: &mut Rng) {
    loop {
        actor.skip_chance(rng);
        if actor.game.is_terminal(&actor.state) || actor.decisions >= cfg.max_episode_len {
            return;
        }
        let Turn::Player(player) = actor.game.turn(&actor.state) else {
            unreachable!("chance was skipped before advancing actor")
        };
        if player == actor.perspective {
            return;
        }
        let entry = &cfg.pool[actor.seat_pool[player]];
        let action = match &entry.kind {
            OpponentKind::SelfPlay => {
                sample_policy_action(cfg, policy, &actor.game, &actor.state, player, rng)
            }
            OpponentKind::Static(agent) => agent.act(&actor.game, &actor.state, player, rng),
        };
        let actions = actor.game.legal_actions(&actor.state);
        actor.game.apply(&mut actor.state, actions[action]);
        actor.decisions += 1;
    }
}

fn sample_policy_action(
    cfg: &Args,
    policy: &PolicyNet,
    game: &LiarsDice,
    state: &LdState,
    player: usize,
    rng: &mut Rng,
) -> usize {
    let (_actions, support) = legal_actions_and_support(game, state);
    let x = encode_for(cfg, game, state, player);
    let mask = support_mask(&support);
    let (probs, _) = policy.act_one(&x, &mask);
    let weights: Vec<f64> = support.iter().map(|&idx| f64::from(probs[idx])).collect();
    rng.pick(&weights)
}

fn bootstrap_values(
    cfg: &Args,
    policy: &PolicyNet,
    actors: &mut [Actor],
    rng: &mut Rng,
) -> Vec<f32> {
    actors
        .iter_mut()
        .map(|actor| {
            advance_to_perspective(cfg, policy, actor, rng);
            if actor.game.is_terminal(&actor.state) || actor.decisions >= cfg.max_episode_len {
                0.0
            } else {
                let Turn::Player(player) = actor.game.turn(&actor.state) else {
                    return 0.0;
                };
                debug_assert_eq!(player, actor.perspective);
                let x = encode_for(cfg, &actor.game, &actor.state, actor.perspective);
                policy.value_one(&x)
            }
        })
        .collect()
}

fn ppo_update(
    policy: &PolicyNet,
    opt: &mut nn::Optimizer,
    cfg: &Args,
    buf: &[Transition],
    bootstrap_values: &[f32],
) -> UpdateStats {
    let batch = buf.len();
    let input = input_len(cfg);
    let mut x_flat = Vec::with_capacity(batch * input);
    let mut mask_flat = Vec::with_capacity(batch * policy_len());
    let mut belief_target_flat = Vec::with_capacity(batch * MAX_PLAYERS * MAX_FACES);
    let mut belief_seat_mask_flat = Vec::with_capacity(batch * MAX_PLAYERS);
    let mut belief_face_mask_flat = Vec::with_capacity(batch * MAX_FACES);
    for tr in buf {
        x_flat.extend_from_slice(&tr.x);
        mask_flat.extend_from_slice(&tr.mask);
        belief_target_flat.extend_from_slice(&tr.belief_target);
        belief_seat_mask_flat.extend_from_slice(&tr.belief_seat_mask);
        belief_face_mask_flat.extend_from_slice(&tr.belief_face_mask);
    }
    let x_all = Tensor::from_slice(&x_flat)
        .view([batch as i64, input as i64])
        .to_device(cfg.device);
    let mask_all = Tensor::from_slice(&mask_flat)
        .view([batch as i64, policy_len() as i64])
        .to_device(cfg.device);
    let action_all =
        Tensor::from_slice(&buf.iter().map(|t| t.action).collect::<Vec<_>>()).to_device(cfg.device);
    let belief_target = Tensor::from_slice(&belief_target_flat)
        .view([batch as i64, (MAX_PLAYERS * MAX_FACES) as i64])
        .to_device(cfg.device);
    let belief_seat_mask = Tensor::from_slice(&belief_seat_mask_flat)
        .view([batch as i64, MAX_PLAYERS as i64])
        .to_device(cfg.device);
    let belief_face_mask = Tensor::from_slice(&belief_face_mask_flat)
        .view([batch as i64, MAX_FACES as i64])
        .to_device(cfg.device);
    let adapter = PpoAdapter {
        policy,
        x_all,
        mask_all,
        action_all,
        belief_target,
        belief_seat_mask,
        belief_face_mask,
    };
    let steps: Vec<Step> = buf
        .iter()
        .map(|t| Step {
            reward: t.reward,
            value: t.value,
            done: t.terminal || t.truncated,
        })
        .collect();
    let old_log_prob: Vec<f32> = buf.iter().map(|t| t.log_prob).collect();
    let ppo_cfg = PpoConfig {
        gamma: cfg.gamma,
        lambda: cfg.lambda,
        clip: cfg.clip,
        value_coef: cfg.value_coef,
        entropy_coef: cfg.entropy_coef,
        max_grad_norm: cfg.max_grad_norm,
        epochs: cfg.epochs,
        steps: cfg.steps,
        value_clip: true,
        adv_norm: AdvNorm::PerMinibatch,
        minibatch: Minibatch::Shuffled {
            count: cfg.minibatches,
        },
        bc_anchor: None,
        aux_coef: cfg.belief_coef,
    };
    ppo_core::update(
        &adapter,
        opt,
        cfg.device,
        &steps,
        &old_log_prob,
        bootstrap_values,
        &ppo_cfg,
    )
}

fn validate_winshares(net: &Mlp, cfg: &Args, seed: u64) -> Vec<(String, f64)> {
    let game = LiarsDice::new(cfg.players, cfg.dice, cfg.faces);
    let agent: Box<dyn Agent<LiarsDice>> = match cfg.input {
        InputMode::Flat => Box::new(NetAgent::new(clone_mlp(net))),
        InputMode::History => Box::new(HistoryNetAgent::new(clone_mlp(net))),
    };
    let random = RandomAgent;
    let belief = ProbabilisticAgent::default_agent();
    let rollout = Rollout::new(
        cfg.eval_rollouts.max(1),
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    );
    [
        (
            "field_random".to_string(),
            winrate_vs_field(
                &game,
                agent.as_ref(),
                &random,
                cfg.eval_games,
                seed ^ 0x9999,
            ),
        ),
        (
            "field_belief".to_string(),
            winrate_vs_field(
                &game,
                agent.as_ref(),
                &belief,
                cfg.eval_games,
                seed ^ 0xB311EF,
            ),
        ),
        (
            "field_rollout".to_string(),
            winrate_vs_field(
                &game,
                agent.as_ref(),
                &rollout,
                cfg.eval_games,
                seed ^ 0x50110,
            ),
        ),
    ]
    .into()
}

fn validate_exploitability(net: &Mlp, cfg: &Args) -> f64 {
    validate_exploitability_configs(net, cfg, &[(1, 6), (2, 4)])
}

fn validate_exploitability_configs(net: &Mlp, cfg: &Args, configs: &[(u8, u8)]) -> f64 {
    let cache = net.infer_cache();
    let mut sum = 0.0;
    for &(d, f) in configs {
        let feat = LiarsDice::new(2, d, f);
        let mut dice = [0u8; MAX_PLAYERS];
        dice[0] = d;
        dice[1] = d;
        let round = RoundSubgame::new(2, d, f, dice, 0, true, 1, DiceShareValue);
        let policy = |_g: &RoundSubgame<DiceShareValue>, s: &LdState, pl: usize| match cfg.input {
            InputMode::Flat => net_policy(net, &cache, &feat, s, pl),
            InputMode::History => history_net_policy(net, &cache, &feat, s, pl),
        };
        let (_, _, nc) = nash_conv(&round, &policy);
        sum += nc / 2.0;
    }
    sum / configs.len().max(1) as f64
}

fn sample_game(cfg: &Args, rng: &mut Rng) -> LiarsDice {
    if !cfg.mixed {
        return LiarsDice::new(cfg.players, cfg.dice, cfg.faces);
    }
    LiarsDice::new(
        sample_range(rng, cfg.min_players, cfg.max_players),
        sample_range(rng, cfg.min_dice, cfg.max_dice),
        sample_range(rng, cfg.min_faces, cfg.max_faces),
    )
}

fn sample_range(rng: &mut Rng, lo: u8, hi: u8) -> u8 {
    lo + rng.below((hi - lo) as usize + 1) as u8
}

/// Sample which pool entry plays each non-perspective seat this episode.
fn sample_seat_pool(
    cfg: &Args,
    game: &LiarsDice,
    perspective: usize,
    rng: &mut Rng,
) -> [usize; MAX_PLAYERS] {
    let weights: Vec<f64> = cfg.pool.iter().map(|e| e.weight).collect();
    let mut seats = [0usize; MAX_PLAYERS];
    for (seat, slot) in seats.iter_mut().enumerate().take(game.players as usize) {
        if seat != perspective {
            *slot = rng.pick(&weights);
        }
    }
    seats
}

/// The feature vector for `(game, state, player)` under `cfg`'s input mode.
fn encode_for(cfg: &Args, game: &LiarsDice, state: &LdState, player: usize) -> Vec<f32> {
    match cfg.input {
        InputMode::Flat => encode(game, state, player),
        InputMode::History => history_encode(game, state, player),
    }
}

/// The net input width for `cfg`'s input mode.
fn input_len(cfg: &Args) -> usize {
    match cfg.input {
        InputMode::Flat => feature_len(),
        InputMode::History => history_feature_len(),
    }
}

fn support_mask(support: &[usize]) -> Vec<f32> {
    let mut mask = vec![-1.0e9f32; policy_len()];
    for &idx in support {
        mask[idx] = 0.0;
    }
    mask
}

fn layout(input: usize, hidden: usize, policy: usize) -> Layout {
    let w1 = 0;
    let b1 = w1 + hidden * input;
    let w2 = b1 + hidden;
    let b2 = w2 + hidden * hidden;
    let wp = b2 + hidden;
    let bp = wp + policy * hidden;
    let wv = bp + policy;
    let bv = wv + hidden;
    Layout {
        w1,
        b1,
        w2,
        b2,
        wp,
        bp,
        wv,
        bv,
        total: bv + 1,
    }
}

fn load_linear(
    linear: &mut nn::Linear,
    weights: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    device: Device,
) {
    tch::no_grad(|| {
        let w = Tensor::from_slice(weights)
            .view([rows as i64, cols as i64])
            .to_device(device);
        linear.ws.copy_(&w);
        if let Some(bs) = linear.bs.as_mut() {
            let b = Tensor::from_slice(bias).to_device(device);
            bs.copy_(&b);
        }
    });
}

fn export_linear_into(
    linear: &nn::Linear,
    params: &mut [f32],
    weight_start: usize,
    bias_start: usize,
    next_start: usize,
) {
    let w = linear
        .ws
        .to_kind(Kind::Float)
        .to_device(Device::Cpu)
        .flatten(0, -1);
    let b = linear
        .bs
        .as_ref()
        .expect("linear bias")
        .to_kind(Kind::Float)
        .to_device(Device::Cpu)
        .flatten(0, -1);
    params[weight_start..bias_start].copy_from_slice(&Vec::<f32>::try_from(w).expect("weights"));
    params[bias_start..next_start].copy_from_slice(&Vec::<f32>::try_from(b).expect("bias"));
}

impl RolloutStats {
    fn mean_reward(self) -> f64 {
        self.reward_sum / self.transitions.max(1) as f64
    }

    fn mean_players(self) -> f64 {
        self.players_sum as f64 / self.transitions.max(1) as f64
    }

    fn mean_dice(self) -> f64 {
        self.dice_sum as f64 / self.transitions.max(1) as f64
    }

    fn mean_faces(self) -> f64 {
        self.faces_sum as f64 / self.transitions.max(1) as f64
    }
}

fn clone_mlp(net: &Mlp) -> Mlp {
    Mlp::from_bytes(&net.to_bytes()).expect("round-trip clone")
}

fn train_range(cfg: &Args) -> String {
    if cfg.mixed {
        format!(
            "{}-{}p{}-{}d{}-{}f",
            cfg.min_players,
            cfg.max_players,
            cfg.min_dice,
            cfg.max_dice,
            cfg.min_faces,
            cfg.max_faces
        )
    } else {
        format!("{}p{}d{}f", cfg.players, cfg.dice, cfg.faces)
    }
}

fn default_device() -> Device {
    if tch::utils::has_mps() {
        Device::Mps
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

fn parse_input_mode(value: &str) -> io::Result<InputMode> {
    match value {
        "flat" => Ok(InputMode::Flat),
        "history" => Ok(InputMode::History),
        other => Err(invalid_input(format!(
            "input must be flat or history, got '{other}'"
        ))),
    }
}

/// Parse `opponents=spec=weight,spec=weight,...` where `spec` is `self`,
/// `belief`, `rollout:<n>`, or `ckpt:<path>`.
fn parse_opponents(value: &str) -> io::Result<Vec<(OpponentSpec, f64)>> {
    let mut out = Vec::new();
    for raw in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (spec, weight) = raw
            .rsplit_once('=')
            .ok_or_else(|| invalid_input(format!("opponents entry '{raw}' must be spec=weight")))?;
        let weight: f64 = weight
            .parse()
            .map_err(|_| invalid_input(format!("bad weight in opponents entry '{raw}'")))?;
        if weight <= 0.0 {
            return Err(invalid_input(format!(
                "opponents weight must be positive, got '{raw}'"
            )));
        }
        let kind = if spec == "self" {
            OpponentSpec::SelfPlay
        } else if spec == "belief" {
            OpponentSpec::Belief
        } else if let Some(n) = spec.strip_prefix("rollout:") {
            OpponentSpec::Rollout(
                n.parse()
                    .map_err(|_| invalid_input(format!("bad rollout count in '{spec}'")))?,
            )
        } else if let Some(path) = spec.strip_prefix("ckpt:") {
            OpponentSpec::Checkpoint(PathBuf::from(path))
        } else {
            return Err(invalid_input(format!(
                "unknown opponents entry '{spec}' (self,belief,rollout:<n>,ckpt:<path>)"
            )));
        };
        out.push((kind, weight));
    }
    if out.is_empty() {
        return Err(invalid_input(
            "opponents must list at least one entry".to_string(),
        ));
    }
    Ok(out)
}

fn parse_device(value: &str) -> io::Result<Device> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
        "cuda" => Ok(Device::Cuda(0)),
        other => Err(invalid_input(format!(
            "unknown device '{other}' (use cpu, mps, or cuda)"
        ))),
    }
}

fn parse_num<T>(value: &str, key: &str) -> io::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e| invalid_input(format!("failed to parse {key}='{value}': {e}")))
}

fn parse_bool(value: &str) -> io::Result<bool> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected bool, got '{other}'"),
        )),
    }
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn write_ppo_config(metrics: &mut std::fs::File, cfg: &Args) -> io::Result<()> {
    writeln!(
        metrics,
        "{{\"event\":\"ppo_config\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"mixed\":{},\"min_players\":{},\"max_players\":{},\
         \"min_dice\":{},\"max_dice\":{},\"min_faces\":{},\"max_faces\":{},\
         \"iters\":{},\"actors\":{},\"steps\":{},\"max_episode_len\":{},\
         \"hidden\":{},\"lr\":{:.8},\"gamma\":{:.6},\"lambda\":{:.6},\
         \"clip\":{:.6},\"value_coef\":{:.6},\"entropy_coef\":{:.8},\
         \"max_grad_norm\":{:.6},\"epochs\":{},\"minibatches\":{},\
         \"val_every\":{},\"eval_games\":{},\"eval_rollouts\":{},\"eval_exploitability\":{},\
         \"keep_checkpoints\":{},\"device\":\"{:?}\",\"seed\":{},\"outdir\":\"{}\",\
         \"input\":\"{:?}\",\"belief_coef\":{:.6},\"pool\":\"{}\"}}",
        cfg.players,
        cfg.dice,
        cfg.faces,
        cfg.mixed,
        cfg.min_players,
        cfg.max_players,
        cfg.min_dice,
        cfg.max_dice,
        cfg.min_faces,
        cfg.max_faces,
        cfg.iters,
        cfg.actors,
        cfg.steps,
        cfg.max_episode_len,
        cfg.hidden,
        cfg.lr,
        cfg.gamma,
        cfg.lambda,
        cfg.clip,
        cfg.value_coef,
        cfg.entropy_coef,
        cfg.max_grad_norm,
        cfg.epochs,
        cfg.minibatches,
        cfg.val_every,
        cfg.eval_games,
        cfg.eval_rollouts,
        cfg.eval_exploitability,
        cfg.keep_checkpoints,
        cfg.device,
        cfg.seed,
        json_escape(&cfg.outdir.display().to_string()),
        cfg.input,
        cfg.belief_coef,
        json_escape(&pool_desc(cfg)),
    )
}

fn pool_desc(cfg: &Args) -> String {
    cfg.pool
        .iter()
        .map(|e| format!("{}x{}", e.label, e.weight))
        .collect::<Vec<_>>()
        .join(",")
}

#[expect(
    clippy::too_many_arguments,
    reason = "flat JSONL schema is easier for downstream curve tooling"
)]
fn write_ppo_iter(
    metrics: &mut std::fs::File,
    cfg: &Args,
    iters_done: usize,
    total_decisions: u64,
    rollout_stats: &RolloutStats,
    stats: &UpdateStats,
    rollout_wall_s: f64,
    update_wall_s: f64,
    iter_wall_s: f64,
    total_wall_s: f64,
    validation_score: Option<f64>,
    exploitability: Option<f64>,
    best_validation_score: f64,
    is_best: bool,
    checkpoint: &Path,
    winshares: &[(String, f64)],
) -> io::Result<()> {
    let mut fields = vec![
        "\"event\":\"ppo_iter\"".to_string(),
        "\"method\":\"ppo\"".to_string(),
        format!("\"players\":{}", cfg.players),
        format!("\"dice\":{}", cfg.dice),
        format!("\"faces\":{}", cfg.faces),
        format!("\"mixed\":{}", cfg.mixed),
        format!("\"mean_train_players\":{:.6}", rollout_stats.mean_players()),
        format!("\"mean_train_dice\":{:.6}", rollout_stats.mean_dice()),
        format!("\"mean_train_faces\":{:.6}", rollout_stats.mean_faces()),
        format!("\"iters_done\":{iters_done}"),
        format!("\"actors\":{}", cfg.actors),
        format!("\"steps\":{}", cfg.steps),
        format!("\"transitions\":{total_decisions}"),
        format!("\"fresh_transitions\":{}", rollout_stats.transitions),
        format!("\"episodes_done\":{}", rollout_stats.done_count),
        format!("\"mean_reward\":{:.6}", rollout_stats.mean_reward()),
        format!("\"policy_loss\":{:.6}", stats.policy_loss),
        format!("\"value_loss\":{:.6}", stats.value_loss),
        format!("\"belief_loss\":{:.6}", stats.aux_loss),
        format!("\"entropy\":{:.6}", stats.entropy),
        format!("\"approx_kl\":{:.8}", stats.approx_kl),
        format!("\"clip_frac\":{:.6}", stats.clip_frac),
        format!("\"explained_variance\":{:.6}", stats.explained_variance),
        format!("\"rollout_wall_s\":{rollout_wall_s:.6}"),
        format!("\"update_wall_s\":{update_wall_s:.6}"),
        format!("\"iter_wall_s\":{iter_wall_s:.6}"),
        format!("\"total_wall_s\":{total_wall_s:.6}"),
        format!("\"is_best\":{is_best}"),
        format!(
            "\"checkpoint\":\"{}\"",
            json_escape(&checkpoint.display().to_string())
        ),
        format!(
            "\"latest_checkpoint\":\"{}\"",
            json_escape(&cfg.outdir.join("ckpt.bin").display().to_string())
        ),
        format!(
            "\"best_checkpoint\":\"{}\"",
            json_escape(&cfg.outdir.join("best.bin").display().to_string())
        ),
    ];
    if let Some(score) = validation_score {
        fields.push(format!("\"winrate_mean\":{score:.6}"));
    }
    if let Some(expl) = exploitability {
        fields.push(format!("\"exploitability\":{expl:.6}"));
    }
    if best_validation_score.is_finite() {
        fields.push(format!("\"best_winrate_mean\":{best_validation_score:.6}"));
    }
    for (name, share) in winshares {
        fields.push(format!("\"winrate_{}\":{share:.6}", json_key(name)));
    }
    writeln!(metrics, "{{{}}}", fields.join(","))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_key(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_mask_marks_only_legal_indices() {
        let mask = support_mask(&[0, 3, 9]);
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[9], 0.0);
        assert!(mask[1] < -1.0e8);
    }

    #[test]
    fn parser_rejects_unknown_or_malformed_args() {
        assert!(Args::parse_from(["unknown=1"]).is_err());
        assert!(Args::parse_from(["iters"]).is_err());
    }

    #[test]
    fn parser_rejects_bad_values() {
        assert!(Args::parse_from(["iters=nope"]).is_err());
        assert!(Args::parse_from(["keep_checkpoints=maybe"]).is_err());
        assert!(Args::parse_from(["device=metal"]).is_err());
    }

    #[test]
    fn actor_step_preserves_fixed_perspective_between_steps() {
        let cfg = Args {
            players: 3,
            dice: 1,
            faces: 2,
            mixed: false,
            max_episode_len: 256,
            hidden: 8,
            device: Device::Cpu,
            ..Args::default()
        };
        let vs = nn::VarStore::new(Device::Cpu);
        let init = Mlp::new(feature_len(), cfg.hidden, policy_len(), cfg.seed);
        let policy = PolicyNet::from_mlp(&vs.root(), &init, Device::Cpu);
        let mut rng = Rng::new(123);
        let mut actor = Actor::new(&cfg, &mut rng);
        let perspective = actor.perspective;
        for _ in 0..8 {
            let tr = actor_step(&cfg, &policy, &mut actor, &mut rng);
            assert_eq!(tr.player, perspective);
            if tr.terminal || tr.truncated {
                break;
            }
        }
    }

    #[test]
    fn az_mlp_export_preserves_initial_params() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let mlp = Mlp::new(feature_len(), 8, policy_len(), 7);
        let net = PolicyNet::from_mlp(&vs.root(), &mlp, device);
        let out = net.export_mlp();
        assert_eq!(out.input_len(), feature_len());
        assert_eq!(out.hidden_len(), 8);
        assert_eq!(out.policy_len(), policy_len());
        for (&a, &b) in mlp.params().iter().zip(out.params()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn tiny_exploitability_probe_is_finite() {
        let net = Mlp::new(feature_len(), 8, policy_len(), 9);
        let expl = validate_exploitability_configs(&net, &Args::default(), &[(1, 2)]);
        assert!(expl.is_finite());
    }

    #[test]
    fn load_or_init_mlp_defaults_to_fresh_random_net() {
        let cfg = Args {
            hidden: 8,
            ..Args::default()
        };
        let net = load_or_init_mlp(&cfg).unwrap();
        assert_eq!(net.input_len(), feature_len());
        assert_eq!(net.hidden_len(), 8);
        assert_eq!(net.policy_len(), policy_len());
    }

    #[test]
    fn load_or_init_mlp_warm_starts_from_a_matching_checkpoint() {
        let dir = std::env::temp_dir().join(format!("ld_ppo_init_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt_path = dir.join("warm.bin");
        let seed_net = Mlp::new(feature_len(), 8, policy_len(), 42);
        seed_net.save(&ckpt_path).unwrap();

        let cfg = Args {
            hidden: 8,
            init: Some(ckpt_path.clone()),
            ..Args::default()
        };
        let loaded = load_or_init_mlp(&cfg).unwrap();
        for (&a, &b) in seed_net.params().iter().zip(loaded.params()) {
            assert!(
                (a - b).abs() < 1e-6,
                "warm-started params should match the checkpoint exactly"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_or_init_mlp_rejects_a_hidden_width_mismatch() {
        let dir =
            std::env::temp_dir().join(format!("ld_ppo_init_test_mismatch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt_path = dir.join("wrong_hidden.bin");
        Mlp::new(feature_len(), 8, policy_len(), 42)
            .save(&ckpt_path)
            .unwrap();

        let cfg = Args {
            hidden: 16, // does not match the checkpoint's hidden=8
            init: Some(ckpt_path.clone()),
            ..Args::default()
        };
        let Err(err) = load_or_init_mlp(&cfg) else {
            panic!("expected a hidden-width mismatch error")
        };
        assert!(err.to_string().contains("hidden width"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_opponents_accepts_all_kinds() {
        let parsed = parse_opponents("self=1,belief=2,rollout:64=1,ckpt:runs/champ.bin=3").unwrap();
        assert_eq!(parsed.len(), 4);
        assert!(matches!(parsed[0], (OpponentSpec::SelfPlay, w) if w == 1.0));
        assert!(matches!(parsed[1], (OpponentSpec::Belief, w) if w == 2.0));
        assert!(matches!(parsed[2], (OpponentSpec::Rollout(64), w) if w == 1.0));
        assert!(
            matches!(&parsed[3], (OpponentSpec::Checkpoint(p), w) if p.to_str() == Some("runs/champ.bin") && *w == 3.0)
        );
    }

    #[test]
    fn parse_opponents_rejects_bad_input() {
        assert!(parse_opponents("").is_err());
        assert!(parse_opponents("self").is_err(), "missing weight");
        assert!(parse_opponents("self=0").is_err(), "nonpositive weight");
        assert!(parse_opponents("self=-1").is_err(), "negative weight");
        assert!(parse_opponents("mystery=1").is_err(), "unknown spec");
        assert!(
            parse_opponents("rollout:oops=1").is_err(),
            "bad rollout count"
        );
    }

    #[test]
    fn build_pool_defaults_to_self_play_when_opponents_unset() {
        let cfg = Args::default();
        let pool = build_pool(&cfg).unwrap();
        assert_eq!(pool.len(), 1);
        assert!(matches!(pool[0].kind, OpponentKind::SelfPlay));
        assert_eq!(pool[0].weight, 1.0);
    }

    #[test]
    fn build_pool_resolves_belief_and_rollout_entries() {
        let cfg = Args {
            opponents: vec![(OpponentSpec::Belief, 2.0), (OpponentSpec::Rollout(8), 1.0)],
            ..Args::default()
        };
        let pool = build_pool(&cfg).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(matches!(pool[0].kind, OpponentKind::Static(_)));
        assert!(matches!(pool[1].kind, OpponentKind::Static(_)));
    }

    #[test]
    fn belief_targets_mask_self_dead_seats_and_extra_faces() {
        let game = LiarsDice::new(3, 2, 4);
        let mut state = game.initial_state();
        while matches!(game.turn(&state), Turn::Chance) {
            let a = game.sample_chance_action(&state, &mut Rng::new(1));
            game.apply(&mut state, a);
        }
        let (target, seat_mask, face_mask) = belief_targets(&game, &state, 0);
        // Self (slot 0) is never supervised.
        assert_eq!(seat_mask[0], 0.0);
        assert!(target[..4].iter().all(|&v| v == 0.0));
        // Every seat's row sums to 1 (it's a normalized histogram) when alive.
        for k in 1..3 {
            let row: f32 = target[k * MAX_FACES..k * MAX_FACES + 4].iter().sum();
            if seat_mask[k] == 1.0 {
                assert!(
                    (row - 1.0).abs() < 1e-5,
                    "seat {k} row should sum to 1, got {row}"
                );
            }
        }
        // Faces beyond the game's 4 are masked out of the softmax.
        assert!(face_mask[4..].iter().all(|&m| m < -1.0e8));
        assert!(face_mask[..4].iter().all(|&m| m == 0.0));
    }

    #[test]
    fn actor_step_runs_under_a_static_opponent_pool() {
        let cfg = Args {
            players: 3,
            dice: 1,
            faces: 2,
            mixed: false,
            max_episode_len: 256,
            hidden: 8,
            device: Device::Cpu,
            opponents: vec![(OpponentSpec::Belief, 1.0)],
            ..Args::default()
        };
        let pool = build_pool(&cfg).unwrap();
        let cfg = Args {
            pool: Rc::new(pool),
            ..cfg
        };
        let vs = nn::VarStore::new(Device::Cpu);
        let init = Mlp::new(input_len(&cfg), cfg.hidden, policy_len(), cfg.seed);
        let policy = PolicyNet::from_mlp(&vs.root(), &init, Device::Cpu);
        let mut rng = Rng::new(123);
        let mut actor = Actor::new(&cfg, &mut rng);
        let perspective = actor.perspective;
        for _ in 0..8 {
            let tr = actor_step(&cfg, &policy, &mut actor, &mut rng);
            assert_eq!(tr.player, perspective);
            assert!(tr.belief_target.iter().all(|v| v.is_finite()));
            if tr.terminal || tr.truncated {
                break;
            }
        }
    }

    #[test]
    fn ppo_update_trains_belief_head_under_history_input() {
        let cfg = Args {
            players: 3,
            dice: 1,
            faces: 2,
            mixed: false,
            max_episode_len: 64,
            hidden: 8,
            actors: 2,
            steps: 4,
            minibatches: 2,
            device: Device::Cpu,
            input: InputMode::History,
            belief_coef: 1.0,
            ..Args::default()
        };
        let vs = nn::VarStore::new(Device::Cpu);
        let init = Mlp::new(input_len(&cfg), cfg.hidden, policy_len(), cfg.seed);
        let policy = PolicyNet::from_mlp(&vs.root(), &init, Device::Cpu);
        let mut opt = nn::Adam::default().build(&vs, cfg.lr).unwrap();
        let mut rng = Rng::new(7);
        let mut actors: Vec<Actor> = (0..cfg.actors)
            .map(|_| Actor::new(&cfg, &mut rng))
            .collect();
        let (buf, _) = collect_rollout(&cfg, &policy, &mut actors, &mut rng);
        let boot = bootstrap_values(&cfg, &policy, &mut actors, &mut rng);
        let stats = ppo_update(&policy, &mut opt, &cfg, &buf, &boot);
        assert!(stats.aux_loss.is_finite());
        assert!(
            stats.aux_loss >= 0.0,
            "cross-entropy loss can't be negative"
        );
    }
}
