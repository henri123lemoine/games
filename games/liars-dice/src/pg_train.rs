//! Regularized policy-gradient self-play for Liar's Dice.
//!
//! This is the lightweight R-NaD/NeuRD-family contender in the bake-off roster:
//! a shared policy/value [`solvers::azero::Mlp`] plays every seat, trains from
//! Monte-Carlo returns with a value baseline, and adds entropy plus a slowly
//! refreshed anchor-policy penalty to reduce convention collapse. It deliberately
//! writes the same checkpoint shape as the distillation and Deep CFR trainers so
//! `examples/tournament` can evaluate it through `nets=label:path,...`.

use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use game_core::{Game, RandomAgent, Rng, Turn, winrate_vs_field};
use solvers::azero::{Mlp, Sample, SgdMomentum};
use solvers::{Rollout, nash_conv};

use crate::features::{
    MAX_DICE_PER, encode, feature_len, history_encode, history_feature_len, history_net_policy,
    legal_actions_and_support, net_policy, policy_len,
};
use crate::{
    BidConditioned, DiceShareValue, HistoryNetAgent, LdState, LiarsDice, MAX_FACES, MAX_PLAYERS,
    NetAgent, ProbabilisticAgent, RoundSubgame,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PgArchitecture {
    Mlp,
    HistoryAttention,
}

impl PgArchitecture {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mlp" | "standard" | "rnad" => Some(Self::Mlp),
            "history" | "history-attention" | "attention" | "transformer" => {
                Some(Self::HistoryAttention)
            }
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::HistoryAttention => "history-attention",
        }
    }

    pub fn method(self) -> &'static str {
        match self {
            Self::Mlp => "rnad",
            Self::HistoryAttention => "history-rnad",
        }
    }

    fn input_len(self) -> usize {
        match self {
            Self::Mlp => feature_len(),
            Self::HistoryAttention => history_feature_len(),
        }
    }

    fn encode(self, game: &LiarsDice, state: &LdState, player: usize) -> Vec<f32> {
        match self {
            Self::Mlp => encode(game, state, player),
            Self::HistoryAttention => history_encode(game, state, player),
        }
    }
}

#[derive(Clone)]
pub struct PgTrainConfig {
    pub players: u8,
    pub dice: u8,
    pub faces: u8,
    /// Network architecture variant: the standard information-state MLP or the
    /// C13 bid-history attention input variant trained with the same objective.
    pub architecture: PgArchitecture,
    /// If true, sample training episodes across the inclusive config range
    /// below. Evaluation still uses `players` × `dice` × `faces`, the bake-off
    /// target for this run.
    pub mixed: bool,
    pub min_players: u8,
    pub max_players: u8,
    pub min_dice: u8,
    pub max_dice: u8,
    pub min_faces: u8,
    pub max_faces: u8,
    pub iters: usize,
    pub episodes_per_iter: usize,
    pub max_episode_len: usize,
    pub hidden: usize,
    pub batch: usize,
    pub epochs: usize,
    pub lr: f32,
    pub momentum: f32,
    pub l2: f32,
    /// Entropy bonus coefficient.
    pub entropy: f32,
    /// Cross-entropy/KL-style penalty toward the current anchor policy.
    pub anchor: f32,
    /// Refresh the anchor from the current policy every N iterations.
    pub anchor_update: usize,
    /// Clip Monte-Carlo advantages before forming pseudo-gradients.
    pub adv_clip: f32,
    pub val_every: usize,
    /// Final/checkpoint win-share panel. Set to 0 for plumbing-only smokes.
    pub eval_games: u32,
    pub eval_rollouts: u32,
    /// Exact tiny-config exploitability sanity probe. This is diagnostic only;
    /// field win-share remains the primary bake-off score.
    pub eval_exploitability: bool,
    pub keep_checkpoints: bool,
    pub outdir: String,
    pub seed: u64,
}

impl Default for PgTrainConfig {
    fn default() -> Self {
        Self {
            players: 5,
            dice: 5,
            faces: 6,
            architecture: PgArchitecture::Mlp,
            mixed: false,
            min_players: 2,
            max_players: 5,
            min_dice: 2,
            max_dice: MAX_DICE_PER as u8,
            min_faces: 2,
            max_faces: MAX_FACES as u8,
            iters: 200,
            episodes_per_iter: 256,
            max_episode_len: 2048,
            hidden: 256,
            batch: 1024,
            epochs: 2,
            lr: 0.01,
            momentum: 0.9,
            l2: 1e-4,
            entropy: 0.02,
            anchor: 0.05,
            anchor_update: 10,
            adv_clip: 2.0,
            val_every: 5,
            eval_games: 200,
            eval_rollouts: 48,
            eval_exploitability: true,
            keep_checkpoints: true,
            outdir: "runs/ld_rnad".into(),
            seed: 0xAC7A_C71C,
        }
    }
}

struct Step {
    x: Vec<f32>,
    support: Vec<usize>,
    chosen: usize,
    player: usize,
    ret: f32,
}

struct Episode {
    steps: Vec<Step>,
    return0: f64,
    entropy_sum: f64,
}

#[derive(Clone, Copy, Debug)]
struct IterStats {
    episodes: usize,
    decisions: usize,
    mean_return0: f64,
    mean_entropy: f64,
    players_sum: u64,
    dice_sum: u64,
    faces_sum: u64,
}

/// Train a regularized actor-critic self-play net and return the final net.
pub fn train(cfg: &PgTrainConfig) -> std::io::Result<Mlp> {
    validate_config(cfg)?;
    std::fs::create_dir_all(&cfg.outdir)?;
    let log_path = format!("{}/train.log", cfg.outdir);
    let mut log = std::fs::File::create(&log_path)?;
    let metrics_path = format!("{}/metrics.jsonl", cfg.outdir);
    let mut metrics = std::fs::File::create(&metrics_path)?;
    write_pg_config(&mut metrics, cfg)?;

    let mut net = Mlp::new(
        cfg.architecture.input_len(),
        cfg.hidden,
        policy_len(),
        cfg.seed,
    );
    let mut anchor = clone_net(&net);
    let mut opt = SgdMomentum::new(cfg.lr, cfg.momentum, cfg.l2);
    let mut rng = Rng::new(cfg.seed ^ 0x91E0_D1CE);
    let mut grad = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut total_episodes = 0u64;
    let mut total_decisions = 0u64;
    let run_start = Instant::now();

    for iter in 0..cfg.iters {
        let iter_start = Instant::now();
        let cache = net.infer_cache();
        let mut steps = Vec::new();
        let mut stat = IterStats {
            episodes: cfg.episodes_per_iter,
            decisions: 0,
            mean_return0: 0.0,
            mean_entropy: 0.0,
            players_sum: 0,
            dice_sum: 0,
            faces_sum: 0,
        };
        for _ in 0..cfg.episodes_per_iter {
            let game = sample_game(cfg, &mut rng);
            let ep = play_episode(
                cfg.architecture,
                &game,
                &net,
                &cache,
                cfg.max_episode_len,
                &mut rng,
            );
            stat.decisions += ep.steps.len();
            stat.mean_return0 += ep.return0;
            stat.mean_entropy += ep.entropy_sum;
            stat.players_sum += u64::from(game.players);
            stat.dice_sum += u64::from(game.dice);
            stat.faces_sum += u64::from(game.faces);
            steps.extend(ep.steps);
        }
        stat.mean_return0 /= cfg.episodes_per_iter.max(1) as f64;
        stat.mean_entropy /= stat.decisions.max(1) as f64;
        total_episodes += cfg.episodes_per_iter as u64;
        total_decisions += stat.decisions as u64;

        let train_start = Instant::now();
        let mut ce_sum = 0.0f32;
        let mut mse_sum = 0.0f32;
        let mut batches = 0u32;
        for _ in 0..cfg.epochs {
            fisher_yates(&mut steps, &mut rng);
            for chunk in steps.chunks(cfg.batch.max(1)) {
                let samples: Vec<Sample> = chunk
                    .iter()
                    .map(|st| pseudo_sample(&net, &anchor, st, cfg))
                    .collect();
                let refs: Vec<&Sample> = samples.iter().collect();
                let (ce, mse) = net.grad(&refs, &mut grad);
                opt.step(&mut net, &grad);
                ce_sum += ce;
                mse_sum += mse;
                batches += 1;
            }
        }
        let train_s = train_start.elapsed().as_secs_f64();
        let batches = batches.max(1);
        let iter_done = iter + 1;
        let ckpt_path = format!("{}/ckpt.bin", cfg.outdir);
        net.save(Path::new(&ckpt_path))?;
        let durable_ckpt = cfg
            .keep_checkpoints
            .then(|| format!("{}/ckpt_{iter_done}.bin", cfg.outdir));
        if let Some(path) = durable_ckpt.as_deref() {
            net.save(Path::new(path))?;
        }

        let val_now = iter % cfg.val_every == 0 || iter + 1 == cfg.iters;
        let mut winshares = Vec::new();
        if val_now && cfg.eval_games > 0 {
            winshares = validate_winshares(&net, cfg, cfg.seed ^ iter as u64);
        }
        let exploitability = if val_now && cfg.eval_exploitability {
            Some(validate_exploitability(&net, cfg.architecture))
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
            net.save(Path::new(&format!("{}/best.bin", cfg.outdir)))?;
        }
        if cfg.anchor_update > 0 && iter_done.is_multiple_of(cfg.anchor_update) {
            anchor = clone_net(&net);
        }

        let iter_s = iter_start.elapsed().as_secs_f64();
        let mut line = format!(
            "iter {iter:4}  episodes {:5}  decisions {:6}  cfg {:.1}p{:.1}d{:.1}f  ret0 {:+.3}  ent {:.3}  ce {:.4}  mse {:.4}  {iter_s:5.1}s",
            stat.episodes,
            stat.decisions,
            stat.mean_players(),
            stat.mean_dice(),
            stat.mean_faces(),
            stat.mean_return0,
            stat.mean_entropy,
            ce_sum / batches as f32,
            mse_sum / batches as f32,
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
        write_pg_iter(
            &mut metrics,
            cfg,
            iter_done,
            &stat,
            total_episodes,
            total_decisions,
            ce_sum / batches as f32,
            mse_sum / batches as f32,
            train_s,
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
    Ok(net)
}

impl IterStats {
    fn mean_players(self) -> f64 {
        self.players_sum as f64 / self.episodes.max(1) as f64
    }

    fn mean_dice(self) -> f64 {
        self.dice_sum as f64 / self.episodes.max(1) as f64
    }

    fn mean_faces(self) -> f64 {
        self.faces_sum as f64 / self.episodes.max(1) as f64
    }
}

fn validate_config(cfg: &PgTrainConfig) -> io::Result<()> {
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
    if cfg.iters == 0 {
        return Err(err("iters must be positive".to_string()));
    }
    if cfg.episodes_per_iter == 0 {
        return Err(err("episodes_per_iter must be positive".to_string()));
    }
    if cfg.batch == 0 {
        return Err(err("batch must be positive".to_string()));
    }
    if cfg.epochs == 0 {
        return Err(err("epochs must be positive".to_string()));
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
    Ok(())
}

fn sample_game(cfg: &PgTrainConfig, rng: &mut Rng) -> LiarsDice {
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
    debug_assert!(lo <= hi);
    lo + rng.below((hi - lo) as usize + 1) as u8
}

fn play_episode(
    architecture: PgArchitecture,
    game: &LiarsDice,
    net: &Mlp,
    cache: &solvers::azero::InferCache,
    max_episode_len: usize,
    rng: &mut Rng,
) -> Episode {
    let mut state = game.initial_state();
    let mut steps = Vec::new();
    let mut entropy_sum = 0.0f64;
    while !game.is_terminal(&state) && steps.len() < max_episode_len {
        match game.turn(&state) {
            Turn::Chance => {
                let action = game.sample_chance_action(&state, rng);
                game.apply(&mut state, action);
            }
            Turn::Player(player) => {
                let (actions, support) = legal_actions_and_support(game, &state);
                let x = architecture.encode(game, &state, player);
                let (probs, _) = net.policy_value_cached(cache, &x, &support);
                entropy_sum += f64::from(entropy(&probs));
                let weights: Vec<f64> = probs.iter().map(|&p| f64::from(p)).collect();
                let chosen = rng.pick(&weights);
                game.apply(&mut state, actions[chosen]);
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
    let terminal = game.is_terminal(&state);
    let returns: Vec<f32> = (0..game.num_players())
        .map(|p| {
            if terminal {
                game.returns(&state, p) as f32
            } else {
                0.0
            }
        })
        .collect();
    for st in &mut steps {
        st.ret = returns[st.player];
    }
    Episode {
        steps,
        return0: f64::from(returns.first().copied().unwrap_or(0.0)),
        entropy_sum,
    }
}

fn pseudo_sample(net: &Mlp, anchor: &Mlp, st: &Step, cfg: &PgTrainConfig) -> Sample {
    let (q, v) = net.policy_value(&st.x, &st.support);
    let (anchor_q, _) = anchor.policy_value(&st.x, &st.support);
    let adv = (st.ret - v).clamp(-cfg.adv_clip, cfg.adv_clip);
    let h = entropy(&q);
    let policy = st
        .support
        .iter()
        .zip(&q)
        .zip(&anchor_q)
        .enumerate()
        .map(|(i, ((&k, &qi), &ai))| {
            let onehot = if i == st.chosen { 1.0 } else { 0.0 };
            let pg_grad = adv * (qi - onehot);
            let entropy_grad = cfg.entropy * qi * (qi.max(1e-12).ln() + h);
            let anchor_grad = cfg.anchor * (qi - ai);
            (k, qi - pg_grad - entropy_grad - anchor_grad)
        })
        .collect();
    Sample {
        x: st.x.clone(),
        policy,
        z: st.ret,
    }
}

fn validate_winshares(net: &Mlp, cfg: &PgTrainConfig, seed: u64) -> Vec<(String, f64)> {
    match cfg.architecture {
        PgArchitecture::Mlp => {
            validate_winshares_with_agent(cfg, &NetAgent::new(clone_net(net)), seed)
        }
        PgArchitecture::HistoryAttention => {
            validate_winshares_with_agent(cfg, &HistoryNetAgent::new(clone_net(net)), seed)
        }
    }
}

fn validate_winshares_with_agent(
    cfg: &PgTrainConfig,
    agent: &dyn game_core::Agent<LiarsDice>,
    seed: u64,
) -> Vec<(String, f64)> {
    let game = LiarsDice::new(cfg.players, cfg.dice, cfg.faces);
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
            winrate_vs_field(&game, agent, &random, cfg.eval_games, seed ^ 0x9999),
        ),
        (
            "field_belief".to_string(),
            winrate_vs_field(&game, agent, &belief, cfg.eval_games, seed ^ 0xB311EF),
        ),
        (
            "field_rollout".to_string(),
            winrate_vs_field(&game, agent, &rollout, cfg.eval_games, seed ^ 0x50110),
        ),
    ]
    .into()
}

/// Tiny exact best-response probe for silent policy collapse. This intentionally
/// stays on small 2-player rounds; 5p5d6f strength is measured by tournaments.
fn validate_exploitability(net: &Mlp, architecture: PgArchitecture) -> f64 {
    validate_exploitability_configs(net, architecture, &[(1, 6), (2, 4)])
}

fn validate_exploitability_configs(
    net: &Mlp,
    architecture: PgArchitecture,
    configs: &[(u8, u8)],
) -> f64 {
    assert!(!configs.is_empty(), "need at least one probe config");
    let cache = net.infer_cache();
    let mut sum = 0.0;
    for &(d, f) in configs {
        let feat = LiarsDice::new(2, d, f);
        let mut dice = [0u8; MAX_PLAYERS];
        dice[0] = d;
        dice[1] = d;
        let round = RoundSubgame::new(2, d, f, dice, 0, true, 1, DiceShareValue);
        let policy = |_g: &RoundSubgame<DiceShareValue>, s: &LdState, pl: usize| match architecture
        {
            PgArchitecture::Mlp => net_policy(net, &cache, &feat, s, pl),
            PgArchitecture::HistoryAttention => history_net_policy(net, &cache, &feat, s, pl),
        };
        let (_, _, nc) = nash_conv(&round, &policy);
        sum += nc / 2.0;
    }
    sum / configs.len() as f64
}

fn entropy(q: &[f32]) -> f32 {
    q.iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

fn clone_net(net: &Mlp) -> Mlp {
    Mlp::from_bytes(&net.to_bytes()).expect("round-trip clone")
}

fn fisher_yates<T>(buf: &mut [T], rng: &mut Rng) {
    for i in (1..buf.len()).rev() {
        buf.swap(i, rng.below(i + 1));
    }
}

fn write_pg_config(metrics: &mut std::fs::File, cfg: &PgTrainConfig) -> std::io::Result<()> {
    writeln!(
        metrics,
        "{{\"event\":\"pg_config\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"architecture\":\"{}\",\"input_len\":{},\
         \"mixed\":{},\"min_players\":{},\"max_players\":{},\
         \"min_dice\":{},\"max_dice\":{},\"min_faces\":{},\"max_faces\":{},\
         \"iters\":{},\"episodes_per_iter\":{},\"max_episode_len\":{},\
         \"hidden\":{},\"batch\":{},\"epochs\":{},\"lr\":{:.8},\
         \"momentum\":{:.6},\"l2\":{:.8},\"entropy\":{:.8},\
         \"anchor\":{:.8},\"anchor_update\":{},\"adv_clip\":{:.6},\
         \"val_every\":{},\"eval_games\":{},\"eval_rollouts\":{},\"eval_exploitability\":{},\
         \"keep_checkpoints\":{},\"seed\":{},\"outdir\":\"{}\"}}",
        cfg.players,
        cfg.dice,
        cfg.faces,
        cfg.architecture.as_str(),
        cfg.architecture.input_len(),
        cfg.mixed,
        cfg.min_players,
        cfg.max_players,
        cfg.min_dice,
        cfg.max_dice,
        cfg.min_faces,
        cfg.max_faces,
        cfg.iters,
        cfg.episodes_per_iter,
        cfg.max_episode_len,
        cfg.hidden,
        cfg.batch,
        cfg.epochs,
        cfg.lr,
        cfg.momentum,
        cfg.l2,
        cfg.entropy,
        cfg.anchor,
        cfg.anchor_update,
        cfg.adv_clip,
        cfg.val_every,
        cfg.eval_games,
        cfg.eval_rollouts,
        cfg.eval_exploitability,
        cfg.keep_checkpoints,
        cfg.seed,
        json_escape(&cfg.outdir),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "flat JSONL schema is easier for downstream curve tooling"
)]
fn write_pg_iter(
    metrics: &mut std::fs::File,
    cfg: &PgTrainConfig,
    iters_done: usize,
    stat: &IterStats,
    total_episodes: u64,
    total_decisions: u64,
    policy_ce: f32,
    value_mse: f32,
    train_wall_s: f64,
    iter_wall_s: f64,
    total_wall_s: f64,
    validation_score: Option<f64>,
    exploitability: Option<f64>,
    best_validation_score: f64,
    is_best: bool,
    checkpoint: &str,
    winshares: &[(String, f64)],
) -> std::io::Result<()> {
    let mut fields = vec![
        "\"event\":\"pg_iter\"".to_string(),
        format!("\"method\":\"{}\"", cfg.architecture.method()),
        format!("\"architecture\":\"{}\"", cfg.architecture.as_str()),
        format!("\"input_len\":{}", cfg.architecture.input_len()),
        format!("\"players\":{}", cfg.players),
        format!("\"dice\":{}", cfg.dice),
        format!("\"faces\":{}", cfg.faces),
        format!("\"mixed\":{}", cfg.mixed),
        format!("\"mean_train_players\":{:.6}", stat.mean_players()),
        format!("\"mean_train_dice\":{:.6}", stat.mean_dice()),
        format!("\"mean_train_faces\":{:.6}", stat.mean_faces()),
        format!("\"iters_done\":{iters_done}"),
        format!("\"episodes_per_iter\":{}", cfg.episodes_per_iter),
        format!("\"episodes\":{total_episodes}"),
        format!("\"decisions\":{total_decisions}"),
        format!("\"fresh_decisions\":{}", stat.decisions),
        format!("\"mean_return0\":{:.6}", stat.mean_return0),
        format!("\"mean_entropy\":{:.6}", stat.mean_entropy),
        format!("\"policy_ce\":{policy_ce:.6}"),
        format!("\"value_mse\":{value_mse:.6}"),
        format!("\"train_wall_s\":{train_wall_s:.6}"),
        format!("\"iter_wall_s\":{iter_wall_s:.6}"),
        format!("\"total_wall_s\":{total_wall_s:.6}"),
        format!("\"is_best\":{is_best}"),
        format!("\"checkpoint\":\"{}\"", json_escape(checkpoint)),
        format!(
            "\"latest_checkpoint\":\"{}\"",
            json_escape(&format!("{}/ckpt.bin", cfg.outdir))
        ),
        format!(
            "\"best_checkpoint\":\"{}\"",
            json_escape(&format!("{}/best.bin", cfg.outdir))
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
    fn pseudo_sample_is_well_formed() {
        let cfg = PgTrainConfig {
            players: 2,
            dice: 1,
            faces: 2,
            ..Default::default()
        };
        let game = LiarsDice::new(cfg.players, cfg.dice, cfg.faces);
        let net = Mlp::new(cfg.architecture.input_len(), 16, policy_len(), 1);
        let anchor = clone_net(&net);
        let mut rng = Rng::new(2);
        let cache = net.infer_cache();
        let ep = play_episode(cfg.architecture, &game, &net, &cache, 64, &mut rng);
        assert!(!ep.steps.is_empty());
        let sample = pseudo_sample(&net, &anchor, &ep.steps[0], &cfg);
        assert_eq!(sample.x.len(), feature_len());
        assert!((-1.0..=1.0).contains(&sample.z));
        assert!(!sample.policy.is_empty());
        let sum: f32 = sample.policy.iter().map(|(_, p)| *p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "pseudo target should preserve probability mass: {sum}"
        );
    }

    #[test]
    fn mixed_sampler_stays_inside_config_range() {
        let cfg = PgTrainConfig {
            mixed: true,
            min_players: 2,
            max_players: 4,
            min_dice: 1,
            max_dice: 3,
            min_faces: 2,
            max_faces: 5,
            ..Default::default()
        };
        validate_config(&cfg).unwrap();
        let mut rng = Rng::new(0xC0FFEE);
        let mut saw_more_than_target = false;
        for _ in 0..64 {
            let game = sample_game(&cfg, &mut rng);
            assert!((2..=4).contains(&game.players));
            assert!((1..=3).contains(&game.dice));
            assert!((2..=5).contains(&game.faces));
            saw_more_than_target |=
                game.players != cfg.players || game.dice != cfg.dice || game.faces != cfg.faces;
        }
        assert!(saw_more_than_target);
    }

    #[test]
    fn exploitability_probe_is_finite() {
        let net = Mlp::new(feature_len(), 16, policy_len(), 3);
        let expl = validate_exploitability_configs(&net, PgArchitecture::Mlp, &[(1, 2)]);
        assert!(expl.is_finite());
        assert!(
            expl >= -1e-9,
            "exploitability should be non-negative: {expl}"
        );
    }
}
