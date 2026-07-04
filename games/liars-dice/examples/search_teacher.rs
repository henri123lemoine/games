//! Search-as-teacher distillation for Liar's Dice policy/value checkpoints.
//!
//! This is the C12 policy-improvement operator in the bake-off plan. It loads
//! any AZ-MLP checkpoint (`train_net`, Deep CFR distill, R-NaD, PPO), samples
//! decision states from self-play, runs a net-guided belief-search teacher over
//! each legal action, and distils the improved action distribution back into the
//! same checkpoint format. The output is a normal `NetAgent` checkpoint, so use
//! it in tournament as `nets=teacher:runs/ld_teacher/best.bin`.
//!
//!     cargo run --release -p liars-dice --example search_teacher -- \
//!         base=runs/ld_rnad/best.bin outdir=runs/ld_teacher iters=20 leaf=terminal

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use game_core::{Determinizer, Game, Rng, Turn, winrate_vs_field};
use liars_dice::features::{encode, feature_len, legal_actions_and_support, policy_len};
use liars_dice::{BidConditioned, LiarsDice, NetAgent, ProbConfig, ProbabilisticAgent, support};
use solvers::Rollout;
use solvers::azero::{InferCache, Mlp, Sample, SgdMomentum};

#[derive(Clone)]
struct Args {
    base: String,
    outdir: String,
    players: u8,
    dice: u8,
    faces: u8,
    iters: usize,
    states_per_iter: usize,
    rollouts: u32,
    plies: u32,
    max_search_actions: usize,
    leaf_mode: LeafMode,
    terminal_plies: u32,
    train_value: bool,
    temperature: f64,
    explore: f64,
    batch: usize,
    epochs: usize,
    buffer_cap: usize,
    lr: f32,
    momentum: f32,
    l2: f32,
    eval_games: u32,
    eval_rollouts: u32,
    eval_every: usize,
    keep_checkpoints: bool,
    seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeafMode {
    Terminal,
    ValueHead,
}

impl LeafMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::ValueHead => "value",
        }
    }
}

#[derive(Clone)]
struct TeacherStats {
    states: usize,
    searched_actions: usize,
    legal_actions: usize,
    entropy_sum: f64,
    gap_sum: f64,
    value_sum: f64,
    decisions_played: u64,
}

impl TeacherStats {
    fn record(&mut self, legal: usize, searched: usize, entropy: f64, gap: f64, value: f64) {
        self.states += 1;
        self.legal_actions += legal;
        self.searched_actions += searched;
        self.entropy_sum += entropy;
        self.gap_sum += gap;
        self.value_sum += value;
    }

    fn mean_entropy(&self) -> f64 {
        self.entropy_sum / self.states.max(1) as f64
    }

    fn mean_gap(&self) -> f64 {
        self.gap_sum / self.states.max(1) as f64
    }

    fn mean_value(&self) -> f64 {
        self.value_sum / self.states.max(1) as f64
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse().map_err(invalid_input)?;
    run(&args)
}

fn run(args: &Args) -> std::io::Result<()> {
    if args.players < 2 || args.dice == 0 || args.faces < 2 {
        return Err(invalid_input("players>=2, dice>=1, faces>=2 required"));
    }
    if args.iters == 0 || args.states_per_iter == 0 || args.rollouts == 0 || args.batch == 0 {
        return Err(invalid_input(
            "iters, states_per_iter, rollouts, and batch must be positive",
        ));
    }
    std::fs::create_dir_all(&args.outdir)?;
    let metrics_path = format!("{}/metrics.jsonl", args.outdir);
    let mut metrics = std::fs::File::create(&metrics_path)?;
    write_config_metric(&mut metrics, args)?;

    let mut net = Mlp::load(Path::new(&args.base))?;
    if net.input_len() != feature_len() || net.policy_len() != policy_len() {
        return Err(invalid_input(format!(
            "checkpoint dims are input={} policy={}, expected input={} policy={}",
            net.input_len(),
            net.policy_len(),
            feature_len(),
            policy_len()
        )));
    }
    let mut opt = SgdMomentum::new(args.lr, args.momentum, args.l2);
    let mut buffer: Vec<Sample> = Vec::new();
    let mut grad = Vec::new();
    let mut rng = Rng::new(args.seed ^ 0x51A5_7EAC);
    let game = LiarsDice::new(args.players, args.dice, args.faces);
    let mut best_score = f64::NEG_INFINITY;
    let run_start = Instant::now();

    println!(
        "search-teacher: base={} target={}p{}d{}f iters={} states/iter={} rollouts={} plies={} leaf={} -> {}",
        args.base,
        args.players,
        args.dice,
        args.faces,
        args.iters,
        args.states_per_iter,
        args.rollouts,
        args.plies,
        args.leaf_mode.as_str(),
        args.outdir
    );

    for iter in 0..args.iters {
        let iter_t = Instant::now();
        let cache = net.infer_cache();
        let collect_t = Instant::now();
        let (fresh, stats) = collect_teacher_samples(args, &game, &net, &cache, &mut rng);
        let collect_s = collect_t.elapsed().as_secs_f64();
        buffer.extend(fresh);
        if buffer.len() > args.buffer_cap {
            let drop = buffer.len() - args.buffer_cap;
            buffer.drain(0..drop);
        }

        let train_t = Instant::now();
        let (ce, mse, batches) =
            train_epoch(args, &mut net, &mut opt, &mut grad, &mut buffer, &mut rng);
        let train_s = train_t.elapsed().as_secs_f64();

        let iter_done = iter + 1;
        let ckpt = format!("{}/ckpt.bin", args.outdir);
        net.save(Path::new(&ckpt))?;
        let durable = args
            .keep_checkpoints
            .then(|| format!("{}/ckpt_{iter_done}.bin", args.outdir));
        if let Some(path) = durable.as_deref() {
            net.save(Path::new(path))?;
        }

        let mut winshare = None;
        let mut best = false;
        let should_eval = args.eval_games > 0
            && (iter_done == 1
                || iter_done == args.iters
                || iter_done % args.eval_every.max(1) == 0);
        if should_eval {
            let score = eval_field_winshare(args, &game, &net, args.seed ^ iter_done as u64);
            winshare = Some(score);
            if score > best_score {
                best_score = score;
                best = true;
                net.save(Path::new(&format!("{}/best.bin", args.outdir)))?;
            }
        } else if iter_done == 1 && args.eval_games == 0 {
            best = true;
            best_score = 0.0;
            net.save(Path::new(&format!("{}/best.bin", args.outdir)))?;
        }

        let iter_s = iter_t.elapsed().as_secs_f64();
        println!(
            "iter {:4} states {:5} buf {:6} ce {:.4} mse {:.4} entropy {:.3} gap {:.3} value {:.3}{}  {:.1}s (collect {:.1} train {:.1})",
            iter_done,
            stats.states,
            buffer.len(),
            ce,
            mse,
            stats.mean_entropy(),
            stats.mean_gap(),
            stats.mean_value(),
            winshare
                .map(|w| format!(" winshare {w:.3}{}", if best { " *best" } else { "" }))
                .unwrap_or_default(),
            iter_s,
            collect_s,
            train_s,
        );
        write_iter_metric(
            &mut metrics,
            args,
            iter_done,
            buffer.len(),
            &stats,
            ce,
            mse,
            batches,
            collect_s,
            train_s,
            iter_s,
            run_start.elapsed().as_secs_f64(),
            durable.as_deref().unwrap_or(&ckpt),
            winshare,
            best,
        )?;
        metrics.flush()?;
    }

    Ok(())
}

fn train_epoch(
    args: &Args,
    net: &mut Mlp,
    opt: &mut SgdMomentum,
    grad: &mut Vec<f32>,
    buffer: &mut [Sample],
    rng: &mut Rng,
) -> (f32, f32, u32) {
    let mut ce = 0.0;
    let mut mse = 0.0;
    let mut batches = 0u32;
    for _ in 0..args.epochs {
        fisher_yates(buffer, rng);
        for chunk in buffer.chunks(args.batch) {
            let refs: Vec<&Sample> = chunk.iter().collect();
            let (c, m) = net.grad(&refs, grad);
            opt.step(net, grad);
            ce += c;
            mse += m;
            batches += 1;
        }
    }
    let denom = batches.max(1) as f32;
    (ce / denom, mse / denom, batches)
}

fn collect_teacher_samples(
    args: &Args,
    game: &LiarsDice,
    net: &Mlp,
    cache: &InferCache,
    rng: &mut Rng,
) -> (Vec<Sample>, TeacherStats) {
    let mut out = Vec::with_capacity(args.states_per_iter);
    let mut stats = TeacherStats {
        states: 0,
        searched_actions: 0,
        legal_actions: 0,
        entropy_sum: 0.0,
        gap_sum: 0.0,
        value_sum: 0.0,
        decisions_played: 0,
    };
    let teacher = SearchTeacher {
        args,
        game,
        net,
        cache,
    };
    while out.len() < args.states_per_iter {
        let mut s = game.initial_state();
        let mut steps = 0usize;
        while !game.is_terminal(&s) && out.len() < args.states_per_iter {
            steps += 1;
            if steps > 100_000 {
                break;
            }
            match game.turn(&s) {
                Turn::Chance => {
                    let a = game.sample_chance_action(&s, rng);
                    game.apply(&mut s, a);
                }
                Turn::Player(player) => {
                    let sample = teacher.sample(&s, player, rng, &mut stats);
                    out.push(sample);
                    let acts = game.legal_actions(&s);
                    let idx = teacher.sample_play_action(&s, player, rng);
                    game.apply(&mut s, acts[idx]);
                    stats.decisions_played += 1;
                }
            }
        }
    }
    (out, stats)
}

fn candidate_actions(base_probs: &[f32], support: &[usize], cap: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..base_probs.len()).collect();
    order.sort_by(|&a, &b| base_probs[b].total_cmp(&base_probs[a]));
    let keep = cap.max(1).min(order.len());
    order.truncate(keep);
    // Always keep the two challenge actions when legal; they are often low
    // prior under a smooth net but strategically decisive.
    for (idx, &policy_idx) in support.iter().enumerate() {
        if (policy_idx == 2 || policy_idx == 3) && !order.contains(&idx) {
            order.push(idx);
        }
    }
    order.sort_unstable();
    order
}

struct SearchTeacher<'a> {
    args: &'a Args,
    game: &'a LiarsDice,
    net: &'a Mlp,
    cache: &'a InferCache,
}

impl SearchTeacher<'_> {
    fn sample(
        &self,
        state: &liars_dice::LdState,
        player: usize,
        rng: &mut Rng,
        stats: &mut TeacherStats,
    ) -> Sample {
        let (actions, sup) = legal_actions_and_support(self.game, state);
        let x = encode(self.game, state, player);
        if actions.len() == 1 {
            stats.record(1, 1, 0.0, 0.0, 0.0);
            return Sample {
                x,
                policy: vec![(sup[0], 1.0)],
                z: f32::NAN,
            };
        }

        let base_probs = self.net.policy_value_cached(self.cache, &x, &sup).0;
        let candidates = candidate_actions(&base_probs, &sup, self.args.max_search_actions);
        let values = self.values(state, player, &actions, &candidates, rng);
        let target_probs = value_softmax(&values, self.args.temperature);
        let mut policy = Vec::with_capacity(sup.len());
        for (idx, &policy_idx) in sup.iter().enumerate() {
            let p = candidates
                .iter()
                .position(|&k| k == idx)
                .map(|pos| target_probs[pos])
                .unwrap_or(0.0);
            policy.push((policy_idx, p as f32));
        }
        let teacher_z = target_probs
            .iter()
            .zip(&values)
            .map(|(&p, &v)| p * v)
            .sum::<f64>()
            .clamp(-1.0, 1.0) as f32;
        let (best, second) = top_two(&values);
        let entropy = entropy(&target_probs);
        stats.record(
            sup.len(),
            candidates.len(),
            entropy,
            (best - second).max(0.0),
            f64::from(teacher_z),
        );
        let z = if self.args.train_value {
            teacher_z
        } else {
            f32::NAN
        };
        Sample { x, policy, z }
    }

    fn values(
        &self,
        state: &liars_dice::LdState,
        player: usize,
        actions: &[liars_dice::Action],
        candidates: &[usize],
        rng: &mut Rng,
    ) -> Vec<f64> {
        let seed0 = rng.next_u64();
        let det = BidConditioned::default();
        let mut sums = vec![0.0; candidates.len()];
        for j in 0..self.args.rollouts {
            let world_seed = seed0 ^ (u64::from(j) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for (slot, &action_idx) in candidates.iter().enumerate() {
                let mut sim_rng = Rng::new(world_seed);
                let mut sim = state.clone();
                det.determinize(self.game, &mut sim, player, &mut sim_rng);
                self.game.apply(&mut sim, actions[action_idx]);
                sums[slot] += self.truncated_value(sim, player, &mut sim_rng);
            }
        }
        let denom = f64::from(self.args.rollouts);
        sums.into_iter().map(|s| s / denom).collect()
    }

    fn truncated_value(&self, mut state: liars_dice::LdState, player: usize, rng: &mut Rng) -> f64 {
        let mut moves = 0;
        let move_limit = match self.args.leaf_mode {
            LeafMode::Terminal => self.args.terminal_plies,
            LeafMode::ValueHead => self.args.plies,
        };
        while moves < move_limit && !self.game.is_terminal(&state) {
            match self.game.turn(&state) {
                Turn::Chance => {
                    let a = self.game.sample_chance_action(&state, rng);
                    self.game.apply(&mut state, a);
                }
                Turn::Player(pl) => {
                    let acts = self.game.legal_actions(&state);
                    let idx = self.sample_play_action(&state, pl, rng);
                    self.game.apply(&mut state, acts[idx]);
                    moves += 1;
                }
            }
        }
        if self.game.is_terminal(&state) {
            self.game.returns(&state, player)
        } else if self.args.leaf_mode == LeafMode::ValueHead {
            let x = encode(self.game, &state, player);
            f64::from(self.net.policy_value_cached(self.cache, &x, &[]).1)
        } else {
            panic!(
                "terminal leaf mode exceeded terminal_plies={} before terminal state",
                self.args.terminal_plies
            );
        }
    }

    fn sample_play_action(
        &self,
        state: &liars_dice::LdState,
        player: usize,
        rng: &mut Rng,
    ) -> usize {
        let sup = support(self.game, state);
        if sup.len() == 1 {
            return 0;
        }
        if rng.unit() < self.args.explore {
            return rng.below(sup.len());
        }
        let x = encode(self.game, state, player);
        let probs = self.net.policy_value_cached(self.cache, &x, &sup).0;
        let weights: Vec<f64> = probs.iter().map(|&p| f64::from(p)).collect();
        rng.pick(&weights)
    }
}

fn value_softmax(values: &[f64], temp: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    if temp <= 0.0 {
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return (0..values.len())
            .map(|i| if i == best { 1.0 } else { 0.0 })
            .collect();
    }
    let inv_t = 1.0 / temp;
    let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = values
        .iter()
        .map(|&v| ((v - max_v) * inv_t).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    if z <= 0.0 || !z.is_finite() {
        let p = 1.0 / values.len() as f64;
        return vec![p; values.len()];
    }
    for w in &mut weights {
        *w /= z;
    }
    weights
}

fn top_two(values: &[f64]) -> (f64, f64) {
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &v in values {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    if !second.is_finite() {
        second = best;
    }
    (best, second)
}

fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn eval_field_winshare(args: &Args, game: &LiarsDice, net: &Mlp, seed: u64) -> f64 {
    let hero = NetAgent::new(clone_net(net));
    let field = Rollout::new(
        args.eval_rollouts,
        ProbabilisticAgent::new(ProbConfig::default()),
        BidConditioned::default(),
    );
    winrate_vs_field(game, &hero, &field, args.eval_games, seed)
}

fn clone_net(net: &Mlp) -> Mlp {
    Mlp::from_bytes(&net.to_bytes()).expect("MLP clone round-trip")
}

fn fisher_yates(buf: &mut [Sample], rng: &mut Rng) {
    for i in (1..buf.len()).rev() {
        buf.swap(i, rng.below(i + 1));
    }
}

fn write_config_metric(out: &mut std::fs::File, args: &Args) -> std::io::Result<()> {
    writeln!(
        out,
        "{{\"event\":\"teacher_config\",\"base\":\"{}\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"iters\":{},\"states_per_iter\":{},\"rollouts\":{},\"plies\":{},\
         \"max_search_actions\":{},\"leaf_mode\":\"{}\",\"terminal_plies\":{},\
         \"train_value\":{},\"temperature\":{},\"explore\":{},\"batch\":{},\
         \"epochs\":{},\"buffer_cap\":{},\"lr\":{},\"momentum\":{},\"l2\":{},\
         \"eval_games\":{},\"eval_rollouts\":{},\"eval_every\":{},\
         \"keep_checkpoints\":{},\"seed\":{}}}",
        json_escape(&args.base),
        args.players,
        args.dice,
        args.faces,
        args.iters,
        args.states_per_iter,
        args.rollouts,
        args.plies,
        args.max_search_actions,
        args.leaf_mode.as_str(),
        args.terminal_plies,
        args.train_value,
        args.temperature,
        args.explore,
        args.batch,
        args.epochs,
        args.buffer_cap,
        args.lr,
        args.momentum,
        args.l2,
        args.eval_games,
        args.eval_rollouts,
        args.eval_every,
        args.keep_checkpoints,
        args.seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_iter_metric(
    out: &mut std::fs::File,
    args: &Args,
    iter: usize,
    buffer_len: usize,
    stats: &TeacherStats,
    ce: f32,
    mse: f32,
    batches: u32,
    collect_s: f64,
    train_s: f64,
    iter_s: f64,
    total_s: f64,
    checkpoint: &str,
    winshare: Option<f64>,
    best: bool,
) -> std::io::Result<()> {
    let winshare_text = winshare
        .map(|w| w.to_string())
        .unwrap_or_else(|| "null".to_string());
    writeln!(
        out,
        "{{\"event\":\"teacher_iter\",\"base\":\"{}\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"iters_done\":{},\"states\":{},\"buffer\":{},\"searched_actions\":{},\
         \"legal_actions\":{},\"decisions_played\":{},\"ce\":{:.6},\"mse\":{:.6},\
         \"batches\":{},\"teacher_entropy\":{:.6},\"teacher_gap\":{:.6},\
         \"teacher_value\":{:.6},\"collect_wall_s\":{:.6},\"train_wall_s\":{:.6},\
         \"iter_wall_s\":{:.6},\"total_wall_s\":{:.6},\"states_per_s\":{:.3},\
         \"checkpoint\":\"{}\",\"winshare_mean\":{},\"is_best\":{}}}",
        json_escape(&args.base),
        args.players,
        args.dice,
        args.faces,
        iter,
        stats.states,
        buffer_len,
        stats.searched_actions,
        stats.legal_actions,
        stats.decisions_played,
        ce,
        mse,
        batches,
        stats.mean_entropy(),
        stats.mean_gap(),
        stats.mean_value(),
        collect_s,
        train_s,
        iter_s,
        total_s,
        stats.states as f64 / collect_s.max(1e-9),
        json_escape(checkpoint),
        winshare_text,
        best,
    )
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            base: "runs/ld_net/best.bin".to_string(),
            outdir: "runs/ld_teacher".to_string(),
            players: 5,
            dice: 5,
            faces: 6,
            iters: 20,
            states_per_iter: 512,
            rollouts: 32,
            plies: 4,
            max_search_actions: 12,
            leaf_mode: LeafMode::Terminal,
            terminal_plies: 4096,
            train_value: false,
            temperature: 0.20,
            explore: 0.05,
            batch: 512,
            epochs: 2,
            buffer_cap: 100_000,
            lr: 0.02,
            momentum: 0.9,
            l2: 1e-4,
            eval_games: 100,
            eval_rollouts: 24,
            eval_every: 5,
            keep_checkpoints: true,
            seed: 0xC12,
        };
        for raw in std::env::args().skip(1) {
            let Some((key, value)) = raw.split_once('=') else {
                return Err(format!("expected key=value argument, got '{raw}'"));
            };
            match key {
                "base" | "net" => args.base = value.to_string(),
                "outdir" => args.outdir = value.to_string(),
                "players" => args.players = parse_num(value, key)?,
                "dice" => args.dice = parse_num(value, key)?,
                "faces" => args.faces = parse_num(value, key)?,
                "iters" => args.iters = parse_num(value, key)?,
                "states_per_iter" | "states" => args.states_per_iter = parse_num(value, key)?,
                "rollouts" => args.rollouts = parse_num(value, key)?,
                "plies" => args.plies = parse_num(value, key)?,
                "max_search_actions" | "cand_cap" => {
                    args.max_search_actions = parse_num(value, key)?;
                }
                "leaf" | "leaf_mode" => args.leaf_mode = parse_leaf_mode(value)?,
                "terminal_plies" | "max_terminal_plies" => {
                    args.terminal_plies = parse_num(value, key)?;
                }
                "train_value" => args.train_value = parse_bool(value)?,
                "temperature" | "temp" => args.temperature = parse_num(value, key)?,
                "explore" => args.explore = parse_num(value, key)?,
                "batch" => args.batch = parse_num(value, key)?,
                "epochs" => args.epochs = parse_num(value, key)?,
                "buffer_cap" => args.buffer_cap = parse_num(value, key)?,
                "lr" => args.lr = parse_num(value, key)?,
                "momentum" => args.momentum = parse_num(value, key)?,
                "l2" => args.l2 = parse_num(value, key)?,
                "eval_games" => args.eval_games = parse_num(value, key)?,
                "eval_rollouts" => args.eval_rollouts = parse_num(value, key)?,
                "eval_every" => args.eval_every = parse_num(value, key)?,
                "keep_checkpoints" => args.keep_checkpoints = parse_bool(value)?,
                "seed" => args.seed = parse_num(value, key)?,
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        Ok(args)
    }
}

fn parse_leaf_mode(s: &str) -> Result<LeafMode, String> {
    match s {
        "terminal" | "terminal_only" | "rollout" => Ok(LeafMode::Terminal),
        "value" | "value_head" | "truncated" => Ok(LeafMode::ValueHead),
        other => Err(format!(
            "expected leaf mode 'terminal' or 'value', got '{other}'"
        )),
    }
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("expected boolean, got '{other}'")),
    }
}

fn parse_num<T: std::str::FromStr>(s: &str, key: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("invalid value for {key}: '{s}'"))
}

fn invalid_input<E: ToString>(err: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, err.to_string())
}
