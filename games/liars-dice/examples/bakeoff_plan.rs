//! Generate the concrete matched-compute bake-off run manifest.
//!
//! This does not train anything by itself. It writes a JSONL manifest (and,
//! optionally, a shell script) with the actual commands for the Phase 2/3
//! workflow: preflight gates, baseline tournament, 1x/3x/10x learned-contender
//! training runs, checkpoint tournament sweeps, population selection, and the
//! final `curve_report` join/extrapolation command.
//!
//!     cargo run --release -p liars-dice --example bakeoff_plan -- \
//!         outdir=runs/ld_bakeoff_5p5d6f script=runs/ld_bakeoff_5p5d6f/run.sh
//!
//! Use `smoke=1` to generate tiny commands suitable for plumbing checks. It
//! defaults to 2p1d2f plus tiny games/rollout budgets; pass `players=`, `dice=`,
//! `faces=`, `games=`, `rollouts=`, `eval_games=`, or `eval_rollouts=` to
//! override those smoke defaults explicitly.

use std::fs::{File, OpenOptions};
use std::io::{self, Write as _};
use std::path::Path;

const DEFAULT_OUTDIR: &str = "runs/ld_bakeoff_5p5d6f";
const DEFAULT_MANIFEST: &str = "runs/ld_bakeoff_5p5d6f/manifest.jsonl";
const FULL_GAMES: u32 = 200;
const FULL_ROLLOUTS: u32 = 48;
const CHECKPOINT_SCREEN_GAMES: u32 = 40;
const CHECKPOINT_SCREEN_ROLLOUTS: u32 = 24;
const SMOKE_PLAYERS: u8 = 2;
const SMOKE_DICE: u8 = 1;
const SMOKE_FACES: u8 = 2;
const SMOKE_GAMES: u32 = 4;
const SMOKE_ROLLOUTS: u32 = 2;

#[derive(Clone)]
struct Args {
    players: u8,
    dice: u8,
    faces: u8,
    outdir: String,
    manifest: Option<String>,
    script: Option<String>,
    smoke: bool,
    threads: usize,
    games: u32,
    rollouts: u32,
    eval_games: u32,
    eval_rollouts: u32,
    multipliers: Vec<u32>,
}

struct Step {
    phase: &'static str,
    method: &'static str,
    scale: Option<u32>,
    command: String,
    metrics: Option<String>,
    checkpoint: Option<String>,
    note: &'static str,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let steps = build_steps(&args);
    write_manifest(&args, &steps).map_err(|e| format!("write manifest: {e}"))?;
    if let Some(path) = args.script.as_deref() {
        write_script(path, &steps).map_err(|e| format!("write script: {e}"))?;
    }
    print_plan(&args, &steps);
    Ok(())
}

fn build_steps(args: &Args) -> Vec<Step> {
    let mut steps = Vec::new();
    steps.push(Step::new(
        "gate",
        "fmt",
        None,
        "cargo fmt --check",
        None,
        None,
        "format gate before each phase",
    ));
    steps.push(Step::new(
        "gate",
        "clippy",
        None,
        "cargo clippy --release --all-targets -- -D warnings",
        None,
        None,
        "warning-free release build",
    ));
    steps.push(Step::new(
        "gate",
        "test",
        None,
        "cargo test --release",
        None,
        None,
        "full release test gate",
    ));
    steps.push(Step::new(
        "gate",
        "ml-rnad-check",
        None,
        "cargo check --release --manifest-path ml/ld-rnad/Cargo.toml",
        None,
        None,
        "separate R-NaD trainer crate check",
    ));
    steps.push(Step::new(
        "gate",
        "ml-rnad-clippy",
        None,
        "cargo clippy --release --manifest-path ml/ld-rnad/Cargo.toml -- -D warnings",
        None,
        None,
        "warning-free R-NaD trainer crate",
    ));
    steps.push(Step::new(
        "gate",
        "ml-rnad-test",
        None,
        "cargo test --release --manifest-path ml/ld-rnad/Cargo.toml",
        None,
        None,
        "release tests for standalone R-NaD trainer crate",
    ));
    steps.push(Step::new(
        "gate",
        "ml-ppo-check",
        None,
        "cargo check --release --manifest-path ml/ld-ppo/Cargo.toml",
        None,
        None,
        "separate PPO trainer crate check",
    ));
    steps.push(Step::new(
        "gate",
        "ml-ppo-clippy",
        None,
        "cargo clippy --release --manifest-path ml/ld-ppo/Cargo.toml -- -D warnings",
        None,
        None,
        "warning-free PPO trainer crate",
    ));
    steps.push(Step::new(
        "gate",
        "ml-ppo-test",
        None,
        "cargo test --release --manifest-path ml/ld-ppo/Cargo.toml",
        None,
        None,
        "release tests for standalone PPO trainer crate",
    ));
    steps.push(Step::new(
        "profile",
        "cheap-search",
        None,
        cheap_profile_command(args),
        Some(format!("{}/profile_search.jsonl", args.outdir)),
        None,
        "machine profile for cheap/search agents before competitive runs",
    ));
    steps.push(Step::new(
        "profile",
        "online-budget",
        None,
        online_budget_profile_command(args),
        Some(format!("{}/profile_online_budget.jsonl", args.outdir)),
        None,
        "blueprint-search budget sweep and target move-cost profile",
    ));
    steps.push(Step::new(
        "eval",
        "baseline",
        None,
        format!(
            "cargo run --release -p liars-dice --example tournament -- players={} dice={} faces={} games={} rollouts={} agents={}{} metrics={}/baseline_tournament.jsonl",
            args.players,
            args.dice,
            args.faces,
            args.games,
            args.rollouts,
            baseline_agents(args),
            tournament_budget_args(args),
            args.outdir
        ),
        Some(format!("{}/baseline_tournament.jsonl", args.outdir)),
        None,
        "cheap contender and baseline-field matrix",
    ));

    for &scale in &args.multipliers {
        add_training_scale(args, scale, &mut steps);
        steps.push(Step::new(
            "eval",
            "checkpoint-tournament",
            Some(scale),
            tournament_command(args, scale),
            Some(format!("{}/tournament_x{scale}.jsonl", args.outdir)),
            None,
            "broad first-pass cross-play screen for same-scale learned checkpoints",
        ));
    }

    steps.push(Step::new(
        "meta",
        "population",
        None,
        population_command(args),
        Some(format!("{}/population.jsonl", args.outdir)),
        None,
        "held-out league selector over all learned checkpoint populations",
    ));
    steps.push(Step::new(
        "report",
        "curve-report",
        None,
        curve_report_command(args),
        None,
        None,
        "joins training+tournament metrics and prints Phase-3 100x table",
    ));
    steps
}

fn add_training_scale(args: &Args, scale: u32, steps: &mut Vec<Step>) {
    let out = |name: &str| format!("{}/{name}_x{scale}", args.outdir);
    let ckpt = |name: &str| format!("{}/{name}_x{scale}/best.bin", args.outdir);
    let hidden = trainer_hidden(args);

    let distill_iters = scaled(args, scale, 2, 80);
    let distill_rounds = if args.smoke { 4 } else { 400 };
    steps.push(Step::new(
        "train",
        "distill-fvi",
        Some(scale),
        format!(
            "cargo run --release -p liars-dice --example train_net -- iters={distill_iters} rounds_per_iter={distill_rounds} playouts={} hidden={hidden} threads={} eval_games={} eval_rollouts={} keep_checkpoints=1{} outdir={}",
            if args.smoke { 1 } else { 12 },
            args.threads,
            args.eval_games,
            args.eval_rollouts,
            smoke_distill_args(args),
            out("distill"),
        ),
        Some(format!("{}/metrics.jsonl", out("distill"))),
        Some(ckpt("distill")),
        "CFR/MCCFR distillation plus fitted value iteration",
    ));

    let deepcfr_iters = scaled(args, scale, 2, 800);
    steps.push(Step::new(
        "train",
        "deepcfr",
        Some(scale),
        format!(
            "cargo run --release -p liars-dice --features parallel --example deepcfr_train -- iters={deepcfr_iters} block={} warmup_iters={} traversals={} hidden={hidden} threads={} eval_games={} eval_rollouts={} keep_checkpoints=1{} outdir={}",
            if args.smoke { 1 } else { 200 },
            if args.smoke { 1 } else { deepcfr_iters / 4 },
            if args.smoke { 1 } else { 64 },
            args.threads,
            args.eval_games,
            args.eval_rollouts,
            smoke_deepcfr_args(args),
            out("deepcfr"),
        ),
        Some(format!("{}/metrics.jsonl", out("deepcfr"))),
        Some(ckpt("deepcfr")),
        "C5 Deep CFR over sampled Liar's Dice round subgames",
    ));

    let rnad_iters = scaled(args, scale, 2, 120);
    let episodes = if args.smoke { 4 } else { 512 };
    steps.push(Step::new(
        "train",
        "rnad",
        Some(scale),
        format!(
            "cargo run --release --manifest-path ml/ld-rnad/Cargo.toml -- players={} dice={} faces={} mixed=1 max_players={} max_dice=8 max_faces={} iters={rnad_iters} episodes_per_iter={episodes} hidden={hidden} eval_games={} eval_rollouts={} keep_checkpoints=1{} outdir={}",
            args.players,
            args.dice,
            args.faces,
            args.players,
            args.faces,
            args.eval_games,
            args.eval_rollouts,
            smoke_pg_args(args),
            out("rnad"),
        ),
        Some(format!("{}/metrics.jsonl", out("rnad"))),
        Some(ckpt("rnad")),
        "C9 regularized policy-gradient self-play",
    ));

    steps.push(Step::new(
        "train",
        "history-rnad",
        Some(scale),
        format!(
            "cargo run --release --manifest-path ml/ld-rnad/Cargo.toml -- architecture=history players={} dice={} faces={} mixed=1 max_players={} max_dice=8 max_faces={} iters={rnad_iters} episodes_per_iter={episodes} hidden={hidden} eval_games={} eval_rollouts={} keep_checkpoints=1{} outdir={}",
            args.players,
            args.dice,
            args.faces,
            args.players,
            args.faces,
            args.eval_games,
            args.eval_rollouts,
            smoke_pg_args(args),
            out("history"),
        ),
        Some(format!("{}/metrics.jsonl", out("history"))),
        Some(ckpt("history")),
        "C13 compact bid-history architecture with the C9 objective",
    ));

    let ppo_iters = scaled(args, scale, 2, 120);
    steps.push(Step::new(
        "train",
        "ppo",
        Some(scale),
        format!(
            "cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- players={} dice={} faces={} mixed=1 max_players={} max_dice=8 max_faces={} iters={ppo_iters} actors={} steps={} hidden={hidden} eval_games={} eval_rollouts={} keep_checkpoints=1{} outdir={}",
            args.players,
            args.dice,
            args.faces,
            args.players,
            args.faces,
            if args.smoke { 2 } else { 64 },
            if args.smoke { 4 } else { 64 },
            args.eval_games,
            args.eval_rollouts,
            smoke_ppo_args(args),
            out("ppo"),
        ),
        Some(format!("{}/metrics.jsonl", out("ppo"))),
        Some(ckpt("ppo")),
        "C10 PPO self-play brittleness/control contender",
    ));

    let rebel_steps = scaled(args, scale, 10, 3_000);
    steps.push(Step::new(
        "train",
        "rebel",
        Some(scale),
        format!(
            "OUTDIR={} STEPS={rebel_steps} HIDDEN={hidden} NUM_ITERS={} DEPTH={} EVAL_EVERY={} WINSHARE_PLAYERS={} WINSHARE_DICE={} WINSHARE_FACES={} WINSHARE_GAMES={} WINSHARE_ROLLOUTS={} WINSHARE_ITERS={} KEEP_CHECKPOINTS=1{} cargo run --release -p liars-dice --features parallel --example monitored_train",
            out("rebel"),
            if args.smoke { 4 } else { 256 },
            if args.smoke { 1 } else { 2 },
            rebel_eval_every(args),
            args.players,
            args.dice,
            args.faces,
            rebel_winshare_games(args),
            rebel_winshare_rollouts(args),
            rebel_winshare_iters(args),
            smoke_rebel_env(args),
        ),
        Some(format!("{}/metrics.jsonl", out("rebel"))),
        Some(ckpt("rebel")),
        "C7 ReBeL deploy-family value-net self-play",
    ));

    let teacher_iters = scaled(args, scale, 1, 40);
    steps.push(Step::new(
        "train",
        "search-teacher",
        Some(scale),
        format!(
            "cargo run --release -p liars-dice --example search_teacher -- base={} outdir={} players={} dice={} faces={} iters={teacher_iters} states_per_iter={} rollouts={} plies={} eval_games={} eval_rollouts={} keep_checkpoints=1{}",
            ckpt("rnad"),
            out("teacher"),
            args.players,
            args.dice,
            args.faces,
            if args.smoke { 4 } else { 512 },
            if args.smoke { 2 } else { 32 },
            if args.smoke { 1 } else { 4 },
            args.eval_games,
            args.eval_rollouts,
            smoke_teacher_args(args),
        ),
        Some(format!("{}/metrics.jsonl", out("teacher"))),
        Some(ckpt("teacher")),
        "C12 expensive net-guided belief-search teacher over the C9 base",
    ));
}

fn tournament_command(args: &Args, scale: u32) -> String {
    format!(
        "cargo run --release -p liars-dice --example tournament -- players={} dice={} faces={} games={} rollouts={} agents={},nets,net-searches,solves,rnads,ppos,histories,rebels{} rebel={}/rebel_x{scale}/best.bin nets={} rnads={} ppos={} histories={} rebels={} metrics={}/tournament_x{scale}.jsonl",
        args.players,
        args.dice,
        args.faces,
        checkpoint_tournament_games(args),
        checkpoint_tournament_rollouts(args),
        baseline_agents(args),
        checkpoint_tournament_budget_args(args),
        args.outdir,
        nets_arg(args, scale),
        rnads_arg(args, scale),
        ppos_arg(args, scale),
        histories_arg(args, scale),
        rebels_arg(args, scale),
        args.outdir,
    )
}

fn cheap_profile_command(args: &Args) -> String {
    format!(
        "cargo run --release -p liars-dice --example tournament -- players={} dice={} faces={} games={} rollouts={} agents={}{} metrics={}/profile_search.jsonl",
        args.players,
        args.dice,
        args.faces,
        profile_games(args),
        profile_rollouts(args),
        profile_agents(args),
        tournament_budget_args(args),
        args.outdir,
    )
}

fn online_budget_profile_command(args: &Args) -> String {
    format!(
        "cargo run --release -p liars-dice --features parallel --example budget_diag -- tests=1,big games={} rollouts={} budgets={} restarts={} big_games=0 metrics={}/profile_online_budget.jsonl",
        profile_games(args),
        profile_rollouts(args),
        online_profile_budgets(args),
        online_profile_restarts(args),
        args.outdir,
    )
}

fn profile_games(args: &Args) -> u32 {
    if args.smoke {
        args.games
    } else {
        args.games.min(40)
    }
    .max(1)
}

fn profile_rollouts(args: &Args) -> u32 {
    if args.smoke {
        args.rollouts
    } else {
        args.rollouts.min(24)
    }
    .max(1)
}

fn online_profile_budgets(args: &Args) -> &'static str {
    if args.smoke { "16,32" } else { "128,256,512" }
}

fn online_profile_restarts(args: &Args) -> usize {
    if args.smoke { 1 } else { 3 }
}

fn trainer_hidden(args: &Args) -> usize {
    if args.smoke { 32 } else { 256 }
}

fn tournament_budget_args(args: &Args) -> &'static str {
    if args.smoke {
        smoke_tournament_budget_args()
    } else {
        full_tournament_budget_args()
    }
}

fn baseline_agents(args: &Args) -> &'static str {
    if args.smoke {
        "random,honest-bayes,aggressive-bluffer,conservative-caller,belief,rollout,abstract-rollout,is-mcts,baseline-field,mccfr,qlearn,blueprint-search"
    } else {
        "random,honest-bayes,aggressive-bluffer,conservative-caller,belief,rollout,abstract-rollout,is-mcts,baseline-field,qlearn,blueprint-search"
    }
}

fn profile_agents(args: &Args) -> &'static str {
    if args.smoke {
        "belief,rollout,abstract-rollout,is-mcts,mccfr,qlearn,blueprint-search"
    } else {
        "belief,rollout,abstract-rollout,is-mcts,qlearn,blueprint-search"
    }
}

fn checkpoint_tournament_games(args: &Args) -> u32 {
    if args.smoke {
        args.games
    } else {
        CHECKPOINT_SCREEN_GAMES
    }
}

fn checkpoint_tournament_rollouts(args: &Args) -> u32 {
    if args.smoke {
        args.rollouts
    } else {
        CHECKPOINT_SCREEN_ROLLOUTS
    }
}

fn checkpoint_tournament_budget_args(args: &Args) -> &'static str {
    if args.smoke {
        smoke_tournament_budget_args()
    } else {
        full_checkpoint_screen_budget_args()
    }
}

fn smoke_tournament_budget_args() -> &'static str {
    " mccfr_iters=4 q_episodes=16 mcts_worlds=1 mcts_sims=1 solve_iters=16 solve_max_iters=16 solve_restarts=1 solve_flat_iters=16 net_search_rollouts=1 net_search_plies=1 rebel_iters=4 rebel_depth=1"
}

fn full_tournament_budget_args() -> &'static str {
    " mccfr_iters=16 q_episodes=1000 mcts_worlds=2 mcts_sims=8 solve_iters=256 solve_max_iters=256 solve_restarts=1 solve_flat_iters=256 net_search_rollouts=12 net_search_plies=3 rebel_iters=16 rebel_depth=2"
}

fn full_checkpoint_screen_budget_args() -> &'static str {
    " mccfr_iters=16 q_episodes=1000 mcts_worlds=2 mcts_sims=4 solve_iters=64 solve_max_iters=64 solve_restarts=1 solve_flat_iters=64 net_search_rollouts=6 net_search_plies=2 rebel_iters=8 rebel_depth=2"
}

fn smoke_distill_args(args: &Args) -> &'static str {
    if args.smoke {
        " warmup_iters=1 cfr_iters=8 es_iters=8 small_total=2 batch=8 epochs=1 buffer_cap=64 val_every=1"
    } else {
        ""
    }
}

fn smoke_deepcfr_args(args: &Args) -> &'static str {
    if args.smoke {
        " train_every=1 adv_reservoir=64 strat_reservoir=64 adv_steps=1 strat_steps=1 batch=8"
    } else {
        ""
    }
}

fn smoke_pg_args(args: &Args) -> String {
    if args.smoke {
        format!(
            " min_players=2 max_players={} min_dice=1 max_dice={} min_faces=2 max_faces={} batch=8 epochs=1 val_every=1 eval_exploitability=0",
            args.players.max(2),
            args.dice.max(2),
            args.faces.max(2),
        )
    } else {
        String::new()
    }
}

fn smoke_ppo_args(args: &Args) -> String {
    if args.smoke {
        format!(
            " min_players=2 max_players={} min_dice=1 max_dice={} min_faces=2 max_faces={} epochs=1 minibatches=1 val_every=1 eval_exploitability=0",
            args.players.max(2),
            args.dice.max(2),
            args.faces.max(2),
        )
    } else {
        String::new()
    }
}

fn smoke_rebel_env(args: &Args) -> &'static str {
    if args.smoke {
        " EVAL_ITERS=8 FIT_ITERS=8 TRAIN_RATIO=1 GEN_PER=1 BATCH=8 WARMUP=4"
    } else {
        ""
    }
}

fn rebel_eval_every(args: &Args) -> u32 {
    if args.smoke { 2 } else { 500 }
}

fn rebel_winshare_games(args: &Args) -> u32 {
    if args.smoke {
        args.eval_games
    } else {
        profile_games(args)
    }
}

fn rebel_winshare_rollouts(args: &Args) -> u32 {
    if args.smoke {
        args.eval_rollouts
    } else {
        profile_rollouts(args)
    }
}

fn rebel_winshare_iters(args: &Args) -> u32 {
    if args.smoke { 4 } else { 16 }
}

fn smoke_teacher_args(args: &Args) -> &'static str {
    if args.smoke {
        " max_search_actions=2 batch=8 epochs=1 buffer_cap=16"
    } else {
        ""
    }
}

fn population_command(args: &Args) -> String {
    format!(
        "cargo run --release -p liars-dice --example population -- players={} dice={} faces={} games={} rollouts={} fields={}{} nets={} rnads={} ppos={} histories={} rebels={} metrics={}/population.jsonl",
        args.players,
        args.dice,
        args.faces,
        args.games,
        args.rollouts,
        baseline_agents(args),
        population_budget_args(args),
        all_nets_arg(args),
        all_kind_arg(args, "rnad", "rnad"),
        all_kind_arg(args, "ppo", "ppo"),
        all_kind_arg(args, "history", "history"),
        all_kind_arg(args, "rebel", "rebel"),
        args.outdir,
    )
}

fn population_budget_args(args: &Args) -> &'static str {
    if args.smoke {
        smoke_tournament_budget_args()
    } else {
        full_tournament_budget_args()
    }
}

fn curve_report_command(args: &Args) -> String {
    let mut parts = vec![
        "cargo run --release -p liars-dice --example curve_report --".to_string(),
        "field=baseline-field".to_string(),
        format!("profile={}/profile_search.jsonl", args.outdir),
        format!("profile={}/profile_online_budget.jsonl", args.outdir),
        format!("train={}/population.jsonl", args.outdir),
    ];
    for &scale in &args.multipliers {
        for method in [
            "distill", "deepcfr", "rnad", "history", "ppo", "rebel", "teacher",
        ] {
            parts.push(format!(
                "train={}/{method}_x{scale}/metrics.jsonl",
                args.outdir
            ));
        }
        parts.push(format!(
            "tournament={}/tournament_x{scale}.jsonl",
            args.outdir
        ));
    }
    parts.push(format!(
        "tournament={}/baseline_tournament.jsonl",
        args.outdir
    ));
    parts.join(" ")
}

fn nets_arg(args: &Args, scale: u32) -> String {
    [
        format!("distill_x{scale}:{}/distill_x{scale}/best.bin", args.outdir),
        format!("deepcfr_x{scale}:{}/deepcfr_x{scale}/best.bin", args.outdir),
        format!("teacher_x{scale}:{}/teacher_x{scale}/best.bin", args.outdir),
    ]
    .join(",")
}

fn rnads_arg(args: &Args, scale: u32) -> String {
    format!("rnad_x{scale}:{}/rnad_x{scale}/best.bin", args.outdir)
}

fn ppos_arg(args: &Args, scale: u32) -> String {
    format!("ppo_x{scale}:{}/ppo_x{scale}/best.bin", args.outdir)
}

fn histories_arg(args: &Args, scale: u32) -> String {
    format!("history_x{scale}:{}/history_x{scale}/best.bin", args.outdir)
}

fn rebels_arg(args: &Args, scale: u32) -> String {
    format!("rebel_x{scale}:{}/rebel_x{scale}/best.bin", args.outdir)
}

fn all_nets_arg(args: &Args) -> String {
    args.multipliers
        .iter()
        .flat_map(|&scale| {
            [
                format!("distill_x{scale}:{}/distill_x{scale}/best.bin", args.outdir),
                format!("deepcfr_x{scale}:{}/deepcfr_x{scale}/best.bin", args.outdir),
                format!("teacher_x{scale}:{}/teacher_x{scale}/best.bin", args.outdir),
            ]
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn all_kind_arg(args: &Args, kind: &str, dir_prefix: &str) -> String {
    args.multipliers
        .iter()
        .map(|scale| {
            format!(
                "{kind}_x{scale}:{}/{dir_prefix}_x{scale}/best.bin",
                args.outdir
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn scaled(args: &Args, scale: u32, smoke_base: u32, full_base: u32) -> u32 {
    let base = if args.smoke { smoke_base } else { full_base };
    base.saturating_mul(scale.max(1))
}

fn write_manifest(args: &Args, steps: &[Step]) -> io::Result<()> {
    let Some(path) = args.manifest.as_deref() else {
        return Ok(());
    };
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    writeln!(
        file,
        "{{\"event\":\"bakeoff_plan_config\",\"players\":{},\"dice\":{},\"faces\":{},\"outdir\":\"{}\",\"smoke\":{},\"threads\":{},\"games\":{},\"rollouts\":{},\"eval_games\":{},\"eval_rollouts\":{},\"multipliers\":[{}]}}",
        args.players,
        args.dice,
        args.faces,
        json_escape(&args.outdir),
        args.smoke,
        args.threads,
        args.games,
        args.rollouts,
        args.eval_games,
        args.eval_rollouts,
        args.multipliers
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )?;
    for (idx, step) in steps.iter().enumerate() {
        writeln!(
            file,
            "{{\"event\":\"bakeoff_step\",\"idx\":{},\"phase\":\"{}\",\"method\":\"{}\",\"scale\":{},\"command\":\"{}\",\"metrics\":{},\"checkpoint\":{},\"note\":\"{}\"}}",
            idx + 1,
            step.phase,
            step.method,
            step.scale
                .map(|s| s.to_string())
                .unwrap_or_else(|| "null".to_string()),
            json_escape(&step.command),
            json_opt(step.metrics.as_deref()),
            json_opt(step.checkpoint.as_deref()),
            json_escape(step.note),
        )?;
    }
    Ok(())
}

fn write_script(path: &str, steps: &[Step]) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    writeln!(file, "#!/usr/bin/env bash")?;
    writeln!(file, "set -euo pipefail")?;
    writeln!(file)?;
    for step in steps {
        writeln!(
            file,
            "echo '[{}:{}{}] {}'",
            step.phase,
            step.method,
            step.scale
                .map(|s| format!(" x{s}"))
                .unwrap_or_else(String::new),
            step.note.replace('\'', "")
        )?;
        writeln!(file, "{}", step.command)?;
        writeln!(file)?;
    }
    Ok(())
}

fn print_plan(args: &Args, steps: &[Step]) {
    println!(
        "Liar's Dice bake-off plan: {}p{}d{}f, scales={:?}, smoke={}",
        args.players, args.dice, args.faces, args.multipliers, args.smoke
    );
    if let Some(path) = args.manifest.as_deref() {
        println!("manifest: {path}");
    }
    if let Some(path) = args.script.as_deref() {
        println!("script: {path}");
    }
    println!("steps:");
    for (idx, step) in steps.iter().enumerate() {
        let scale = step
            .scale
            .map(|s| format!(" x{s}"))
            .unwrap_or_else(String::new);
        println!(
            "  {:>2}. {:<8} {:<18}{}  {}",
            idx + 1,
            step.phase,
            step.method,
            scale,
            step.note
        );
    }
}

impl Step {
    fn new(
        phase: &'static str,
        method: &'static str,
        scale: Option<u32>,
        command: impl Into<String>,
        metrics: Option<String>,
        checkpoint: Option<String>,
        note: &'static str,
    ) -> Self {
        Self {
            phase,
            method,
            scale,
            command: command.into(),
            metrics,
            checkpoint,
            note,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1))
    }

    fn parse_from<I, S>(raw_args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut args = Self {
            players: 5,
            dice: 5,
            faces: 6,
            outdir: DEFAULT_OUTDIR.to_string(),
            manifest: Some(DEFAULT_MANIFEST.to_string()),
            script: None,
            smoke: false,
            threads: 18,
            games: FULL_GAMES,
            rollouts: FULL_ROLLOUTS,
            eval_games: FULL_GAMES,
            eval_rollouts: FULL_ROLLOUTS,
            multipliers: vec![1, 3, 10],
        };
        let mut players_set = false;
        let mut dice_set = false;
        let mut faces_set = false;
        let mut games_set = false;
        let mut rollouts_set = false;
        let mut eval_games_set = false;
        let mut eval_rollouts_set = false;
        for raw in raw_args {
            let raw = raw.as_ref();
            let Some((key, value)) = raw.split_once('=') else {
                return Err(format!("expected key=value argument, got '{raw}'"));
            };
            match key {
                "players" => {
                    args.players = parse_num(value, key)?;
                    players_set = true;
                }
                "dice" => {
                    args.dice = parse_num(value, key)?;
                    dice_set = true;
                }
                "faces" => {
                    args.faces = parse_num(value, key)?;
                    faces_set = true;
                }
                "outdir" => {
                    args.outdir = value.to_string();
                    if args.manifest.as_deref() == Some(DEFAULT_MANIFEST) {
                        args.manifest = Some(format!("{}/manifest.jsonl", args.outdir));
                    }
                }
                "manifest" => {
                    args.manifest = if value == "none" {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "script" => {
                    args.script = if value == "none" {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "smoke" => args.smoke = parse_bool(value, key)?,
                "threads" => args.threads = parse_num(value, key)?,
                "games" => {
                    args.games = parse_num(value, key)?;
                    games_set = true;
                }
                "rollouts" => {
                    args.rollouts = parse_num(value, key)?;
                    rollouts_set = true;
                }
                "eval_games" => {
                    args.eval_games = parse_num(value, key)?;
                    eval_games_set = true;
                }
                "eval_rollouts" => {
                    args.eval_rollouts = parse_num(value, key)?;
                    eval_rollouts_set = true;
                }
                "multipliers" | "scales" => args.multipliers = parse_multipliers(value)?,
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        if args.smoke {
            if !players_set {
                args.players = SMOKE_PLAYERS;
            }
            if !dice_set {
                args.dice = SMOKE_DICE;
            }
            if !faces_set {
                args.faces = SMOKE_FACES;
            }
            if !games_set {
                args.games = SMOKE_GAMES;
            }
            if !rollouts_set {
                args.rollouts = SMOKE_ROLLOUTS;
            }
            if !eval_games_set {
                args.eval_games = SMOKE_GAMES;
            }
            if !eval_rollouts_set {
                args.eval_rollouts = SMOKE_ROLLOUTS;
            }
        }
        if args.players < 2 {
            return Err("players must be at least 2".to_string());
        }
        if args.dice == 0 {
            return Err("dice must be positive".to_string());
        }
        if args.faces < 2 {
            return Err("faces must be at least 2".to_string());
        }
        if args.games == 0 || args.eval_games == 0 {
            return Err("games and eval_games must be positive".to_string());
        }
        if args.rollouts == 0 || args.eval_rollouts == 0 {
            return Err("rollouts and eval_rollouts must be positive".to_string());
        }
        if args.threads == 0 {
            return Err("threads must be positive".to_string());
        }
        if args.multipliers.is_empty() {
            return Err("need at least one multiplier".to_string());
        }
        args.multipliers.sort_unstable();
        args.multipliers.dedup();
        Ok(args)
    }
}

fn parse_multipliers(value: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in value.split(',').filter(|s| !s.is_empty()) {
        let n: u32 = parse_num(part.trim(), "multipliers")?;
        if n == 0 {
            return Err("multipliers must be positive".to_string());
        }
        out.push(n);
    }
    Ok(out)
}

fn parse_bool(value: &str, key: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("expected boolean for {key}, got '{value}'")),
    }
}

fn parse_num<T: std::str::FromStr>(value: &str, key: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid numeric value for {key}: '{value}'"))
}

fn json_opt(value: Option<&str>) -> String {
    value
        .map(|s| format!("\"{}\"", json_escape(s)))
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_uses_tiny_default_budgets() {
        let args = Args::parse_from(["smoke=1"]).unwrap();
        assert_eq!(args.players, SMOKE_PLAYERS);
        assert_eq!(args.dice, SMOKE_DICE);
        assert_eq!(args.faces, SMOKE_FACES);
        assert_eq!(args.games, SMOKE_GAMES);
        assert_eq!(args.rollouts, SMOKE_ROLLOUTS);
        assert_eq!(args.eval_games, SMOKE_GAMES);
        assert_eq!(args.eval_rollouts, SMOKE_ROLLOUTS);
    }

    #[test]
    fn smoke_preserves_explicit_target_and_budgets() {
        let args = Args::parse_from([
            "smoke=1",
            "players=5",
            "dice=5",
            "faces=6",
            "games=11",
            "rollouts=12",
            "eval_games=13",
            "eval_rollouts=14",
        ])
        .unwrap();
        assert_eq!(args.players, 5);
        assert_eq!(args.dice, 5);
        assert_eq!(args.faces, 6);
        assert_eq!(args.games, 11);
        assert_eq!(args.rollouts, 12);
        assert_eq!(args.eval_games, 13);
        assert_eq!(args.eval_rollouts, 14);
    }

    #[test]
    fn smoke_pg_args_keep_mixed_family_tiny_by_default() {
        let args = Args::parse_from(["smoke=1"]).unwrap();
        let pg = smoke_pg_args(&args);
        assert!(pg.contains("min_dice=1"));
        assert!(pg.contains("max_dice=2"));
        assert!(pg.contains("max_players=2"));
        let ppo = smoke_ppo_args(&args);
        assert!(ppo.contains("min_dice=1"));
        assert!(ppo.contains("max_dice=2"));
        assert!(ppo.contains("max_players=2"));
    }

    #[test]
    fn full_plan_makes_tournament_budgets_explicit() {
        let args = Args::parse_from(std::iter::empty::<&str>()).unwrap();
        let steps = build_steps(&args);
        for method in ["cheap-search", "baseline"] {
            let step = steps
                .iter()
                .find(|step| step.method == method)
                .unwrap_or_else(|| panic!("missing step {method}"));
            assert!(step.command.contains("mccfr_iters=16"));
            assert!(step.command.contains("solve_iters=256"));
            assert!(step.command.contains("net_search_rollouts=12"));
            assert!(step.command.contains("rebel_iters=16"));
        }

        let population = steps
            .iter()
            .find(|step| step.phase == "meta" && step.method == "population")
            .unwrap();
        assert!(population.command.contains("mccfr_iters=16"));
        assert!(population.command.contains("solve_iters=256"));
    }

    #[test]
    fn full_checkpoint_tournament_uses_broad_screen_budget() {
        let args = Args::parse_from(std::iter::empty::<&str>()).unwrap();
        let steps = build_steps(&args);
        let tournament = steps
            .iter()
            .find(|step| step.phase == "eval" && step.method == "checkpoint-tournament")
            .unwrap();
        assert!(tournament.command.contains("games=40"));
        assert!(tournament.command.contains("rollouts=24"));
        assert!(tournament.command.contains("solve_iters=64"));
        assert!(tournament.command.contains("net_search_rollouts=6"));
        assert!(tournament.command.contains("rebel_iters=8"));
    }

    #[test]
    fn full_plan_defers_unbounded_mccfr_roster_entry() {
        let smoke = Args::parse_from(["smoke=1"]).unwrap();
        let smoke_steps = build_steps(&smoke);
        let smoke_profile = smoke_steps
            .iter()
            .find(|step| step.method == "cheap-search")
            .unwrap();
        assert!(smoke_profile.command.contains(
            "agents=belief,rollout,abstract-rollout,is-mcts,mccfr,qlearn,blueprint-search"
        ));

        let full = Args::parse_from(std::iter::empty::<&str>()).unwrap();
        let full_steps = build_steps(&full);
        for method in ["cheap-search", "baseline", "checkpoint-tournament"] {
            let step = full_steps
                .iter()
                .find(|step| step.method == method)
                .unwrap_or_else(|| panic!("missing step {method}"));
            let roster = step
                .command
                .split(" agents=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .unwrap();
            assert!(!roster.split(',').any(|agent| agent == "mccfr"));
        }

        let population = full_steps
            .iter()
            .find(|step| step.phase == "meta" && step.method == "population")
            .unwrap();
        let fields = population
            .command
            .split(" fields=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .unwrap();
        assert!(!fields.split(',').any(|field| field == "mccfr"));
    }

    #[test]
    fn gates_cover_standalone_ml_crates() {
        let args = Args::parse_from(["smoke=1"]).unwrap();
        let steps = build_steps(&args);
        for method in [
            "ml-rnad-check",
            "ml-rnad-clippy",
            "ml-rnad-test",
            "ml-ppo-check",
            "ml-ppo-clippy",
            "ml-ppo-test",
        ] {
            let step = steps
                .iter()
                .find(|step| step.phase == "gate" && step.method == method)
                .unwrap_or_else(|| panic!("missing gate {method}"));
            assert!(step.command.contains("--manifest-path ml/ld-"));
        }
    }

    #[test]
    fn rebel_winshare_target_tracks_smoke_and_full_targets() {
        let smoke = Args::parse_from(["smoke=1"]).unwrap();
        let smoke_steps = build_steps(&smoke);
        let smoke_rebel = smoke_steps
            .iter()
            .find(|step| step.method == "rebel" && step.scale == Some(1))
            .unwrap();
        assert!(smoke_rebel.command.contains("WINSHARE_PLAYERS=2"));
        assert!(smoke_rebel.command.contains("WINSHARE_DICE=1"));
        assert!(smoke_rebel.command.contains("WINSHARE_FACES=2"));

        let full = Args::parse_from(std::iter::empty::<&str>()).unwrap();
        let full_steps = build_steps(&full);
        let full_rebel = full_steps
            .iter()
            .find(|step| step.method == "rebel" && step.scale == Some(1))
            .unwrap();
        assert!(full_rebel.command.contains("WINSHARE_PLAYERS=5"));
        assert!(full_rebel.command.contains("WINSHARE_DICE=5"));
        assert!(full_rebel.command.contains("WINSHARE_FACES=6"));
    }

    #[test]
    fn full_rebel_base_run_uses_bounded_keepbest_eval() {
        let args = Args::parse_from(std::iter::empty::<&str>()).unwrap();
        let steps = build_steps(&args);
        let rebel = steps
            .iter()
            .find(|step| step.method == "rebel" && step.scale == Some(1))
            .unwrap();
        assert!(rebel.command.contains("STEPS=3000"));
        assert!(rebel.command.contains("EVAL_EVERY=500"));
        assert!(rebel.command.contains("WINSHARE_GAMES=40"));
        assert!(rebel.command.contains("WINSHARE_ROLLOUTS=24"));
        assert!(rebel.command.contains("WINSHARE_ITERS=16"));
    }

    #[test]
    fn bakeoff_scores_against_named_baseline_field() {
        let args = Args::parse_from(["smoke=1"]).unwrap();
        let steps = build_steps(&args);
        let baseline = steps
            .iter()
            .find(|step| step.phase == "eval" && step.method == "baseline")
            .unwrap();
        assert!(baseline.command.contains("baseline-field"));

        let tournament = steps
            .iter()
            .find(|step| step.phase == "eval" && step.method == "checkpoint-tournament")
            .unwrap();
        assert!(tournament.command.contains("baseline-field"));
        assert!(tournament.command.contains("rebel="));
        assert!(tournament.command.contains("/rebel_x1/best.bin"));

        let population = steps
            .iter()
            .find(|step| step.phase == "meta" && step.method == "population")
            .unwrap();
        assert!(population.command.contains("baseline-field"));

        let report = steps
            .iter()
            .find(|step| step.phase == "report" && step.method == "curve-report")
            .unwrap();
        assert!(report.command.contains("field=baseline-field"));
    }

    #[test]
    fn outdir_retargets_only_the_default_manifest() {
        let args = Args::parse_from(["outdir=/tmp/ld_bakeoff"]).unwrap();
        assert_eq!(
            args.manifest.as_deref(),
            Some("/tmp/ld_bakeoff/manifest.jsonl")
        );

        let args =
            Args::parse_from(["manifest=/tmp/custom.jsonl", "outdir=/tmp/ld_bakeoff"]).unwrap();
        assert_eq!(args.manifest.as_deref(), Some("/tmp/custom.jsonl"));
    }
}
