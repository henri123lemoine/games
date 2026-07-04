//! Summarize Liar's Dice training curves and tournament checkpoint sweeps.
//!
//! The report joins training JSONL rows (`train_net` / `deepcfr_train` /
//! `monitored_train` / `rnad_train` / `ld-ppo` / `population`) with
//! `examples/tournament` metrics through checkpoint paths. It can also read
//! `profile=` JSONL rows from the bake-off planner's machine-optimization pass.
//! It prints the profile summary, measured tournament rows, then a conservative
//! Phase-3-style 100x compute extrapolation only for method/config groups with
//! at least three joined positive-duration points.
//!
//!     cargo run --release -p liars-dice --example curve_report -- \
//!         train=runs/ld_deepcfr/metrics.jsonl \
//!         train=runs/ld_rebel/metrics.jsonl \
//!         tournament=runs/ld_tournament_metrics.jsonl
//!
//! Optional:
//!   * `field=rollout-400` scores each checkpoint against one tournament field.
//!   * `field=mean` (default) averages every non-self field cell.
//!   * `profile=path.jsonl` prints machine-profile budget / throughput rows.
//!   * `roster=1` prints `nets=` / `rnads=` / `ppos=` / `rebels=` fragments and an `agents=` line
//!     that includes raw nets, neural-guided `net-searches`, and learned-value `solves`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use game_core::stats::wilson;
use serde_json::Value;

#[derive(Default)]
struct Args {
    trains: Vec<PathBuf>,
    tournaments: Vec<PathBuf>,
    profiles: Vec<PathBuf>,
    field: Option<String>,
    emit_roster: bool,
}

#[derive(Clone, Debug)]
struct TrainPoint {
    method: String,
    checkpoint: String,
    agent: Option<String>,
    compute_units: f64,
    train_wall_s: f64,
    primary_metric: Option<f64>,
    diagnostic: Option<f64>,
    config: String,
}

#[derive(Clone, Debug, Default)]
struct AgentMeta {
    source: String,
    budget_per_move: f64,
    setup_wall_s: f64,
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    games: f64,
    win_share: f64,
    wall_s: f64,
    decisions: f64,
}

#[derive(Clone, Debug)]
struct Measured {
    agent: String,
    source: String,
    method: String,
    train_wall_s: Option<f64>,
    compute_units: Option<f64>,
    primary_metric: Option<f64>,
    diagnostic: Option<f64>,
    config: String,
    budget_per_move: f64,
    setup_wall_s: f64,
    eval_games: f64,
    win_share: f64,
    wilson_lo: f64,
    wilson_hi: f64,
    eval_wall_s: f64,
    decisions: f64,
}

#[derive(Clone, Debug, Default)]
struct ProfileReport {
    search: HashMap<String, SearchProfile>,
    online: Vec<OnlineBudgetProfile>,
}

impl ProfileReport {
    fn is_empty(&self) -> bool {
        self.search.is_empty() && self.online.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
struct SearchProfile {
    agent: String,
    budget_per_move: f64,
    setup_kind: String,
    setup_wall_s: f64,
    setup_units_per_s: f64,
    games: f64,
    share_sum: f64,
    wall_s: f64,
    decisions: f64,
    cells: usize,
}

#[derive(Clone, Debug)]
struct OnlineBudgetProfile {
    kind: String,
    config: String,
    flat_iters: f64,
    restarts: f64,
    effective_iters: f64,
    win_share: Option<f64>,
    fair: Option<f64>,
    ms_per_move: f64,
    wall_s: Option<f64>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let mut train_points = Vec::new();
    for path in &args.trains {
        train_points.extend(read_train_points(path)?);
    }
    let mut profiles = ProfileReport::default();
    for path in &args.profiles {
        read_profile(path, &mut profiles)?;
    }
    if args.emit_roster {
        print_roster_fragments(&train_points);
        if args.tournaments.is_empty() {
            if !profiles.is_empty() {
                print_profile_summary(&profiles);
            }
            return Ok(());
        }
    }
    if args.tournaments.is_empty() {
        if !profiles.is_empty() {
            print_profile_summary(&profiles);
            return Ok(());
        }
        return Err("provide at least one tournament=metrics.jsonl".to_string());
    }

    let mut train_by_checkpoint = HashMap::new();
    let mut population_by_agent = HashMap::new();
    for point in train_points {
        if let Some(agent) = point.agent.clone() {
            population_by_agent.insert(agent, point);
        } else {
            train_by_checkpoint.insert(norm_path(&point.checkpoint), point);
        }
    }

    let mut agents = HashMap::new();
    let mut cells: HashMap<String, Vec<(String, Cell)>> = HashMap::new();
    for path in &args.tournaments {
        read_tournament(path, &mut agents, &mut cells)?;
    }

    let mut measured = Vec::new();
    for (agent, meta) in &agents {
        let selected = selected_cells(agent, &cells, args.field.as_deref());
        if selected.is_empty() {
            continue;
        }
        let games: f64 = selected.iter().map(|(_, c)| c.games).sum();
        let share_sum: f64 = selected.iter().map(|(_, c)| c.win_share * c.games).sum();
        let eval_wall_s: f64 = selected.iter().map(|(_, c)| c.wall_s).sum();
        let decisions: f64 = selected.iter().map(|(_, c)| c.decisions).sum();
        let win_share = share_sum / games.max(1.0);
        let (lo, hi) = wilson(share_sum, games, 1.96);
        let train = train_by_checkpoint
            .get(&norm_path(&meta.source))
            .or_else(|| population_by_agent.get(agent));
        let method = joined_method(agent, train.map(|p| p.method.as_str()));
        let config = joined_config(
            train.map(|p| p.config.as_str()).unwrap_or_default(),
            eval_config(agent, meta.budget_per_move).as_deref(),
        );
        measured.push(Measured {
            agent: agent.clone(),
            source: meta.source.clone(),
            method,
            train_wall_s: train.map(|p| p.train_wall_s),
            compute_units: train.map(|p| p.compute_units),
            primary_metric: train.and_then(|p| p.primary_metric),
            diagnostic: train.and_then(|p| p.diagnostic),
            config,
            budget_per_move: meta.budget_per_move,
            setup_wall_s: meta.setup_wall_s,
            eval_games: games,
            win_share,
            wilson_lo: lo,
            wilson_hi: hi,
            eval_wall_s,
            decisions,
        });
    }
    if measured.is_empty()
        && args
            .field
            .as_deref()
            .is_some_and(|field| !matches!(field, "mean" | "all"))
    {
        return Err(format!(
            "field '{}' not found for any tournament agent",
            args.field.as_deref().unwrap_or_default()
        ));
    }

    measured.sort_by(|a, b| b.win_share.total_cmp(&a.win_share));
    print_profile_summary(&profiles);
    print_measured(&measured, args.field.as_deref().unwrap_or("mean"));
    print_population_selections(&population_by_agent);
    print_roster_coverage(&measured, &population_by_agent);
    print_extrapolation(&measured);
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        field: Some("mean".to_string()),
        ..Args::default()
    };
    for raw in std::env::args().skip(1) {
        let Some((key, value)) = raw.split_once('=') else {
            return Err(format!("expected key=value argument, got '{raw}'"));
        };
        match key {
            "train" | "training" => args.trains.push(PathBuf::from(value)),
            "tournament" | "eval" => args.tournaments.push(PathBuf::from(value)),
            "profile" | "profiles" => args.profiles.push(PathBuf::from(value)),
            "field" => args.field = Some(value.to_string()),
            "roster" | "emit_roster" => args.emit_roster = parse_bool(value)?,
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(args)
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("expected boolean, got '{value}'")),
    }
}

fn read_train_points(path: &Path) -> Result<Vec<TrainPoint>, String> {
    let file = File::open(path).map_err(|e| format!("open '{}': {e}", path.display()))?;
    let mut points = Vec::new();
    let mut best_alias = None;
    let mut config = String::new();
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read '{}': {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse '{}:{}': {e}", path.display(), lineno + 1))?;
        let Some(event) = v.get("event").and_then(Value::as_str) else {
            continue;
        };
        match event {
            "deepcfr_config"
            | "rebel_monitored_config"
            | "distill_config"
            | "teacher_config"
            | "pg_config"
            | "ppo_config" => {
                config = config_summary(event, &v);
            }
            "deepcfr_block" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: "deepcfr".to_string(),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "iters_done").unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "total_wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winshare_mean"),
                        diagnostic: None,
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "eval" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: "rebel".to_string(),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "samples").unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winshare")
                            .or_else(|| f64_field(&v, "winshare_5p5d6f"))
                            .or_else(|| f64_field(&v, "winshare_3p3d4f")),
                        diagnostic: f64_field(&v, "expl_2p2d3f"),
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "distill_iter" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: "distill".to_string(),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "samples")
                            .or_else(|| f64_field(&v, "iters_done"))
                            .unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "total_wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winrate_mean"),
                        diagnostic: f64_field(&v, "exploitability"),
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "teacher_iter" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: "teacher".to_string(),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "states")
                            .or_else(|| f64_field(&v, "iters_done"))
                            .unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "total_wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winshare_mean"),
                        diagnostic: f64_field(&v, "teacher_gap").or_else(|| f64_field(&v, "mse")),
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "pg_iter" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: str_field(&v, "method").unwrap_or_else(|| "rnad".to_string()),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "decisions")
                            .or_else(|| f64_field(&v, "episodes"))
                            .or_else(|| f64_field(&v, "iters_done"))
                            .unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "total_wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winrate_mean"),
                        diagnostic: f64_field(&v, "exploitability")
                            .or_else(|| f64_field(&v, "value_mse")),
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "ppo_iter" => {
                if let Some(checkpoint) = str_field(&v, "checkpoint") {
                    let point = TrainPoint {
                        method: "ppo".to_string(),
                        checkpoint,
                        agent: None,
                        compute_units: f64_field(&v, "transitions")
                            .or_else(|| f64_field(&v, "iters_done"))
                            .unwrap_or(0.0),
                        train_wall_s: f64_field(&v, "total_wall_s").unwrap_or(0.0),
                        primary_metric: f64_field(&v, "winrate_mean"),
                        diagnostic: f64_field(&v, "exploitability")
                            .or_else(|| f64_field(&v, "value_loss")),
                        config: config.clone(),
                    };
                    push_train_point(&mut points, &mut best_alias, point, &v);
                }
            }
            "population_selection" => {
                if let Some(checkpoint) = str_field(&v, "source") {
                    let selected_method =
                        str_field(&v, "method").unwrap_or_else(|| "selected".to_string());
                    points.push(TrainPoint {
                        method: format!("population-{selected_method}"),
                        checkpoint,
                        agent: str_field(&v, "selected"),
                        compute_units: f64_field(&v, "eval_games")
                            .or_else(|| f64_field(&v, "games"))
                            .unwrap_or(0.0),
                        train_wall_s: 0.0,
                        primary_metric: f64_field(&v, "win_share"),
                        diagnostic: None,
                        config: population_config_summary(&v),
                    });
                }
            }
            _ => {}
        }
    }
    if let Some(alias) = best_alias {
        points.push(alias);
    }
    Ok(points)
}

fn push_train_point(
    points: &mut Vec<TrainPoint>,
    best_alias: &mut Option<TrainPoint>,
    point: TrainPoint,
    row: &Value,
) {
    if bool_field(row, "is_best").unwrap_or(false)
        && let Some(best_checkpoint) =
            str_field(row, "best_checkpoint").or_else(|| derived_best_checkpoint(&point.checkpoint))
    {
        let mut alias = point.clone();
        alias.checkpoint = best_checkpoint;
        *best_alias = Some(alias);
    }
    points.push(point);
}

fn derived_best_checkpoint(checkpoint: &str) -> Option<String> {
    let parent = Path::new(checkpoint).parent()?;
    Some(parent.join("best.bin").to_string_lossy().to_string())
}

fn read_tournament(
    path: &Path,
    agents: &mut HashMap<String, AgentMeta>,
    cells: &mut HashMap<String, Vec<(String, Cell)>>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("open '{}': {e}", path.display()))?;
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read '{}': {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse '{}:{}': {e}", path.display(), lineno + 1))?;
        match v.get("event").and_then(Value::as_str) {
            Some("tournament_agent") => {
                let agent = str_field(&v, "agent").unwrap_or_default();
                agents.insert(
                    agent,
                    AgentMeta {
                        source: str_field(&v, "source").unwrap_or_default(),
                        budget_per_move: f64_field(&v, "budget_per_move").unwrap_or(0.0),
                        setup_wall_s: f64_field(&v, "setup_wall_s").unwrap_or(0.0),
                    },
                );
            }
            Some("tournament_cell") => {
                let hero = str_field(&v, "hero").unwrap_or_default();
                let field = str_field(&v, "field").unwrap_or_default();
                cells.entry(hero).or_default().push((
                    field,
                    Cell {
                        games: f64_field(&v, "games").unwrap_or(0.0),
                        win_share: f64_field(&v, "win_share").unwrap_or(0.0),
                        wall_s: f64_field(&v, "wall_s").unwrap_or(0.0),
                        decisions: f64_field(&v, "decisions").unwrap_or(0.0),
                    },
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn read_profile(path: &Path, report: &mut ProfileReport) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("open '{}': {e}", path.display()))?;
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read '{}': {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| format!("parse '{}:{}': {e}", path.display(), lineno + 1))?;
        match v.get("event").and_then(Value::as_str) {
            Some("tournament_agent") => {
                let agent = str_field(&v, "agent").unwrap_or_default();
                if agent.is_empty() {
                    continue;
                }
                let row = report
                    .search
                    .entry(agent.clone())
                    .or_insert_with(|| SearchProfile {
                        agent: agent.clone(),
                        ..SearchProfile::default()
                    });
                row.budget_per_move = f64_field(&v, "budget_per_move").unwrap_or(0.0);
                row.setup_kind = str_field(&v, "setup_kind").unwrap_or_default();
                row.setup_wall_s = f64_field(&v, "setup_wall_s").unwrap_or(0.0);
                row.setup_units_per_s = f64_field(&v, "setup_units_per_s").unwrap_or(0.0);
            }
            Some("tournament_cell") => {
                let agent = str_field(&v, "hero").unwrap_or_default();
                if agent.is_empty() {
                    continue;
                }
                let row = report
                    .search
                    .entry(agent.clone())
                    .or_insert_with(|| SearchProfile {
                        agent: agent.clone(),
                        ..SearchProfile::default()
                    });
                let games = f64_field(&v, "games").unwrap_or(0.0);
                row.games += games;
                row.share_sum += f64_field(&v, "win_share").unwrap_or(0.0) * games;
                row.wall_s += f64_field(&v, "wall_s").unwrap_or(0.0);
                row.decisions += f64_field(&v, "decisions").unwrap_or(0.0);
                row.cells += 1;
            }
            Some("budget_sweep" | "budget_big_move") => {
                let kind = v
                    .get("event")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                report.online.push(OnlineBudgetProfile {
                    kind,
                    config: str_field(&v, "config").unwrap_or_default(),
                    flat_iters: f64_field(&v, "flat_iters").unwrap_or(0.0),
                    restarts: f64_field(&v, "restarts").unwrap_or(0.0),
                    effective_iters: f64_field(&v, "effective_iters").unwrap_or(0.0),
                    win_share: f64_field(&v, "win_share"),
                    fair: f64_field(&v, "fair"),
                    ms_per_move: f64_field(&v, "ms_per_move").unwrap_or(0.0),
                    wall_s: f64_field(&v, "wall_s"),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn print_roster_fragments(points: &[TrainPoint]) {
    if points.is_empty() {
        println!("no training checkpoints found for roster fragments");
        println!();
        return;
    }
    let mut nets = Vec::new();
    let mut rnads = Vec::new();
    let mut ppos = Vec::new();
    let mut histories = Vec::new();
    let mut rebels = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for point in points {
        let base = format!(
            "{}_{}",
            point.method,
            compact_number(point.compute_units.max(point.train_wall_s))
        );
        let count = counts.entry(base.clone()).or_insert(0);
        *count += 1;
        let label = if *count == 1 {
            clean_label(&base)
        } else {
            clean_label(&format!("{base}_{count}"))
        };
        let entry = format!("{label}:{}", point.checkpoint);
        match point.method.as_str() {
            "deepcfr" | "distill" | "teacher" => nets.push(entry),
            "rnad" => rnads.push(entry),
            "history-rnad" => histories.push(entry),
            "ppo" => ppos.push(entry),
            "rebel" => rebels.push(entry),
            method if method.starts_with("population-rnad") => rnads.push(entry),
            method if method.starts_with("population-history") => histories.push(entry),
            method if method.starts_with("population-ppo") => ppos.push(entry),
            method if method.starts_with("population-rebel") => rebels.push(entry),
            method if method.starts_with("population-") => nets.push(entry),
            _ => {}
        }
    }

    println!("tournament roster fragments from training metrics:");
    if !nets.is_empty() {
        println!("  nets={}", nets.join(","));
    }
    if !rnads.is_empty() {
        println!("  rnads={}", rnads.join(","));
    }
    if !ppos.is_empty() {
        println!("  ppos={}", ppos.join(","));
    }
    if !histories.is_empty() {
        println!("  histories={}", histories.join(","));
    }
    if !rebels.is_empty() {
        println!("  rebels={}", rebels.join(","));
    }
    if !nets.is_empty()
        || !rnads.is_empty()
        || !ppos.is_empty()
        || !histories.is_empty()
        || !rebels.is_empty()
    {
        let mut agents = vec!["random", "honest-bayes", "belief", "rollout"];
        if !nets.is_empty() {
            agents.push("nets");
            agents.push("net-searches");
            agents.push("solves");
        }
        if !rnads.is_empty() {
            agents.push("rnads");
            if !agents.contains(&"solves") {
                agents.push("solves");
            }
        }
        if !ppos.is_empty() {
            agents.push("ppos");
            if !agents.contains(&"solves") {
                agents.push("solves");
            }
        }
        if !histories.is_empty() {
            agents.push("histories");
        }
        if !rebels.is_empty() {
            agents.push("rebels");
        }
        println!("  agents={}", agents.join(","));
    }
    println!();
}

fn config_summary(event: &str, v: &Value) -> String {
    match event {
        "deepcfr_config" => format!(
            "deepcfr hidden={} threads={} block={} traversals={} train_every={} eval={}x{} seed={}",
            value_text(v, "hidden"),
            value_text(v, "threads"),
            value_text(v, "block"),
            value_text(v, "traversals"),
            value_text(v, "train_every"),
            value_text(v, "eval_rollouts"),
            value_text(v, "eval_games"),
            value_text(v, "seed"),
        ),
        "distill_config" => format!(
            "distill hidden={} threads={} rounds={} playouts={} cfr={} es={} warmup={} val_every={} eval={}x{} seed={}",
            value_text(v, "hidden"),
            value_text(v, "threads"),
            value_text(v, "rounds_per_iter"),
            value_text(v, "playouts"),
            value_text(v, "cfr_iters"),
            value_text(v, "es_iters"),
            value_text(v, "warmup_iters"),
            value_text(v, "val_every"),
            value_text(v, "eval_rollouts"),
            value_text(v, "eval_games"),
            value_text(v, "seed"),
        ),
        "teacher_config" => format!(
            "teacher base={} target={}p{}d{}f states={} rollouts={} plies={} cap={} temp={} eval={}x{} seed={}",
            teacher_base_family(v),
            value_text(v, "players"),
            value_text(v, "dice"),
            value_text(v, "faces"),
            value_text(v, "states_per_iter"),
            value_text(v, "rollouts"),
            value_text(v, "plies"),
            value_text(v, "max_search_actions"),
            value_text(v, "temperature"),
            value_text(v, "eval_rollouts"),
            value_text(v, "eval_games"),
            value_text(v, "seed"),
        ),
        "rebel_monitored_config" => format!(
            "rebel hidden={} iters={} depth={} gen_per={} train_ratio={} eval_every={} winshare_target={} winshare={}x{} seed={}",
            value_text(v, "hidden"),
            value_text(v, "num_iters"),
            value_text(v, "depth"),
            value_text(v, "gen_per"),
            value_text(v, "train_ratio"),
            value_text(v, "eval_every"),
            rebel_winshare_target(v),
            value_text(v, "winshare_rollouts"),
            value_text(v, "winshare_games"),
            value_text(v, "seed"),
        ),
        "pg_config" => format!(
            "rnad arch={} target={}p{}d{}f train={} hidden={} episodes={} entropy={} anchor={} eval={}x{} expl={} seed={}",
            value_text(v, "architecture"),
            value_text(v, "players"),
            value_text(v, "dice"),
            value_text(v, "faces"),
            pg_train_range(v),
            value_text(v, "hidden"),
            value_text(v, "episodes_per_iter"),
            value_text(v, "entropy"),
            value_text(v, "anchor"),
            value_text(v, "eval_rollouts"),
            value_text(v, "eval_games"),
            value_text(v, "eval_exploitability"),
            value_text(v, "seed"),
        ),
        "ppo_config" => format!(
            "ppo target={}p{}d{}f train={} hidden={} actors={} steps={} epochs={} minibatches={} lr={} entropy={} eval={}x{} expl={} seed={}",
            value_text(v, "players"),
            value_text(v, "dice"),
            value_text(v, "faces"),
            pg_train_range(v),
            value_text(v, "hidden"),
            value_text(v, "actors"),
            value_text(v, "steps"),
            value_text(v, "epochs"),
            value_text(v, "minibatches"),
            value_text(v, "lr"),
            value_text(v, "entropy_coef"),
            value_text(v, "eval_rollouts"),
            value_text(v, "eval_games"),
            value_text(v, "eval_exploitability"),
            value_text(v, "seed"),
        ),
        _ => String::new(),
    }
}

fn teacher_base_family(v: &Value) -> String {
    let Some(base) = str_field(v, "base") else {
        return "-".to_string();
    };
    let parent = Path::new(&base)
        .parent()
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        .unwrap_or(base.as_str());
    strip_scale_suffix(parent).to_string()
}

fn rebel_winshare_target(v: &Value) -> String {
    let Some(players) = v.get("winshare_players") else {
        return "3p3d4f".to_string();
    };
    format!(
        "{}p{}d{}f",
        players,
        value_text(v, "winshare_dice"),
        value_text(v, "winshare_faces")
    )
}

fn strip_scale_suffix(value: &str) -> &str {
    let Some((stem, scale)) = value.rsplit_once("_x") else {
        return value;
    };
    if !scale.is_empty() && scale.chars().all(|c| c.is_ascii_digit()) {
        stem
    } else {
        value
    }
}

fn pg_train_range(v: &Value) -> String {
    if v.get("mixed").and_then(Value::as_bool).unwrap_or(false) {
        format!(
            "{}-{}p{}-{}d{}-{}f",
            value_text(v, "min_players"),
            value_text(v, "max_players"),
            value_text(v, "min_dice"),
            value_text(v, "max_dice"),
            value_text(v, "min_faces"),
            value_text(v, "max_faces"),
        )
    } else {
        format!(
            "{}p{}d{}f",
            value_text(v, "players"),
            value_text(v, "dice"),
            value_text(v, "faces"),
        )
    }
}

fn population_config_summary(v: &Value) -> String {
    format!(
        "population selected={} source={} heldout={}p{}d{}f eval_games={} score={} ci=[{},{}]",
        value_text(v, "selected"),
        value_text(v, "source"),
        value_text(v, "players"),
        value_text(v, "dice"),
        value_text(v, "faces"),
        value_text(v, "eval_games"),
        value_text(v, "win_share"),
        value_text(v, "wilson_lo"),
        value_text(v, "wilson_hi"),
    )
}

fn selected_cells(
    agent: &str,
    cells: &HashMap<String, Vec<(String, Cell)>>,
    field: Option<&str>,
) -> Vec<(String, Cell)> {
    let all = cells.get(agent).cloned().unwrap_or_default();
    match field.unwrap_or("mean") {
        "mean" | "all" => all
            .into_iter()
            .filter(|(field, _)| field != agent)
            .collect(),
        wanted => all
            .into_iter()
            .filter(|(field, _)| field == wanted)
            .collect(),
    }
}

fn print_profile_summary(report: &ProfileReport) {
    if report.is_empty() {
        return;
    }
    if !report.search.is_empty() {
        let mut rows: Vec<_> = report.search.values().collect();
        rows.sort_by(|a, b| {
            a.budget_per_move
                .total_cmp(&b.budget_per_move)
                .then_with(|| a.agent.cmp(&b.agent))
        });
        println!("machine profile: cheap/search agents");
        println!(
            "  {:<24} {:>8} {:<16} {:>8} {:>10} {:>8} {:>8} {:>9} {:>9}",
            "agent", "budget", "setup", "setup_s", "setup_u/s", "games", "share", "eval_s", "dec/s"
        );
        println!("  {}", "-".repeat(118));
        for row in rows {
            let share = (row.games > 0.0).then_some(row.share_sum / row.games.max(1.0));
            let dec_s = (row.wall_s > 0.0).then_some(row.decisions / row.wall_s.max(1e-9));
            println!(
                "  {:<24} {:>8} {:<16} {:>8.3} {:>10} {:>8.0} {:>8} {:>9.3} {:>9}",
                row.agent,
                compact_budget(row.budget_per_move),
                if row.setup_kind.is_empty() {
                    "-"
                } else {
                    row.setup_kind.as_str()
                },
                row.setup_wall_s,
                fmt_opt(Some(row.setup_units_per_s)),
                row.games,
                fmt_opt(share),
                row.wall_s,
                fmt_opt(dec_s),
            );
        }
        println!();
    }

    if !report.online.is_empty() {
        let mut rows = report.online.clone();
        rows.sort_by(|a, b| {
            a.config
                .cmp(&b.config)
                .then_with(|| a.effective_iters.total_cmp(&b.effective_iters))
                .then_with(|| a.kind.cmp(&b.kind))
        });
        println!("machine profile: blueprint-search budget");
        println!(
            "  {:<16} {:<16} {:>9} {:>8} {:>9} {:>8} {:>8} {:>9} {:>8}",
            "event", "config", "flat", "restarts", "eff", "share", "fair", "ms/move", "wall_s"
        );
        println!("  {}", "-".repeat(111));
        for row in rows {
            println!(
                "  {:<16} {:<16} {:>9} {:>8} {:>9} {:>8} {:>8} {:>9.3} {:>8}",
                row.kind,
                row.config,
                compact_budget(row.flat_iters),
                compact_budget(row.restarts),
                compact_budget(row.effective_iters),
                fmt_opt(row.win_share),
                fmt_opt(row.fair),
                row.ms_per_move,
                fmt_opt(row.wall_s),
            );
        }
        println!();
    }
}

fn print_measured(rows: &[Measured], field: &str) {
    println!("measured tournament checkpoints (field={field}):");
    println!(
        "  {:<24} {:<18} {:>10} {:>9} {:>8} {:>8} {:>7} {:>8} {:>18} {:>9} {:>9} {:>9} {:>10}",
        "agent",
        "method",
        "train_s",
        "units",
        "metric",
        "diag",
        "budget",
        "load_s",
        "win_share 95% CI",
        "games",
        "eval_s",
        "dec/s",
        "source"
    );
    println!("  {}", "-".repeat(169));
    for row in rows {
        let dec_s = row.decisions / row.eval_wall_s.max(1e-9);
        println!(
            "  {:<24} {:<18} {:>10} {:>9} {:>8} {:>8} {:>7.0} {:>8.3} {:>6.3} [{:>.3},{:>.3}] {:>9.0} {:>9.2} {:>9.0} {:>10}",
            row.agent,
            row.method,
            fmt_opt(row.train_wall_s),
            fmt_opt(row.compute_units),
            fmt_opt(row.primary_metric),
            fmt_opt(row.diagnostic),
            row.budget_per_move,
            row.setup_wall_s,
            row.win_share,
            row.wilson_lo,
            row.wilson_hi,
            row.eval_games,
            row.eval_wall_s,
            dec_s,
            compact_source(&row.source),
        );
    }
    let configs: Vec<_> = rows
        .iter()
        .filter(|row| !row.config.is_empty())
        .map(|row| (row.agent.as_str(), row.config.as_str()))
        .collect();
    if !configs.is_empty() {
        println!();
        println!("training configs:");
        for (agent, config) in configs {
            println!("  {agent}: {config}");
        }
    }
    println!();
}

fn print_population_selections(population_by_agent: &HashMap<String, TrainPoint>) {
    if population_by_agent.is_empty() {
        return;
    }
    let mut rows: Vec<_> = population_by_agent.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    println!("population selections:");
    println!(
        "  {:<34} {:<18} {:>9} {:>8} config",
        "selected", "method", "games", "score"
    );
    println!("  {}", "-".repeat(96));
    for (agent, point) in rows {
        println!(
            "  {:<34} {:<18} {:>9.0} {:>8} {}",
            agent,
            point.method,
            point.compute_units,
            fmt_opt(point.primary_metric),
            point.config
        );
    }
    println!();
}

#[derive(Clone, Debug)]
struct RosterRequirement {
    id: &'static str,
    label: &'static str,
    exact: &'static [&'static str],
    prefixes: &'static [&'static str],
}

fn roster_requirements() -> [RosterRequirement; 13] {
    [
        RosterRequirement {
            id: "C1",
            label: "heuristic exact-Bayes",
            exact: &["honest-bayes"],
            prefixes: &[],
        },
        RosterRequirement {
            id: "C2",
            label: "belief + rollout",
            exact: &["rollout", "abstract-rollout"],
            prefixes: &["rollout+", "abstract-rollout+"],
        },
        RosterRequirement {
            id: "C3",
            label: "IS-MCTS",
            exact: &["is-mcts"],
            prefixes: &["is-mcts+"],
        },
        RosterRequirement {
            id: "C4",
            label: "MCCFR/DCFR",
            exact: &["mccfr"],
            prefixes: &["mccfr+"],
        },
        RosterRequirement {
            id: "C5",
            label: "Deep CFR",
            exact: &["deepcfr"],
            prefixes: &["deepcfr+"],
        },
        RosterRequirement {
            id: "C6",
            label: "Q-learning",
            exact: &["qlearn"],
            prefixes: &["qlearn+"],
        },
        RosterRequirement {
            id: "C7",
            label: "ReBeL",
            exact: &["rebel"],
            prefixes: &["rebel+"],
        },
        RosterRequirement {
            id: "C8",
            label: "blueprint search",
            exact: &["blueprint-search"],
            prefixes: &["blueprint-search+"],
        },
        RosterRequirement {
            id: "C9",
            label: "R-NaD/NeuRD",
            exact: &["rnad"],
            prefixes: &["rnad+"],
        },
        RosterRequirement {
            id: "C10",
            label: "PPO",
            exact: &["ppo"],
            prefixes: &["ppo+"],
        },
        RosterRequirement {
            id: "C11",
            label: "population/PSRO",
            exact: &[],
            prefixes: &["population-"],
        },
        RosterRequirement {
            id: "C12",
            label: "search teacher",
            exact: &["teacher"],
            prefixes: &["teacher+"],
        },
        RosterRequirement {
            id: "C13",
            label: "history attention",
            exact: &["history-rnad", "history"],
            prefixes: &["history-rnad+", "history+"],
        },
    ]
}

fn method_matches_requirement(method: &str, req: &RosterRequirement) -> bool {
    req.exact.contains(&method) || req.prefixes.iter().any(|prefix| method.starts_with(prefix))
}

fn print_roster_coverage(rows: &[Measured], population_by_agent: &HashMap<String, TrainPoint>) {
    println!("roadmap roster coverage (from measured rows + population selections):");
    println!("  {:<4} {:<22} {:<8} evidence", "id", "contender", "status");
    println!("  {}", "-".repeat(74));
    for req in roster_requirements() {
        let mut agents = Vec::new();
        for row in rows {
            if method_matches_requirement(&row.method, &req) && !agents.contains(&row.agent) {
                agents.push(row.agent.clone());
            }
        }
        for (agent, point) in population_by_agent {
            if method_matches_requirement(&point.method, &req) && !agents.contains(agent) {
                agents.push(agent.clone());
            }
        }
        agents.sort();
        let status = if agents.is_empty() { "missing" } else { "seen" };
        let evidence = if agents.is_empty() {
            "-".to_string()
        } else {
            agents.join(", ")
        };
        println!(
            "  {:<4} {:<22} {:<8} {}",
            req.id, req.label, status, evidence
        );
    }
    println!();
}

fn print_extrapolation(rows: &[Measured]) {
    let best_measured = rows
        .iter()
        .map(|r| r.win_share)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut by_group: HashMap<(String, String), Vec<&Measured>> = HashMap::new();
    for row in rows {
        if row.train_wall_s.is_some() {
            by_group
                .entry((row.method.clone(), row.config.clone()))
                .or_default()
                .push(row);
        }
    }

    println!("100x compute extrapolation (log-linear, grouped by method+config):");
    if best_measured.is_finite() {
        println!("  decision threshold: best measured win-share = {best_measured:.3}");
    }
    println!(
        "  {:<18} {:<28} {:>3} {:>10} {:>10} {:<28} {:>8} {:>17} {:>12}",
        "method", "config", "n", "best", "1h", "curve", "slope", "est_100x 95% CI", "verdict"
    );
    println!("  {}", "-".repeat(149));
    let mut groups: Vec<_> = by_group.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    for ((method, config), mut points) in groups {
        points.sort_by(|a, b| {
            a.train_wall_s
                .unwrap_or(0.0)
                .total_cmp(&b.train_wall_s.unwrap_or(0.0))
        });
        let fit_points: Vec<_> = points
            .iter()
            .copied()
            .filter(|p| p.train_wall_s.is_some_and(|x| x > 0.0))
            .collect();
        let best = points
            .iter()
            .map(|r| r.win_share)
            .fold(f64::NEG_INFINITY, f64::max);
        let curve = curve_summary(&fit_points);
        let config = short_config(&config);
        if fit_points.len() < 3 {
            println!(
                "  {:<18} {:<28} {:>3} {:>10.3} {:>10} {:<28} {:>8} {:>17} {:>12}",
                method,
                config,
                fit_points.len(),
                best,
                "n/a",
                curve,
                "n/a",
                "need >=3 positive points",
                "uncertain"
            );
            continue;
        }
        let fit = fit_log_linear(&fit_points);
        if fit.sxx <= 0.0 {
            println!(
                "  {:<18} {:<28} {:>3} {:>10.3} {:>10} {:<28} {:>8} {:>17} {:>12}",
                method,
                config,
                fit_points.len(),
                best,
                "n/a",
                curve,
                "n/a",
                "need distinct compute",
                "uncertain"
            );
            continue;
        }
        let max_s = fit_points
            .iter()
            .filter_map(|r| r.train_wall_s)
            .fold(0.0, f64::max);
        let (one_h, _one_h_lo, _one_h_hi) = prediction_interval(&fit, 3600.0_f64.ln());
        let target_x = (max_s * 100.0).ln();
        let (estimate, lo, hi) = prediction_interval(&fit, target_x);
        let verdict = if lo > best_measured {
            "yes"
        } else if hi <= best_measured {
            "no"
        } else {
            "uncertain"
        };
        println!(
            "  {:<18} {:<28} {:>3} {:>10.3} {:>10.3} {:<28} {:>8.4} {:>6.3} [{:>.3},{:>.3}] {:>12}",
            method,
            config,
            fit_points.len(),
            best,
            one_h,
            curve,
            fit.slope,
            estimate,
            lo,
            hi,
            verdict
        );
    }
    println!();
    if rows.iter().all(|r| r.train_wall_s.is_none()) {
        println!(
            "note: no tournament agents matched training checkpoint paths; extrapolation is unavailable."
        );
    }
}

fn prediction_interval(fit: &Fit, x: f64) -> (f64, f64, f64) {
    let estimate = (fit.intercept + fit.slope * x).clamp(0.0, 1.0);
    let half = 1.96 * fit.prediction_se(x);
    (
        estimate,
        (estimate - half).clamp(0.0, 1.0),
        (estimate + half).clamp(0.0, 1.0),
    )
}

fn curve_summary(points: &[&Measured]) -> String {
    if points.is_empty() {
        return "-".to_string();
    }
    let mut pieces = Vec::new();
    for point in points.iter().take(4) {
        pieces.push(format!(
            "{}:{:.3}",
            compact_duration(point.train_wall_s.unwrap_or(0.0)),
            point.win_share
        ));
    }
    if points.len() > 4 {
        pieces.push(format!("+{}", points.len() - 4));
    }
    pieces.join(",")
}

fn compact_duration(seconds: f64) -> String {
    if seconds >= 3600.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{seconds:.0}s")
    }
}

fn short_config(config: &str) -> String {
    if config.is_empty() {
        return "-".to_string();
    }
    const MAX: usize = 28;
    if config.chars().count() <= MAX {
        config.to_string()
    } else {
        let mut s: String = config.chars().take(MAX - 3).collect();
        s.push_str("...");
        s
    }
}

struct Fit {
    intercept: f64,
    slope: f64,
    mean_x: f64,
    sxx: f64,
    sigma: f64,
    n: usize,
}

impl Fit {
    fn prediction_se(&self, x: f64) -> f64 {
        if self.n <= 2 || self.sxx <= 0.0 {
            return f64::INFINITY;
        }
        self.sigma * (1.0 + 1.0 / self.n as f64 + (x - self.mean_x).powi(2) / self.sxx).sqrt()
    }
}

fn fit_log_linear(points: &[&Measured]) -> Fit {
    let xs: Vec<f64> = points
        .iter()
        .map(|p| p.train_wall_s.unwrap_or(1e-9).max(1e-9).ln())
        .collect();
    let ys: Vec<f64> = points.iter().map(|p| p.win_share).collect();
    let n = xs.len();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let sxx = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let sxy = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = mean_y - slope * mean_x;
    let rss = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum::<f64>();
    let sigma = if n > 2 {
        (rss / (n - 2) as f64).sqrt()
    } else {
        f64::INFINITY
    };
    Fit {
        intercept,
        slope,
        mean_x,
        sxx,
        sigma,
        n,
    }
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(ToOwned::to_owned)
}

fn f64_field(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn bool_field(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}

fn value_text(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "-".to_string(),
    }
}

fn norm_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| PathBuf::from(path))
        .to_string_lossy()
        .to_string()
}

fn joined_method(agent: &str, train_method: Option<&str>) -> String {
    let base = train_method
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| infer_method(agent));
    if is_net_search_agent(agent) && !method_has_wrapper(&base, "net-search") {
        format!("{base}+net-search")
    } else if is_solve_agent(agent) && !method_has_wrapper(&base, "solve") {
        format!("{base}+solve")
    } else {
        base
    }
}

fn method_has_wrapper(method: &str, wrapper: &str) -> bool {
    method == wrapper || method.ends_with(&format!("+{wrapper}"))
}

fn joined_config(train_config: &str, eval_config: Option<&str>) -> String {
    match (train_config.is_empty(), eval_config) {
        (_, Some("")) => train_config.to_string(),
        (true, Some(eval)) => eval.to_string(),
        (false, Some(eval)) => format!("{train_config} {eval}"),
        (_, None) => train_config.to_string(),
    }
}

fn eval_config(agent: &str, budget_per_move: f64) -> Option<String> {
    if let Some(suffix) = net_search_suffix(agent) {
        return Some(format!(
            "eval=net-search {suffix} budget={}",
            compact_budget(budget_per_move)
        ));
    }
    if let Some(suffix) = blueprint_search_suffix(agent) {
        return Some(format!(
            "eval=blueprint-search {suffix} budget={}",
            compact_budget(budget_per_move)
        ));
    }
    solve_suffix(agent).map(|suffix| {
        format!(
            "eval=learned-solve {suffix} budget={}",
            compact_budget(budget_per_move)
        )
    })
}

fn infer_method(agent: &str) -> String {
    if is_solve_agent(agent) {
        return infer_wrapped_base_method(agent, "solve")
            .map(|base| format!("{base}+solve"))
            .unwrap_or_else(|| "solve".to_string());
    }
    if is_net_search_agent(agent) {
        return infer_wrapped_base_method(agent, "net-search")
            .map(|base| format!("{base}+net-search"))
            .unwrap_or_else(|| "net-search".to_string());
    }
    if let Some(base) = infer_learned_base_method(agent) {
        return base.to_string();
    }
    if agent.starts_with("online-solve-") {
        return "online-solve".to_string();
    }
    if agent.starts_with("blueprint-search-") {
        return "blueprint-search".to_string();
    }
    if agent.starts_with("ab-rollout-") {
        return "abstract-rollout".to_string();
    }
    if agent.starts_with("rollout-") {
        return "rollout".to_string();
    }
    if agent.starts_with("is-mcts-") {
        return "is-mcts".to_string();
    }
    if agent.starts_with("mccfr-") {
        return "mccfr".to_string();
    }
    if agent.starts_with("qlearn-") {
        return "qlearn".to_string();
    }
    if agent.starts_with("rebel-") {
        return "rebel".to_string();
    }
    if agent.starts_with("rnad-") {
        return "rnad".to_string();
    }
    if agent.starts_with("ppo-") {
        return "ppo".to_string();
    }
    if agent.starts_with("history-") {
        return "history-rnad".to_string();
    }
    if agent.starts_with("honest-bayes") {
        return "honest-bayes".to_string();
    }
    if agent.starts_with("aggressive-bluffer") {
        return "aggressive-bluffer".to_string();
    }
    if agent.starts_with("conservative-caller") {
        return "conservative-caller".to_string();
    }
    agent.split('-').next().unwrap_or(agent).to_string()
}

fn infer_wrapped_base_method<'a>(agent: &'a str, wrapper: &str) -> Option<&'a str> {
    let rest = agent.strip_prefix(wrapper)?.strip_prefix('-')?;
    let stem = strip_eval_suffix(rest);
    infer_learned_base_method(stem)
}

fn strip_eval_suffix(agent: &str) -> &str {
    let Some((stem, suffix)) = agent.rsplit_once('-') else {
        return agent;
    };
    let Some((left, right)) = suffix.split_once('x') else {
        return agent;
    };
    if !left.is_empty()
        && !right.is_empty()
        && left.chars().all(|c| c.is_ascii_digit())
        && right.chars().all(|c| c.is_ascii_digit())
    {
        stem
    } else {
        agent
    }
}

fn infer_learned_base_method(agent: &str) -> Option<&'static str> {
    let stem = agent.strip_prefix("net-").unwrap_or(agent);
    if stem.starts_with("deepcfr") {
        Some("deepcfr")
    } else if stem.starts_with("distill") {
        Some("distill")
    } else if stem.starts_with("teacher") {
        Some("teacher")
    } else if stem.starts_with("rnad") {
        Some("rnad")
    } else if stem.starts_with("ppo") {
        Some("ppo")
    } else if stem.starts_with("history") {
        Some("history-rnad")
    } else if stem.starts_with("rebel") {
        Some("rebel")
    } else {
        None
    }
}

fn is_net_search_agent(agent: &str) -> bool {
    agent == "net-search" || agent.starts_with("net-search-")
}

fn is_solve_agent(agent: &str) -> bool {
    agent == "solve" || agent.starts_with("solve-")
}

fn net_search_suffix(agent: &str) -> Option<&str> {
    if !is_net_search_agent(agent) {
        return None;
    }
    let suffix = agent.rsplit('-').next()?;
    let (rollouts, plies) = suffix.split_once('x')?;
    if !rollouts.is_empty()
        && !plies.is_empty()
        && rollouts.chars().all(|c| c.is_ascii_digit())
        && plies.chars().all(|c| c.is_ascii_digit())
    {
        Some(suffix)
    } else {
        None
    }
}

fn blueprint_search_suffix(agent: &str) -> Option<&str> {
    if !agent.starts_with("blueprint-search-") {
        return None;
    }
    solve_like_suffix(agent)
}

fn solve_suffix(agent: &str) -> Option<&str> {
    if !is_solve_agent(agent) {
        return None;
    }
    solve_like_suffix(agent)
}

fn solve_like_suffix(agent: &str) -> Option<&str> {
    let suffix = agent.rsplit('-').next()?;
    let (iters, restarts) = suffix.split_once('x')?;
    if !iters.is_empty()
        && !restarts.is_empty()
        && iters.chars().all(|c| c.is_ascii_digit())
        && restarts.chars().all(|c| c.is_ascii_digit())
    {
        Some(suffix)
    } else {
        None
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| {
            if v >= 10_000.0 {
                format!("{v:.0}")
            } else {
                format!("{v:.1}")
            }
        })
        .unwrap_or_else(|| "-".to_string())
}

fn compact_number(value: f64) -> String {
    if !value.is_finite() || value <= 0.0 {
        "0".to_string()
    } else if value >= 1000.0 || (value - value.round()).abs() < 1e-6 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}").replace('.', "p")
    }
}

fn compact_budget(value: f64) -> String {
    if !value.is_finite() {
        "-".to_string()
    } else if (value - value.round()).abs() < 1e-6 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn clean_label(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn compact_source(source: &str) -> String {
    if source.is_empty() {
        return "-".to_string();
    }
    Path::new(source)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(source)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_search_checkpoint_rows_get_distinct_method_and_eval_config() {
        assert_eq!(
            joined_method("net-search-deepcfr_100-48x3", Some("deepcfr")),
            "deepcfr+net-search"
        );
        assert_eq!(
            eval_config("net-search-deepcfr_100-1x1", 8.0).as_deref(),
            Some("eval=net-search 1x1 budget=8")
        );
    }

    #[test]
    fn blueprint_search_rows_keep_the_c8_method_and_eval_config() {
        assert_eq!(
            joined_method("blueprint-search-1000x1", None),
            "blueprint-search"
        );
        assert_eq!(
            eval_config("blueprint-search-1000x1", 5000.0).as_deref(),
            Some("eval=blueprint-search 1000x1 budget=5000")
        );
    }

    #[test]
    fn inferred_methods_preserve_multiword_roster_names() {
        assert_eq!(infer_method("ab-rollout-48"), "abstract-rollout");
        assert_eq!(infer_method("is-mcts-8x32"), "is-mcts");
        assert_eq!(infer_method("mccfr-256i-12k"), "mccfr");
        assert_eq!(infer_method("qlearn-1000e-4k"), "qlearn");
        assert_eq!(infer_method("history-history_x1"), "history-rnad");
        assert_eq!(infer_method("net-deepcfr_x3"), "deepcfr");
        assert_eq!(infer_method("net-teacher_x10"), "teacher");
        assert_eq!(
            infer_method("net-search-deepcfr_x3-1x1"),
            "deepcfr+net-search"
        );
        assert_eq!(
            infer_method("net-search-teacher_x10-1x1"),
            "teacher+net-search"
        );
        assert_eq!(infer_method("solve-net-deepcfr_x3-16x1"), "deepcfr+solve");
        assert_eq!(infer_method("solve-net-teacher_x10-16x1"), "teacher+solve");
    }

    #[test]
    fn roster_requirements_match_expected_method_families() {
        let reqs = roster_requirements();
        let c5 = reqs.iter().find(|req| req.id == "C5").unwrap();
        let c3 = reqs.iter().find(|req| req.id == "C3").unwrap();
        let c11 = reqs.iter().find(|req| req.id == "C11").unwrap();
        let c12 = reqs.iter().find(|req| req.id == "C12").unwrap();
        let c13 = reqs.iter().find(|req| req.id == "C13").unwrap();
        assert!(method_matches_requirement("is-mcts", c3));
        assert!(!method_matches_requirement("is", c3));
        assert!(method_matches_requirement("deepcfr+net-search", c5));
        assert!(method_matches_requirement("population-rnad+solve", c11));
        assert!(method_matches_requirement("teacher+solve", c12));
        assert!(method_matches_requirement("history-rnad+net-search", c13));
    }

    #[test]
    fn learned_solve_checkpoint_rows_get_distinct_method_and_eval_config() {
        assert_eq!(
            joined_method("solve-net-deepcfr_100-1000x1", Some("deepcfr")),
            "deepcfr+solve"
        );
        assert_eq!(
            joined_method("solve-net-deepcfr_100-1000x1", Some("population-net+solve")),
            "population-net+solve"
        );
        assert_eq!(
            eval_config("solve-rnad-rnad_3-1000x1", 5000.0).as_deref(),
            Some("eval=learned-solve 1000x1 budget=5000")
        );
    }

    #[test]
    fn raw_net_checkpoint_rows_keep_the_training_method() {
        assert_eq!(joined_method("net-deepcfr_100", Some("deepcfr")), "deepcfr");
        assert!(eval_config("net-deepcfr_100", 1.0).is_none());
    }

    #[test]
    fn distill_search_rows_are_separate_from_raw_distill() {
        assert_eq!(
            joined_method("net-search-distill_8-1x1", Some("distill")),
            "distill+net-search"
        );
        assert_eq!(compact_number(8.0), "8");
        assert_eq!(compact_number(8.5), "8p5");
    }

    #[test]
    fn best_checkpoint_alias_uses_run_best_bin() {
        assert_eq!(
            derived_best_checkpoint("/tmp/run/deepcfr_x3/ckpt_4.bin").as_deref(),
            Some("/tmp/run/deepcfr_x3/best.bin")
        );
    }

    #[test]
    fn teacher_base_family_groups_multiplier_runs() {
        let v: Value = serde_json::json!({
            "base": "/tmp/ld_bakeoff/rnad_x10/best.bin"
        });
        assert_eq!(teacher_base_family(&v), "rnad");
        assert_eq!(strip_scale_suffix("teacher_x3"), "teacher");
        assert_eq!(strip_scale_suffix("teacher_final"), "teacher_final");
    }

    #[test]
    fn rebel_winshare_target_defaults_legacy_and_reads_new_fields() {
        let old: Value = serde_json::json!({});
        assert_eq!(rebel_winshare_target(&old), "3p3d4f");

        let new: Value = serde_json::json!({
            "winshare_players": 5,
            "winshare_dice": 5,
            "winshare_faces": 6
        });
        assert_eq!(rebel_winshare_target(&new), "5p5d6f");
    }
}
