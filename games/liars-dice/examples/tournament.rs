//! N-seat Liar's Dice tournament harness.
//!
//! The primary score is field win-share: one hero agent is rotated through every
//! seat against a homogeneous field, so fair play is exactly `1 / players`.
//! The matrix is row hero vs column field. Wilson intervals and mean placement
//! are printed for each cell; a field-normalized Bradley-Terry Elo table gives a
//! compact ranking across the same results.
//!
//!     cargo run --release -p liars-dice --example tournament
//!     cargo run --release -p liars-dice --example tournament -- games=1000 rollouts=160
//!     cargo run --release -p liars-dice --example tournament -- rebel=runs/ld_rebel/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=belief,is-mcts,net net=runs/ld_deepcfr/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=belief,rollout,baseline-field
//!     cargo run --release -p liars-dice --example tournament -- nets=small:runs/ld_deepcfr/small.bin,big:runs/ld_deepcfr/big.bin
//!     cargo run --release -p liars-dice --example tournament -- resume=1 cells=net-small:rollout-48,net-big:rollout-48
//!     cargo run --release -p liars-dice --example tournament -- agents=belief,mccfr mccfr_iters=4096
//!     cargo run --release -p liars-dice --example tournament -- agents=belief,qlearn q_episodes=20000
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,blueprint-search solve_iters=4000 solve_restarts=2
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,net-search net=runs/ld_deepcfr/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,solve net=runs/ld_deepcfr/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,rnad rnad=runs/ld_rnad/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,ppo ppo=runs/ld_ppo/best.bin
//!     cargo run --release -p liars-dice --example tournament -- agents=rollout,history history=runs/ld_history/best.bin

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use game_core::stats::{fit_elo, wilson};
use game_core::{Agent, Game, RandomAgent, Rng, Turn, win_share};
use liars_dice::rebel::{PbsNet, RebelAgent};
use liars_dice::{
    AbstractedMccfrAgent, AbstractedQAgent, AbstractedRolloutAgent, ActionAbstractionConfig,
    BidConditioned, DeterminizedMctsAgent, DiceShareValue, HistoryNetAgent, LiarsDice, NetAgent,
    NetOnlineSolveAgent, NetTruncRollout, OnlineSolveAgent, OnlineSolveConfig, ProbConfig,
    ProbabilisticAgent,
};
use serde_json::Value;
use solvers::Rollout;

const DEFAULT_NET: &str = "runs/ld_deepcfr/best.bin";
const DEFAULT_RNAD: &str = "runs/ld_rnad/best.bin";
const DEFAULT_PPO: &str = "runs/ld_ppo/best.bin";
const DEFAULT_HISTORY: &str = "runs/ld_history/best.bin";
const DEFAULT_REBEL: &str = "runs/ld_rebel/best.bin";
const NET_SEARCH_CAND_CAP: u64 = 8;
const SEED_POLICY: &str = "name-stable-cell-v1";
const HERO_SEAT_POLICY: &str = "game-index-mod-players";
const BASELINE_FIELD_NAME: &str = "baseline-field";

struct Args {
    players: u8,
    dice: u8,
    faces: u8,
    games: u32,
    rollouts: u32,
    rollout_sweep: Vec<u32>,
    ab_rollout_sweep: Vec<u32>,
    mccfr_iters: u64,
    mccfr_seed: u64,
    mccfr_max_decision_plies: Option<u16>,
    q_episodes: u64,
    q_seed: u64,
    mcts_worlds: u32,
    mcts_sims: u32,
    net_search_rollouts: u32,
    net_search_plies: u32,
    solve_iters: u64,
    solve_max_iters: u64,
    solve_restarts: usize,
    solve_seed: u64,
    solve_flat_iters: Option<u64>,
    seed: u64,
    agents: Option<Vec<String>>,
    metrics: Option<String>,
    resume: bool,
    cells: Vec<CellSpec>,
    net: Option<String>,
    nets: Vec<CheckpointSpec>,
    rnad: Option<String>,
    rnads: Vec<CheckpointSpec>,
    ppo: Option<String>,
    ppos: Vec<CheckpointSpec>,
    history: Option<String>,
    histories: Vec<CheckpointSpec>,
    rebel: Option<String>,
    rebels: Vec<CheckpointSpec>,
    rebel_iters: usize,
    rebel_depth: u32,
}

#[derive(Clone, Debug)]
struct CheckpointSpec {
    label: String,
    path: String,
}

#[derive(Clone, Debug)]
struct CellSpec {
    hero: String,
    field: String,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Args {
            players: 5,
            dice: 5,
            faces: 6,
            games: 200,
            rollouts: 48,
            rollout_sweep: Vec::new(),
            ab_rollout_sweep: Vec::new(),
            mccfr_iters: 256,
            mccfr_seed: 0xC0F5_D1CE,
            mccfr_max_decision_plies: None,
            q_episodes: 1000,
            q_seed: 0xA11C_E5E5,
            mcts_worlds: 8,
            mcts_sims: 32,
            net_search_rollouts: 48,
            net_search_plies: 3,
            solve_iters: OnlineSolveConfig::default().iters,
            solve_max_iters: OnlineSolveConfig::default().max_iters,
            solve_restarts: OnlineSolveConfig::default().restarts,
            solve_seed: OnlineSolveConfig::default().seed,
            solve_flat_iters: OnlineSolveConfig::default().flat_iters,
            seed: 0x51A5_D1CE,
            agents: None,
            metrics: Some("runs/ld_tournament_metrics.jsonl".to_string()),
            resume: false,
            cells: Vec::new(),
            net: None,
            nets: Vec::new(),
            rnad: None,
            rnads: Vec::new(),
            ppo: None,
            ppos: Vec::new(),
            history: None,
            histories: Vec::new(),
            rebel: None,
            rebels: Vec::new(),
            rebel_iters: 96,
            rebel_depth: 2,
        };

        for raw in std::env::args().skip(1) {
            let Some((key, value)) = raw.split_once('=') else {
                return Err(format!("expected key=value argument, got '{raw}'"));
            };
            match key {
                "players" => args.players = parse_num(value, key)?,
                "dice" => args.dice = parse_num(value, key)?,
                "faces" => args.faces = parse_num(value, key)?,
                "games" => args.games = parse_num(value, key)?,
                "rollouts" => args.rollouts = parse_num(value, key)?,
                "rollout_sweep" | "rollouts_sweep" => {
                    args.rollout_sweep = parse_u32_list(value, key)?;
                }
                "ab_rollout_sweep" | "ab_rollouts_sweep" | "abstract_rollout_sweep" => {
                    args.ab_rollout_sweep = parse_u32_list(value, key)?;
                }
                "mccfr_iters" => args.mccfr_iters = parse_num(value, key)?,
                "mccfr_seed" => args.mccfr_seed = parse_u64(value, key)?,
                "mccfr_max_decision_plies" | "mccfr_depth" | "mccfr_max_plies" => {
                    args.mccfr_max_decision_plies = parse_optional_u16(value, key)?;
                }
                "q_episodes" => args.q_episodes = parse_num(value, key)?,
                "q_seed" => args.q_seed = parse_u64(value, key)?,
                "mcts_worlds" => args.mcts_worlds = parse_num(value, key)?,
                "mcts_sims" => args.mcts_sims = parse_num(value, key)?,
                "net_search_rollouts" | "search_rollouts" => {
                    args.net_search_rollouts = parse_num(value, key)?;
                }
                "net_search_plies" | "search_plies" => {
                    args.net_search_plies = parse_num(value, key)?;
                }
                "solve_iters" | "online_iters" => args.solve_iters = parse_num(value, key)?,
                "solve_max_iters" | "online_max_iters" => {
                    args.solve_max_iters = parse_num(value, key)?;
                }
                "solve_restarts" | "online_restarts" => {
                    args.solve_restarts = parse_num(value, key)?;
                }
                "solve_seed" | "online_seed" => args.solve_seed = parse_u64(value, key)?,
                "solve_flat_iters" | "online_flat_iters" => {
                    args.solve_flat_iters = parse_optional_u64(value, key)?;
                }
                "seed" => args.seed = parse_u64(value, key)?,
                "agents" => {
                    args.agents = Some(
                        value
                            .split(',')
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect(),
                    );
                }
                "metrics" => {
                    args.metrics = if value == "none" {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "resume" | "append" => args.resume = parse_bool(value, key)?,
                "cell" | "cells" => args.cells = parse_cell_specs(value)?,
                "net" => {
                    args.net = if value == "none" {
                        Some("none".to_string())
                    } else {
                        Some(value.to_string())
                    };
                }
                "nets" => {
                    args.nets = if value == "none" {
                        Vec::new()
                    } else {
                        parse_checkpoint_specs(value, key)?
                    };
                }
                "rnad" => {
                    args.rnad = if value == "none" {
                        Some("none".to_string())
                    } else {
                        Some(value.to_string())
                    };
                }
                "rnads" => {
                    args.rnads = if value == "none" {
                        Vec::new()
                    } else {
                        parse_checkpoint_specs(value, key)?
                    };
                }
                "ppo" => {
                    args.ppo = if value == "none" {
                        Some("none".to_string())
                    } else {
                        Some(value.to_string())
                    };
                }
                "ppos" => {
                    args.ppos = if value == "none" {
                        Vec::new()
                    } else {
                        parse_checkpoint_specs(value, key)?
                    };
                }
                "history" | "history_net" => {
                    args.history = if value == "none" {
                        Some("none".to_string())
                    } else {
                        Some(value.to_string())
                    };
                }
                "histories" | "history_nets" => {
                    args.histories = if value == "none" {
                        Vec::new()
                    } else {
                        parse_checkpoint_specs(value, key)?
                    };
                }
                "rebel" => {
                    args.rebel = if value == "none" {
                        Some("none".to_string())
                    } else {
                        Some(value.to_string())
                    };
                }
                "rebels" => {
                    args.rebels = if value == "none" {
                        Vec::new()
                    } else {
                        parse_checkpoint_specs(value, key)?
                    };
                }
                "rebel_iters" => args.rebel_iters = parse_num(value, key)?,
                "rebel_depth" => args.rebel_depth = parse_num(value, key)?,
                _ => return Err(format!("unknown argument '{key}'")),
            }
        }
        if args.players < 2 {
            return Err("players must be at least 2".to_string());
        }
        if args.faces < 2 {
            return Err("faces must be at least 2".to_string());
        }
        if args.games == 0 {
            return Err("games must be positive".to_string());
        }
        if args.solve_iters == 0 {
            return Err("solve_iters must be positive".to_string());
        }
        if args.solve_max_iters == 0 {
            return Err("solve_max_iters must be positive".to_string());
        }
        if args.solve_restarts == 0 {
            return Err("solve_restarts must be positive".to_string());
        }
        if args.net_search_rollouts == 0 {
            return Err("net_search_rollouts must be positive".to_string());
        }
        if args.rollout_sweep.contains(&0) {
            return Err("rollout_sweep entries must be positive".to_string());
        }
        if args.ab_rollout_sweep.contains(&0) {
            return Err("ab_rollout_sweep entries must be positive".to_string());
        }
        Ok(args)
    }

    fn solve_config(&self) -> OnlineSolveConfig {
        OnlineSolveConfig {
            iters: self.solve_iters,
            max_iters: self.solve_max_iters,
            restarts: self.solve_restarts,
            seed: self.solve_seed,
            flat_iters: self.solve_flat_iters,
        }
    }
}

struct Entry {
    name: String,
    agent: Box<dyn Agent<LiarsDice>>,
    field_components: Option<Vec<FieldComponent>>,
    budget_per_move: u64,
    setup: SetupStats,
}

impl Entry {
    fn new(
        name: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        budget_per_move: u64,
    ) -> Self {
        Self {
            name: name.into(),
            agent,
            field_components: None,
            budget_per_move,
            setup: SetupStats::default(),
        }
    }

    fn with_setup(
        name: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        budget_per_move: u64,
        setup: SetupStats,
    ) -> Self {
        Self {
            name: name.into(),
            agent,
            field_components: None,
            budget_per_move,
            setup,
        }
    }

    fn baseline_field(
        name: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        field_components: Vec<FieldComponent>,
        budget_per_move: u64,
        setup: SetupStats,
    ) -> Self {
        Self {
            name: name.into(),
            agent,
            field_components: Some(field_components),
            budget_per_move,
            setup,
        }
    }

    fn field_agent(
        &self,
        player: usize,
        hero_seat: usize,
        game_idx: u32,
        players: usize,
    ) -> &dyn Agent<LiarsDice> {
        if let Some(components) = &self.field_components {
            let idx =
                scheduled_field_component(player, hero_seat, game_idx, players, components.len());
            &*components[idx].agent
        } else {
            &*self.agent
        }
    }
}

struct FieldComponent {
    name: String,
    agent: Box<dyn Agent<LiarsDice>>,
}

struct BaselineFieldHero {
    components: Vec<FieldComponent>,
}

impl Agent<LiarsDice> for BaselineFieldHero {
    fn act(
        &self,
        game: &LiarsDice,
        state: &liars_dice::LdState,
        player: usize,
        rng: &mut Rng,
    ) -> usize {
        let idx = player % self.components.len();
        self.components[idx].agent.act(game, state, player, rng)
    }
}

#[derive(Clone, Debug)]
struct SetupStats {
    kind: &'static str,
    units: u64,
    rows: usize,
    wall_s: f64,
    source: Option<String>,
}

impl Default for SetupStats {
    fn default() -> Self {
        Self {
            kind: "none",
            units: 0,
            rows: 0,
            wall_s: 0.0,
            source: None,
        }
    }
}

impl SetupStats {
    fn measured(kind: &'static str, units: u64, rows: usize, wall_s: f64) -> Self {
        Self {
            kind,
            units,
            rows,
            wall_s,
            source: None,
        }
    }

    fn source(kind: &'static str, source: impl Into<String>, wall_s: f64) -> Self {
        Self {
            kind,
            units: 0,
            rows: 0,
            wall_s,
            source: Some(source.into()),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Cell {
    share_sum: f64,
    placement_sum: f64,
    games: u32,
    decisions: u64,
    wall_s: f64,
}

impl Cell {
    fn has_data(self) -> bool {
        self.games > 0
    }

    fn share(self) -> f64 {
        if self.has_data() {
            self.share_sum / f64::from(self.games)
        } else {
            0.0
        }
    }

    fn mean_placement(self) -> f64 {
        if self.has_data() {
            self.placement_sum / f64::from(self.games)
        } else {
            0.0
        }
    }
}

#[derive(Default)]
struct ExistingMetrics {
    agents: HashSet<String>,
    cells: HashMap<(String, String), Cell>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let game = LiarsDice::new(args.players, args.dice, args.faces);
    let roster = build_roster(&args)?;
    if roster.len() < 2 {
        return Err("tournament needs at least two agents".to_string());
    }
    validate_cell_specs(&args.cells, &roster)?;

    let mut existing = ExistingMetrics::default();
    if args.resume {
        if let Some(path) = args.metrics.as_deref() {
            existing = read_existing_metrics(path, &args)?;
        } else {
            return Err("resume=1 requires metrics=<path>, not metrics=none".to_string());
        }
    }
    let mut metrics = open_metrics(args.metrics.as_deref(), args.resume)?;
    println!(
        "Liar's Dice tournament: {}p{}d{}f, {} games/cell, seed={:#x}",
        args.players, args.dice, args.faces, args.games, args.seed
    );
    println!("seed policy: {SEED_POLICY}; hero seats: {HERO_SEAT_POLICY}");
    println!(
        "agents: {}",
        roster
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(path) = args.metrics.as_deref() {
        println!(
            "metrics: {path}{}",
            if args.resume { " (append/resume)" } else { "" }
        );
    }
    if !args.cells.is_empty() {
        println!(
            "cell filter: {}",
            args.cells
                .iter()
                .map(|c| format!("{}:{}", c.hero, c.field))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    write_config_metric(&mut metrics, &game, &args, &roster)
        .map_err(|e| format!("failed to write tournament config metrics: {e}"))?;
    for entry in &roster {
        if !existing.agents.contains(&entry.name) {
            write_agent_metric(&mut metrics, &game, &args, entry)
                .map_err(|e| format!("failed to write agent metrics: {e}"))?;
        }
    }
    println!();

    let n = roster.len();
    let mut cells = vec![vec![Cell::default(); n]; n];
    for (i, hero) in roster.iter().enumerate() {
        for (j, field) in roster.iter().enumerate() {
            if let Some(cell) = existing.cells.get(&(hero.name.clone(), field.name.clone())) {
                cells[i][j] = *cell;
            }
        }
    }
    for hero in 0..n {
        for field in 0..n {
            if !selected_cell(&args, &roster, hero, field) {
                continue;
            }
            if args.resume && cells[hero][field].games >= args.games {
                println!(
                    "skip existing cell {} vs {} ({} games)",
                    roster[hero].name, roster[field].name, cells[hero][field].games
                );
                continue;
            }
            let seed = cell_seed(args.seed, &roster[hero].name, &roster[field].name);
            let cell = evaluate_cell(&game, &roster[hero], &roster[field], args.games, seed);
            write_metric(
                &mut metrics,
                &game,
                &args,
                &roster[hero],
                &roster[field],
                cell,
            )
            .map_err(|e| format!("failed to write metrics: {e}"))?;
            cells[hero][field] = cell;
        }
    }

    print_matrix(&roster, &cells);
    print_details(&roster, &cells);
    if complete_matrix(&cells) {
        print_elo(args.players as usize, &roster, &cells);
        write_elo_metrics(
            &mut metrics,
            args.players as usize,
            &game,
            &args,
            &roster,
            &cells,
        )
        .map_err(|e| format!("failed to write Elo metrics: {e}"))?;
    } else {
        println!("field-normalized Elo skipped: matrix is incomplete");
    }
    Ok(())
}

fn build_roster(args: &Args) -> Result<Vec<Entry>, String> {
    let mut names = args.agents.clone().unwrap_or_else(|| {
        let mut default = vec![
            "random".to_string(),
            "honest-bayes".to_string(),
            "aggressive-bluffer".to_string(),
            "conservative-caller".to_string(),
            "belief".to_string(),
            "rollout".to_string(),
            "abstract-rollout".to_string(),
            "is-mcts".to_string(),
            BASELINE_FIELD_NAME.to_string(),
        ];
        if args.rebel.as_deref().is_some_and(|path| path != "none") {
            default.push("rebel".to_string());
        }
        if !args.rebels.is_empty() {
            default.push("rebels".to_string());
        }
        if args.net.as_deref().is_some_and(|path| path != "none") {
            default.push("net".to_string());
        }
        if !args.nets.is_empty() {
            default.push("nets".to_string());
            default.push("solves".to_string());
        }
        if args.rnad.as_deref().is_some_and(|path| path != "none") {
            default.push("rnad".to_string());
        }
        if !args.rnads.is_empty() {
            default.push("rnads".to_string());
            if !default.iter().any(|name| name == "solves") {
                default.push("solves".to_string());
            }
        }
        if args.ppo.as_deref().is_some_and(|path| path != "none") {
            default.push("ppo".to_string());
        }
        if !args.ppos.is_empty() {
            default.push("ppos".to_string());
            if !default.iter().any(|name| name == "solves") {
                default.push("solves".to_string());
            }
        }
        if args.history.as_deref().is_some_and(|path| path != "none") {
            default.push("history".to_string());
        }
        if !args.histories.is_empty() {
            default.push("histories".to_string());
        }
        default
    });
    names.dedup();

    let explicit_agents = args.agents.is_some();
    let mut entries = Vec::new();
    for name in names {
        match name.as_str() {
            "random" | "random-legal" => {
                entries.push(Entry::new("random", Box::new(RandomAgent), 0));
            }
            "honest" | "honest-bayes" => entries.push(Entry::new(
                "honest-bayes",
                Box::new(ProbabilisticAgent::new(ProbConfig::honest_bayes())),
                0,
            )),
            "aggressive" | "aggressive-bluffer" => entries.push(Entry::new(
                "aggressive-bluffer",
                Box::new(ProbabilisticAgent::new(ProbConfig::aggressive_bluffer())),
                0,
            )),
            "conservative" | "conservative-caller" => entries.push(Entry::new(
                "conservative-caller",
                Box::new(ProbabilisticAgent::new(ProbConfig::conservative_caller())),
                0,
            )),
            "belief" | "prob" | "probabilistic" => entries.push(Entry::new(
                "belief",
                Box::new(ProbabilisticAgent::default_agent()),
                0,
            )),
            "rollout" => entries.push(Entry::new(
                format!("rollout-{}", args.rollouts),
                Box::new(Rollout::new(
                    args.rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                )),
                u64::from(args.rollouts),
            )),
            "rollout-sweep" | "rollouts-sweep" => {
                if args.rollout_sweep.is_empty() {
                    return Err(
                        "agents includes rollout-sweep but no rollout_sweep=... was supplied"
                            .to_string(),
                    );
                }
                for &rollouts in &args.rollout_sweep {
                    entries.push(build_rollout_entry(rollouts));
                }
            }
            "abstract-rollout" | "ab-rollout" => {
                let abstraction = ActionAbstractionConfig::default();
                let max_candidates = abstraction.max_candidates as u64;
                entries.push(Entry::new(
                    format!("ab-rollout-{}", args.rollouts),
                    Box::new(AbstractedRolloutAgent::with_config(
                        args.rollouts,
                        ProbabilisticAgent::default_agent(),
                        BidConditioned::default(),
                        abstraction,
                    )),
                    u64::from(args.rollouts) * max_candidates,
                ));
            }
            "abstract-rollout-sweep" | "ab-rollout-sweep" | "ab-rollouts-sweep" => {
                if args.ab_rollout_sweep.is_empty() {
                    return Err(
                        "agents includes ab-rollout-sweep but no ab_rollout_sweep=... was supplied"
                            .to_string(),
                    );
                }
                for &rollouts in &args.ab_rollout_sweep {
                    entries.push(build_abstract_rollout_entry(rollouts));
                }
            }
            "is-mcts" | "det-mcts" => {
                let abstraction = ActionAbstractionConfig::default();
                let max_candidates = abstraction.max_candidates as u64;
                entries.push(Entry::new(
                    format!("is-mcts-{}x{}", args.mcts_worlds, args.mcts_sims),
                    Box::new(DeterminizedMctsAgent::with_config(
                        args.mcts_worlds,
                        args.mcts_sims,
                        BidConditioned::default(),
                        abstraction,
                    )),
                    u64::from(args.mcts_worlds) * u64::from(args.mcts_sims) * max_candidates,
                ));
            }
            "baseline" | "baseline-field" | "baseline-mix" => {
                entries.push(build_baseline_field(args)?);
            }
            "mccfr" | "abstract-mccfr" => {
                let train_game = LiarsDice::new(args.players, args.dice, args.faces);
                let start = Instant::now();
                let agent = AbstractedMccfrAgent::train_with_config_and_max_decision_plies(
                    train_game,
                    args.mccfr_iters,
                    args.mccfr_seed,
                    ActionAbstractionConfig::default(),
                    args.mccfr_max_decision_plies,
                );
                let wall_s = start.elapsed().as_secs_f64();
                let infosets = agent.num_infosets();
                let cap_label = args
                    .mccfr_max_decision_plies
                    .map(|d| format!("-d{d}"))
                    .unwrap_or_default();
                let mut setup =
                    SetupStats::measured("mccfr_train", args.mccfr_iters, infosets, wall_s);
                if let Some(depth) = args.mccfr_max_decision_plies {
                    setup.source = Some(format!("max_decision_plies={depth}"));
                }
                entries.push(Entry::with_setup(
                    format!(
                        "mccfr{cap_label}-{}i-{}k",
                        args.mccfr_iters,
                        infosets / 1000
                    ),
                    Box::new(agent),
                    args.mccfr_iters,
                    setup,
                ));
            }
            "qlearn" | "q-learning" | "q" => {
                let train_game = LiarsDice::new(args.players, args.dice, args.faces);
                let start = Instant::now();
                let agent = AbstractedQAgent::train(train_game, args.q_episodes, args.q_seed);
                let wall_s = start.elapsed().as_secs_f64();
                let rows = agent.table_size();
                entries.push(Entry::with_setup(
                    format!("qlearn-{}e-{}k", args.q_episodes, rows / 1000),
                    Box::new(agent),
                    args.q_episodes,
                    SetupStats::measured("qlearn_train", args.q_episodes, rows, wall_s),
                ));
            }
            "online-solve" | "dice-share-solve" => {
                entries.push(build_blueprint_search_entry(args, "online-solve"));
            }
            "blueprint-search" | "pluribus" | "pluribus-search" => {
                entries.push(build_blueprint_search_entry(args, "blueprint-search"));
            }
            "net" | "deepcfr-net" | "distill-net" => {
                if let Some(entry) = build_net(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "nets" | "deepcfr-nets" | "distill-nets" => {
                entries.extend(build_nets(args, explicit_agents)?);
            }
            "rnad" | "rnad-net" | "neurd" => {
                if let Some(entry) = build_rnad(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "rnads" | "rnad-nets" | "neurd-nets" => {
                entries.extend(build_rnads(args, explicit_agents)?);
            }
            "ppo" | "ppo-net" => {
                if let Some(entry) = build_ppo(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "ppos" | "ppo-nets" => {
                entries.extend(build_ppos(args, explicit_agents)?);
            }
            "history" | "history-net" | "history-rnad" | "transformer" => {
                if let Some(entry) = build_history(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "histories" | "history-nets" | "history-rnads" | "transformers" => {
                entries.extend(build_histories(args, explicit_agents)?);
            }
            "net-search" | "trunc-net" | "net-trunc-rollout" => {
                if let Some(entry) = build_net_search(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "net-searches" | "trunc-nets" | "net-trunc-rollouts" => {
                entries.extend(build_net_searches(args, explicit_agents)?);
            }
            "solve" | "net-solve" | "online-net-solve" => {
                if let Some(entry) = build_solve(args, explicit_agents, name.as_str())? {
                    entries.push(entry);
                }
            }
            "solves" | "net-solves" | "online-net-solves" => {
                entries.extend(build_solves(args, explicit_agents)?);
            }
            "rebel" => {
                if let Some(entry) = build_rebel(args, explicit_agents)? {
                    entries.push(entry);
                }
            }
            "rebels" => {
                entries.extend(build_rebels(args, explicit_agents)?);
            }
            other => {
                return Err(format!(
                    "unknown agent '{other}' (random,honest-bayes,aggressive-bluffer,\
	                    conservative-caller,belief,rollout,abstract-rollout,is-mcts,mccfr,qlearn,\
	                     rollout-sweep,ab-rollout-sweep,baseline-field,online-solve,blueprint-search,net,nets,rnad,rnads,ppo,ppos,history,histories,\
	                     net-search,net-searches,solve,solves,rebel,rebels)"
                ));
            }
        }
    }
    Ok(entries)
}

fn build_rollout_entry(rollouts: u32) -> Entry {
    Entry::new(
        format!("rollout-{rollouts}"),
        Box::new(Rollout::new(
            rollouts,
            ProbabilisticAgent::default_agent(),
            BidConditioned::default(),
        )),
        u64::from(rollouts),
    )
}

fn build_abstract_rollout_entry(rollouts: u32) -> Entry {
    let abstraction = ActionAbstractionConfig::default();
    let max_candidates = abstraction.max_candidates as u64;
    Entry::new(
        format!("ab-rollout-{rollouts}"),
        Box::new(AbstractedRolloutAgent::with_config(
            rollouts,
            ProbabilisticAgent::default_agent(),
            BidConditioned::default(),
            abstraction,
        )),
        u64::from(rollouts) * max_candidates,
    )
}

fn build_baseline_field(args: &Args) -> Result<Entry, String> {
    let start = Instant::now();
    let hero_components = baseline_components(args)?;
    let component_names = hero_components
        .iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let component_count = hero_components.len() as u64;
    let field_components = baseline_components(args)?;
    let budget_per_move = field_components
        .iter()
        .map(|c| component_budget(&c.name, args))
        .max()
        .unwrap_or(0);
    Ok(Entry::baseline_field(
        BASELINE_FIELD_NAME,
        Box::new(BaselineFieldHero {
            components: hero_components,
        }),
        field_components,
        budget_per_move,
        SetupStats {
            kind: "baseline_field",
            units: component_count,
            rows: 0,
            wall_s: start.elapsed().as_secs_f64(),
            source: Some(component_names),
        },
    ))
}

fn baseline_components(args: &Args) -> Result<Vec<FieldComponent>, String> {
    let mut components = vec![
        FieldComponent {
            name: "random".to_string(),
            agent: Box::new(RandomAgent),
        },
        FieldComponent {
            name: "honest-bayes".to_string(),
            agent: Box::new(ProbabilisticAgent::new(ProbConfig::honest_bayes())),
        },
        FieldComponent {
            name: "aggressive-bluffer".to_string(),
            agent: Box::new(ProbabilisticAgent::new(ProbConfig::aggressive_bluffer())),
        },
        FieldComponent {
            name: "conservative-caller".to_string(),
            agent: Box::new(ProbabilisticAgent::new(ProbConfig::conservative_caller())),
        },
        FieldComponent {
            name: "belief".to_string(),
            agent: Box::new(ProbabilisticAgent::default_agent()),
        },
        FieldComponent {
            name: format!("rollout-{}", args.rollouts),
            agent: Box::new(Rollout::new(
                args.rollouts,
                ProbabilisticAgent::default_agent(),
                BidConditioned::default(),
            )),
        },
    ];

    if let Some(path) = baseline_rebel_path(args) {
        let net = PbsNet::load(Path::new(path))
            .map_err(|e| format!("failed to load baseline-field rebel checkpoint '{path}': {e}"))?;
        components.push(FieldComponent {
            name: format!("rebel-{}x{}", args.rebel_iters, args.rebel_depth),
            agent: Box::new(RebelAgent::with_config(
                net,
                args.rebel_iters,
                args.rebel_depth,
            )),
        });
    }
    Ok(components)
}

fn baseline_rebel_path(args: &Args) -> Option<&str> {
    match args.rebel.as_deref() {
        Some("none") => None,
        Some(path) => Some(path),
        None => None,
    }
}

fn component_budget(name: &str, args: &Args) -> u64 {
    if name.starts_with("rollout-") {
        u64::from(args.rollouts)
    } else if name.starts_with("rebel-") {
        args.rebel_iters as u64
    } else {
        0
    }
}

fn build_blueprint_search_entry(args: &Args, name: &str) -> Entry {
    let cfg = args.solve_config();
    let total_dice = u32::from(args.players) * u32::from(args.dice);
    let (iters, restarts) = cfg.budget_for_total_dice(total_dice);
    Entry::new(
        format!("{name}-{}x{}", iters, restarts),
        Box::new(OnlineSolveAgent::with_config(|| DiceShareValue, cfg)),
        iters * restarts as u64 * u64::from(args.players),
    )
}

fn build_net(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.net.as_deref() == Some("none") {
        if required {
            return Err("agents includes net but net=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.net.as_deref().unwrap_or(DEFAULT_NET);
    let required = required || args.net.is_some();
    build_net_path(checkpoint_name(label, path), path, required)
}

fn build_nets(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    if args.nets.is_empty() {
        if required {
            return Err("agents includes nets but no nets=label:path,... was supplied".to_string());
        }
        return Ok(Vec::new());
    }

    let mut entries = Vec::with_capacity(args.nets.len());
    for spec in &args.nets {
        if let Some(entry) = build_net_path(format!("net-{}", spec.label), &spec.path, true)? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_net_path(name: String, path: &str, required: bool) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("net checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping net: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let start = Instant::now();
    let agent =
        NetAgent::load(Path::new(path)).map_err(|e| format!("failed to load net '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        name,
        Box::new(agent),
        1,
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn build_rnad(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.rnad.as_deref() == Some("none") {
        if required {
            return Err("agents includes rnad but rnad=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.rnad.as_deref().unwrap_or(DEFAULT_RNAD);
    let required = required || args.rnad.is_some();
    build_named_net_path(checkpoint_name(label, path), path, required, "rnad")
}

fn build_rnads(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    if args.rnads.is_empty() {
        if required {
            return Err(
                "agents includes rnads but no rnads=label:path,... was supplied".to_string(),
            );
        }
        return Ok(Vec::new());
    }

    let mut entries = Vec::with_capacity(args.rnads.len());
    for spec in &args.rnads {
        if let Some(entry) =
            build_named_net_path(format!("rnad-{}", spec.label), &spec.path, true, "rnad")?
        {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_ppo(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.ppo.as_deref() == Some("none") {
        if required {
            return Err("agents includes ppo but ppo=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.ppo.as_deref().unwrap_or(DEFAULT_PPO);
    let required = required || args.ppo.is_some();
    build_named_net_path(checkpoint_name(label, path), path, required, "ppo")
}

fn build_ppos(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    if args.ppos.is_empty() {
        if required {
            return Err("agents includes ppos but no ppos=label:path,... was supplied".to_string());
        }
        return Ok(Vec::new());
    }

    let mut entries = Vec::with_capacity(args.ppos.len());
    for spec in &args.ppos {
        if let Some(entry) =
            build_named_net_path(format!("ppo-{}", spec.label), &spec.path, true, "ppo")?
        {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_history(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.history.as_deref() == Some("none") {
        if required {
            return Err("agents includes history but history=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.history.as_deref().unwrap_or(DEFAULT_HISTORY);
    let required = required || args.history.is_some();
    build_history_path(checkpoint_name(label, path), path, required, "history")
}

fn build_histories(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    if args.histories.is_empty() {
        if required {
            return Err(
                "agents includes histories but no histories=label:path,... was supplied"
                    .to_string(),
            );
        }
        return Ok(Vec::new());
    }

    let mut entries = Vec::with_capacity(args.histories.len());
    for spec in &args.histories {
        if let Some(entry) = build_history_path(
            format!("history-{}", spec.label),
            &spec.path,
            true,
            "history",
        )? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_history_path(
    name: String,
    path: &str,
    required: bool,
    kind: &str,
) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("{kind} checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping {kind}: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let start = Instant::now();
    let agent = HistoryNetAgent::load(Path::new(path))
        .map_err(|e| format!("failed to load {kind} checkpoint '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        name,
        Box::new(agent),
        1,
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn build_named_net_path(
    name: String,
    path: &str,
    required: bool,
    kind: &str,
) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("{kind} checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping {kind}: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let start = Instant::now();
    let agent = NetAgent::load(Path::new(path))
        .map_err(|e| format!("failed to load {kind} checkpoint '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        name,
        Box::new(agent),
        1,
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn build_net_search(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.net.as_deref() == Some("none") {
        if required {
            return Err("agents includes net-search but net=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.net.as_deref().unwrap_or(DEFAULT_NET);
    let required = required || args.net.is_some();
    build_net_search_path(
        checkpoint_name(label, path),
        path,
        required,
        args.net_search_rollouts,
        args.net_search_plies,
    )
}

fn build_net_searches(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for spec in &args.nets {
        if let Some(entry) = build_net_search_path(
            format!("net-search-{}", spec.label),
            &spec.path,
            true,
            args.net_search_rollouts,
            args.net_search_plies,
        )? {
            entries.push(entry);
        }
    }
    for spec in &args.rnads {
        if let Some(entry) = build_net_search_path(
            format!("net-search-rnad-{}", spec.label),
            &spec.path,
            true,
            args.net_search_rollouts,
            args.net_search_plies,
        )? {
            entries.push(entry);
        }
    }
    for spec in &args.ppos {
        if let Some(entry) = build_net_search_path(
            format!("net-search-ppo-{}", spec.label),
            &spec.path,
            true,
            args.net_search_rollouts,
            args.net_search_plies,
        )? {
            entries.push(entry);
        }
    }
    if entries.is_empty() && required {
        return Err(
            "agents includes net-searches but no standard nets=, rnads=, or ppos= checkpoints were supplied"
                .to_string(),
        );
    }
    Ok(entries)
}

fn build_net_search_path(
    name: String,
    path: &str,
    required: bool,
    rollouts: u32,
    plies: u32,
) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("net-search checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping net-search: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let start = Instant::now();
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read net-search checkpoint '{path}': {e}"))?;
    let agent = NetTruncRollout::from_bytes(&bytes, rollouts, plies)
        .map_err(|e| format!("failed to load net-search checkpoint '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        format!("{name}-{}x{}", rollouts, plies),
        Box::new(agent),
        u64::from(rollouts) * u64::from(plies.max(1)) * NET_SEARCH_CAND_CAP,
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn build_solve(args: &Args, required: bool, label: &str) -> Result<Option<Entry>, String> {
    if args.net.as_deref() == Some("none") {
        if required {
            return Err("agents includes solve but net=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.net.as_deref().unwrap_or(DEFAULT_NET);
    let required = required || args.net.is_some();
    build_solve_path(checkpoint_name(label, path), args, path, required)
}

fn build_solves(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for spec in &args.nets {
        if let Some(entry) =
            build_solve_path(format!("solve-net-{}", spec.label), args, &spec.path, true)?
        {
            entries.push(entry);
        }
    }
    for spec in &args.rnads {
        if let Some(entry) =
            build_solve_path(format!("solve-rnad-{}", spec.label), args, &spec.path, true)?
        {
            entries.push(entry);
        }
    }
    for spec in &args.ppos {
        if let Some(entry) =
            build_solve_path(format!("solve-ppo-{}", spec.label), args, &spec.path, true)?
        {
            entries.push(entry);
        }
    }
    if entries.is_empty() && required {
        return Err(
            "agents includes solves but no standard nets=, rnads=, or ppos= checkpoints were supplied"
                .to_string(),
        );
    }
    Ok(entries)
}

fn build_solve_path(
    name: String,
    args: &Args,
    path: &str,
    required: bool,
) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("solve checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping solve: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let cfg = args.solve_config();
    let total_dice = u32::from(args.players) * u32::from(args.dice);
    let (iters, restarts) = cfg.budget_for_total_dice(total_dice);
    let start = Instant::now();
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read solve checkpoint '{path}': {e}"))?;
    let agent = NetOnlineSolveAgent::from_bytes_with_config(&bytes, cfg)
        .map_err(|e| format!("failed to load solve checkpoint '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        format!("{name}-{iters}x{restarts}"),
        Box::new(agent),
        iters * restarts as u64 * u64::from(args.players),
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn build_rebel(args: &Args, required: bool) -> Result<Option<Entry>, String> {
    if args.rebel.as_deref() == Some("none") {
        if required {
            return Err("agents includes rebel but rebel=none was supplied".to_string());
        }
        return Ok(None);
    }
    let path = args.rebel.as_deref().unwrap_or(DEFAULT_REBEL);
    let required = required || args.rebel.is_some();
    build_rebel_path(
        format!("rebel-{}x{}", args.rebel_iters, args.rebel_depth),
        args,
        path,
        required,
    )
}

fn build_rebels(args: &Args, required: bool) -> Result<Vec<Entry>, String> {
    if args.rebels.is_empty() {
        if required {
            return Err(
                "agents includes rebels but no rebels=label:path,... was supplied".to_string(),
            );
        }
        return Ok(Vec::new());
    }

    let mut entries = Vec::with_capacity(args.rebels.len());
    for spec in &args.rebels {
        if let Some(entry) = build_rebel_path(
            format!(
                "rebel-{}-{}x{}",
                spec.label, args.rebel_iters, args.rebel_depth
            ),
            args,
            &spec.path,
            true,
        )? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_rebel_path(
    name: String,
    args: &Args,
    path: &str,
    required: bool,
) -> Result<Option<Entry>, String> {
    if !Path::new(path).exists() {
        if required {
            return Err(format!("rebel checkpoint '{path}' does not exist"));
        }
        eprintln!("skipping rebel: default checkpoint '{path}' not found");
        return Ok(None);
    }
    let start = Instant::now();
    let net = PbsNet::load(Path::new(path))
        .map_err(|e| format!("failed to load rebel checkpoint '{path}': {e}"))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Some(Entry::with_setup(
        name,
        Box::new(RebelAgent::with_config(
            net,
            args.rebel_iters,
            args.rebel_depth,
        )),
        args.rebel_iters as u64,
        SetupStats::source("checkpoint_load", path, wall_s),
    )))
}

fn checkpoint_name(label: &str, path: &str) -> String {
    format!("{label}-{}", checkpoint_label(path))
}

fn evaluate_cell(game: &LiarsDice, hero: &Entry, field: &Entry, games: u32, seed: u64) -> Cell {
    let mut rng = Rng::new(seed);
    let start = Instant::now();
    let mut cell = Cell {
        games,
        ..Cell::default()
    };
    for g in 0..games {
        let hero_seat = (g as usize) % game.num_players();
        let agents: Vec<&dyn Agent<LiarsDice>> = (0..game.num_players())
            .map(|p| {
                if p == hero_seat {
                    &*hero.agent
                } else {
                    field.field_agent(p, hero_seat, g, game.num_players())
                }
            })
            .collect();
        let (terminal, decisions) = play_counted(game, &agents, &mut rng);
        cell.share_sum += win_share(game, &terminal, hero_seat);
        cell.placement_sum += placement(game, &terminal, hero_seat);
        cell.decisions += decisions;
    }
    cell.wall_s = start.elapsed().as_secs_f64();
    cell
}

fn scheduled_field_component(
    player: usize,
    hero_seat: usize,
    game_idx: u32,
    players: usize,
    components: usize,
) -> usize {
    debug_assert!(components > 0);
    let field_rank = if player < hero_seat {
        player
    } else {
        player.saturating_sub(1)
    };
    (game_idx as usize * players.saturating_sub(1) + field_rank) % components
}

fn play_counted(
    game: &LiarsDice,
    agents: &[&dyn Agent<LiarsDice>],
    rng: &mut Rng,
) -> (liars_dice::LdState, u64) {
    let mut state = game.initial_state();
    let mut decisions = 0;
    while !game.is_terminal(&state) {
        match game.turn(&state) {
            Turn::Chance => {
                let action = game.sample_chance_action(&state, rng);
                game.apply(&mut state, action);
            }
            Turn::Player(p) => {
                decisions += 1;
                let i = agents[p].act(game, &state, p, rng);
                let action = game.action_at(&state, i);
                game.apply(&mut state, action);
            }
        }
    }
    (state, decisions)
}

fn placement(game: &LiarsDice, terminal: &liars_dice::LdState, player: usize) -> f64 {
    let mut returns: Vec<(usize, f64)> = (0..game.num_players())
        .map(|p| (p, game.returns(terminal, p)))
        .collect();
    returns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut i = 0;
    while i < returns.len() {
        let mut j = i + 1;
        while j < returns.len() && returns[j].1 == returns[i].1 {
            j += 1;
        }
        if returns[i..j].iter().any(|&(p, _)| p == player) {
            let first_rank = i + 1;
            let last_rank = j;
            return (first_rank + last_rank) as f64 / 2.0;
        }
        i = j;
    }
    unreachable!("player must appear in placement table");
}

fn print_matrix(roster: &[Entry], cells: &[Vec<Cell>]) {
    let width = roster
        .iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(8)
        .max(8);
    println!("win-share matrix (row hero vs column field):");
    print!("{:>width$}", "", width = width);
    for entry in roster {
        print!(" {:>8}", short_name(&entry.name));
    }
    println!();
    for (i, entry) in roster.iter().enumerate() {
        print!("{:>width$}", entry.name, width = width);
        for cell in &cells[i] {
            if cell.has_data() {
                print!(" {:>8.3}", cell.share());
            } else {
                print!(" {:>8}", ".");
            }
        }
        println!();
    }
    println!();
}

fn print_details(roster: &[Entry], cells: &[Vec<Cell>]) {
    println!("cell details:");
    println!(
        "  {:<22} {:<22} {:>7} {:>15} {:>8} {:>8} {:>10}",
        "hero", "field", "share", "95% CI", "place", "sec", "decisions"
    );
    println!("  {}", "-".repeat(105));
    for (i, hero) in roster.iter().enumerate() {
        for (j, field) in roster.iter().enumerate() {
            let cell = cells[i][j];
            if !cell.has_data() {
                continue;
            }
            let (lo, hi) = wilson(cell.share_sum, f64::from(cell.games), 1.96);
            println!(
                "  {:<22} {:<22} {:>7.3} [{:>.3},{:>.3}] {:>8.2} {:>8.2} {:>10}",
                hero.name,
                field.name,
                cell.share(),
                lo,
                hi,
                cell.mean_placement(),
                cell.wall_s,
                cell.decisions
            );
        }
    }
    println!();
}

fn print_elo(players: usize, roster: &[Entry], cells: &[Vec<Cell>]) {
    let rows = elo_rows(players, roster, cells);
    println!("field-normalized Elo (mean anchored at 0):");
    for row in rows {
        println!(
            "  {:>2}. {:<22} elo {:+7.0}   budget/move {:>5}   setup {:>6.2}s   row wall {:>7.1}s",
            row.rank,
            roster[row.agent_idx].name,
            row.elo,
            roster[row.agent_idx].budget_per_move,
            roster[row.agent_idx].setup.wall_s,
            row.row_wall_s
        );
    }
}

#[derive(Clone, Copy)]
struct EloRow {
    rank: usize,
    agent_idx: usize,
    elo: f64,
    row_wall_s: f64,
}

fn elo_rows(players: usize, roster: &[Entry], cells: &[Vec<Cell>]) -> Vec<EloRow> {
    let n = roster.len();
    let mut records = vec![vec![(0u64, 0u64, 0u64); n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let a = field_to_pair_score(cells[i][j].share(), players);
            let b = field_to_pair_score(cells[j][i].share(), players);
            let score = (a + (1.0 - b)) / 2.0;
            let games = cells[i][j].games.min(cells[j][i].games) as u64;
            let wins = (score * games as f64).round().clamp(0.0, games as f64) as u64;
            let losses = games - wins;
            records[i][j] = (wins, 0, losses);
            records[j][i] = (losses, 0, wins);
        }
    }
    let elos = fit_elo(&records);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| elos[b].partial_cmp(&elos[a]).unwrap());
    order
        .into_iter()
        .enumerate()
        .map(|(rank, i)| EloRow {
            rank: rank + 1,
            agent_idx: i,
            elo: elos[i],
            row_wall_s: cells[i].iter().map(|c| c.wall_s).sum(),
        })
        .collect()
}

fn complete_matrix(cells: &[Vec<Cell>]) -> bool {
    cells.iter().flatten().all(|cell| cell.has_data())
}

fn field_to_pair_score(hero_share: f64, players: usize) -> f64 {
    let field_seats = (players - 1) as f64;
    let per_field_seat = (1.0 - hero_share) / field_seats;
    let denom = hero_share + per_field_seat;
    if denom <= 0.0 {
        0.5
    } else {
        (hero_share / denom).clamp(0.0, 1.0)
    }
}

fn selected_cell(args: &Args, roster: &[Entry], hero: usize, field: usize) -> bool {
    args.cells.is_empty()
        || args.cells.iter().any(|spec| {
            selector_matches(&spec.hero, roster, hero)
                && selector_matches(&spec.field, roster, field)
        })
}

fn validate_cell_specs(specs: &[CellSpec], roster: &[Entry]) -> Result<(), String> {
    for spec in specs {
        if !selector_matches_any(&spec.hero, roster) {
            return Err(format!(
                "cell hero selector '{}' does not match any roster agent or index",
                spec.hero
            ));
        }
        if !selector_matches_any(&spec.field, roster) {
            return Err(format!(
                "cell field selector '{}' does not match any roster agent or index",
                spec.field
            ));
        }
    }
    Ok(())
}

fn selector_matches_any(selector: &str, roster: &[Entry]) -> bool {
    (0..roster.len()).any(|idx| selector_matches(selector, roster, idx))
}

fn selector_matches(selector: &str, roster: &[Entry], idx: usize) -> bool {
    selector == "*"
        || selector.eq_ignore_ascii_case("all")
        || selector == idx.to_string()
        || selector == roster[idx].name
}

fn read_existing_metrics(path: &str, args: &Args) -> Result<ExistingMetrics, String> {
    if !Path::new(path).exists() {
        return Ok(ExistingMetrics::default());
    }

    let file = File::open(path).map_err(|e| format!("failed to read metrics '{path}': {e}"))?;
    let reader = BufReader::new(file);
    let mut existing = ExistingMetrics::default();
    for (line_idx, line) in reader.lines().enumerate() {
        let line_no = line_idx + 1;
        let line =
            line.map_err(|e| format!("failed to read metrics '{path}' line {line_no}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("failed to parse metrics '{path}' line {line_no}: {e}"))?;
        let Some(event) = value.get("event").and_then(Value::as_str) else {
            continue;
        };
        if !matches_current_run(&value, args) {
            continue;
        }
        match event {
            "tournament_agent" => {
                if let Some(agent) = value.get("agent").and_then(Value::as_str) {
                    existing.agents.insert(agent.to_string());
                }
            }
            "tournament_cell" => {
                let Some(hero) = value.get("hero").and_then(Value::as_str) else {
                    return Err(format!("metrics '{path}' line {line_no} is missing hero"));
                };
                let Some(field) = value.get("field").and_then(Value::as_str) else {
                    return Err(format!("metrics '{path}' line {line_no} is missing field"));
                };
                let games = read_u32(&value, "games", path, line_no)?;
                let win_share = read_f64(&value, "win_share", path, line_no)?;
                let mean_placement = read_f64(&value, "mean_placement", path, line_no)?;
                let decisions = read_u64(&value, "decisions", path, line_no)?;
                let wall_s = read_f64(&value, "wall_s", path, line_no)?;
                let recorded_seed = read_u64(&value, "cell_seed", path, line_no)?;
                let expected_seed = cell_seed(args.seed, hero, field);
                if recorded_seed != expected_seed {
                    return Err(format!(
                        "metrics '{path}' line {line_no} has cell_seed={recorded_seed}, \
                         expected {expected_seed} for {hero}:{field}"
                    ));
                }
                existing.cells.insert(
                    (hero.to_string(), field.to_string()),
                    Cell {
                        share_sum: win_share * f64::from(games),
                        placement_sum: mean_placement * f64::from(games),
                        games,
                        decisions,
                        wall_s,
                    },
                );
            }
            _ => {}
        }
    }
    Ok(existing)
}

fn matches_current_run(value: &Value, args: &Args) -> bool {
    matches_u64(value, "players", u64::from(args.players))
        && matches_u64(value, "dice", u64::from(args.dice))
        && matches_u64(value, "faces", u64::from(args.faces))
        && matches_u64(value, "seed", args.seed)
        && value.get("seed_policy").and_then(Value::as_str) == Some(SEED_POLICY)
}

fn matches_u64(value: &Value, key: &str, expected: u64) -> bool {
    value.get(key).and_then(Value::as_u64) == Some(expected)
}

fn read_u32(value: &Value, key: &str, path: &str, line_no: usize) -> Result<u32, String> {
    let raw = read_u64(value, key, path, line_no)?;
    raw.try_into()
        .map_err(|_| format!("metrics '{path}' line {line_no} has out-of-range {key}={raw}"))
}

fn read_u64(value: &Value, key: &str, path: &str, line_no: usize) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("metrics '{path}' line {line_no} is missing numeric {key}"))
}

fn read_f64(value: &Value, key: &str, path: &str, line_no: usize) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("metrics '{path}' line {line_no} is missing numeric {key}"))
}

fn open_metrics(path: Option<&str>, resume: bool) -> Result<Option<File>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create metrics dir '{}': {e}", parent.display()))?;
    }
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if resume {
        options.append(true);
    } else {
        options.truncate(true);
    }
    options
        .open(path)
        .map(Some)
        .map_err(|e| format!("failed to open metrics '{path}': {e}"))
}

fn write_config_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    roster: &[Entry],
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    writeln!(
        out,
        "{{\"event\":\"tournament_config\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"games\":{},\"seed\":{},\"rollouts\":{},\"mccfr_iters\":{},\
         \"mccfr_max_decision_plies\":{},\
         \"q_episodes\":{},\"mcts_worlds\":{},\"mcts_sims\":{},\
         \"net_search_rollouts\":{},\"net_search_plies\":{},\
         \"solve_iters\":{},\"solve_max_iters\":{},\"solve_restarts\":{},\
         \"rebel_iters\":{},\"rebel_depth\":{},\"resume\":{},\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"cell_filter\":\"{}\",\"agents\":[{}]}}",
        game.players,
        game.dice,
        game.faces,
        args.games,
        args.seed,
        args.rollouts,
        args.mccfr_iters,
        opt_u16_json(args.mccfr_max_decision_plies),
        args.q_episodes,
        args.mcts_worlds,
        args.mcts_sims,
        args.net_search_rollouts,
        args.net_search_plies,
        args.solve_iters,
        args.solve_max_iters,
        args.solve_restarts,
        args.rebel_iters,
        args.rebel_depth,
        args.resume,
        SEED_POLICY,
        HERO_SEAT_POLICY,
        json_escape(&cell_filter_label(&args.cells)),
        roster
            .iter()
            .map(|e| format!("\"{}\"", json_escape(&e.name)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn write_agent_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    entry: &Entry,
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    writeln!(
        out,
        "{{\"event\":\"tournament_agent\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"seed\":{},\"agent\":\"{}\",\"budget_per_move\":{},\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"setup_kind\":\"{}\",\"setup_units\":{},\"setup_rows\":{},\
         \"setup_wall_s\":{:.6},\"setup_units_per_s\":{:.3},\"source\":\"{}\"}}",
        game.players,
        game.dice,
        game.faces,
        args.seed,
        json_escape(&entry.name),
        entry.budget_per_move,
        SEED_POLICY,
        HERO_SEAT_POLICY,
        entry.setup.kind,
        entry.setup.units,
        entry.setup.rows,
        entry.setup.wall_s,
        entry.setup.units as f64 / entry.setup.wall_s.max(1e-9),
        json_escape(entry.setup.source.as_deref().unwrap_or(""))
    )
}

fn write_elo_metrics(
    out: &mut Option<File>,
    players: usize,
    game: &LiarsDice,
    args: &Args,
    roster: &[Entry],
    cells: &[Vec<Cell>],
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    for row in elo_rows(players, roster, cells) {
        let entry = &roster[row.agent_idx];
        writeln!(
            out,
            "{{\"event\":\"tournament_elo\",\"players\":{},\"dice\":{},\"faces\":{},\
             \"games\":{},\"seed\":{},\"rank\":{},\"agent\":\"{}\",\
             \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
             \"elo\":{:.6},\"budget_per_move\":{},\"setup_kind\":\"{}\",\
             \"setup_units\":{},\"setup_rows\":{},\"setup_wall_s\":{:.6},\
             \"row_wall_s\":{:.6},\"source\":\"{}\"}}",
            game.players,
            game.dice,
            game.faces,
            args.games,
            args.seed,
            row.rank,
            json_escape(&entry.name),
            SEED_POLICY,
            HERO_SEAT_POLICY,
            row.elo,
            entry.budget_per_move,
            entry.setup.kind,
            entry.setup.units,
            entry.setup.rows,
            entry.setup.wall_s,
            row.row_wall_s,
            json_escape(entry.setup.source.as_deref().unwrap_or(""))
        )?;
    }
    Ok(())
}

fn write_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    hero: &Entry,
    field: &Entry,
    cell: Cell,
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    let (lo, hi) = wilson(cell.share_sum, f64::from(cell.games), 1.96);
    let seed = cell_seed(args.seed, &hero.name, &field.name);
    writeln!(
        out,
        "{{\"event\":\"tournament_cell\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"games\":{},\"seed\":{},\"cell_seed\":{},\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"hero\":\"{}\",\"field\":\"{}\",\
         \"hero_budget_per_move\":{},\"field_budget_per_move\":{},\
         \"win_share\":{:.6},\"wilson_lo\":{:.6},\"wilson_hi\":{:.6},\
         \"mean_placement\":{:.6},\"decisions\":{},\"wall_s\":{:.6},\
         \"games_per_s\":{:.3},\"decisions_per_s\":{:.3}}}",
        game.players,
        game.dice,
        game.faces,
        cell.games,
        args.seed,
        seed,
        SEED_POLICY,
        HERO_SEAT_POLICY,
        json_escape(&hero.name),
        json_escape(&field.name),
        hero.budget_per_move,
        field.budget_per_move,
        cell.share(),
        lo,
        hi,
        cell.mean_placement(),
        cell.decisions,
        cell.wall_s,
        f64::from(cell.games) / cell.wall_s.max(1e-9),
        cell.decisions as f64 / cell.wall_s.max(1e-9)
    )
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn cell_filter_label(cells: &[CellSpec]) -> String {
    if cells.is_empty() {
        return String::new();
    }
    cells
        .iter()
        .map(|c| format!("{}:{}", c.hero, c.field))
        .collect::<Vec<_>>()
        .join(",")
}

fn short_name(s: &str) -> &str {
    s.split('-').next().unwrap_or(s)
}

fn cell_seed(base: u64, hero: &str, field: &str) -> u64 {
    let mut key = game_core::hash::combine(base, game_core::hash::fnv1a(SEED_POLICY.as_bytes()));
    key = game_core::hash::combine(key, game_core::hash::fnv1a(hero.as_bytes()));
    game_core::hash::combine(key, game_core::hash::fnv1a(field.as_bytes()))
}

fn parse_num<T>(value: &str, key: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("failed to parse {key}='{value}'"))
}

fn parse_u64(value: &str, key: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|_| format!("failed to parse {key}='{value}'"))
    } else {
        parse_num(value, key)
    }
}

fn parse_optional_u64(value: &str, key: &str) -> Result<Option<u64>, String> {
    if value.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_u64(value, key).map(Some)
    }
}

fn parse_optional_u16(value: &str, key: &str) -> Result<Option<u16>, String> {
    let Some(value) = parse_optional_u64(value, key)? else {
        return Ok(None);
    };
    value
        .try_into()
        .map(Some)
        .map_err(|_| format!("{key}={value} is too large for u16"))
}

fn parse_u32_list(value: &str, key: &str) -> Result<Vec<u32>, String> {
    if value == "none" {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    for raw in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let parsed = parse_num(raw, key)?;
        if values.contains(&parsed) {
            return Err(format!("duplicate {key} entry '{parsed}'"));
        }
        values.push(parsed);
    }
    if values.is_empty() {
        return Err(format!("{key} must include at least one value"));
    }
    Ok(values)
}

fn parse_bool(value: &str, key: &str) -> Result<bool, String> {
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "failed to parse {key}='{value}' as boolean (use 1/0, true/false, yes/no, on/off)"
        )),
    }
}

fn opt_u16_json(value: Option<u16>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn parse_cell_specs(value: &str) -> Result<Vec<CellSpec>, String> {
    if value == "none" {
        return Ok(Vec::new());
    }
    let mut specs = Vec::new();
    for raw in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((hero, field)) = raw.split_once(':') else {
            return Err(format!("cell entries must be hero:field, got '{raw}'"));
        };
        if hero.is_empty() || field.is_empty() {
            return Err(format!("cell entries must be hero:field, got '{raw}'"));
        }
        specs.push(CellSpec {
            hero: hero.to_string(),
            field: field.to_string(),
        });
    }
    if specs.is_empty() {
        return Err("cells must include at least one hero:field entry".to_string());
    }
    Ok(specs)
}

fn parse_checkpoint_specs(value: &str, key: &str) -> Result<Vec<CheckpointSpec>, String> {
    let mut specs = Vec::new();
    for raw in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (label, path) = match raw.split_once(':') {
            Some((label, path)) => {
                if label.is_empty() || path.is_empty() {
                    return Err(format!(
                        "{key} entries must be label:path or path, got '{raw}'"
                    ));
                }
                (clean_label(label), path.to_string())
            }
            None => (checkpoint_label(raw), raw.to_string()),
        };
        if specs
            .iter()
            .any(|spec: &CheckpointSpec| spec.label == label)
        {
            return Err(format!("duplicate {key} label '{label}'"));
        }
        specs.push(CheckpointSpec { label, path });
    }
    if specs.is_empty() {
        return Err(format!("{key} must include at least one checkpoint path"));
    }
    Ok(specs)
}

fn checkpoint_label(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(clean_label)
        .unwrap_or_else(|| "checkpoint".to_string())
}

fn clean_label(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "checkpoint".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            players: 2,
            dice: 1,
            faces: 2,
            games: 4,
            rollouts: 1,
            rollout_sweep: Vec::new(),
            ab_rollout_sweep: Vec::new(),
            mccfr_iters: 1,
            mccfr_seed: 11,
            mccfr_max_decision_plies: None,
            q_episodes: 1,
            q_seed: 13,
            mcts_worlds: 1,
            mcts_sims: 1,
            net_search_rollouts: 1,
            net_search_plies: 1,
            solve_iters: 1,
            solve_max_iters: 1,
            solve_restarts: 1,
            solve_seed: 17,
            solve_flat_iters: None,
            seed: 0x51A5_D1CE,
            agents: Some(vec!["random".to_string(), "belief".to_string()]),
            metrics: None,
            resume: false,
            cells: Vec::new(),
            net: None,
            nets: Vec::new(),
            rnad: None,
            rnads: Vec::new(),
            ppo: None,
            ppos: Vec::new(),
            history: None,
            histories: Vec::new(),
            rebel: None,
            rebels: Vec::new(),
            rebel_iters: 1,
            rebel_depth: 1,
        }
    }

    fn temp_metrics_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ld_tournament_{label}_{}_{}.jsonl",
            std::process::id(),
            game_core::hash::splitmix64(cell_seed(0, label, SEED_POLICY))
        ))
    }

    #[test]
    fn cell_seed_is_name_stable() {
        let seed = 0x51A5_D1CE;
        let cell = cell_seed(seed, "belief", "rollout-48");
        assert_eq!(cell, cell_seed(seed, "belief", "rollout-48"));
        assert_ne!(cell, cell_seed(seed, "rollout-48", "belief"));
        assert_ne!(cell, cell_seed(seed + 1, "belief", "rollout-48"));
        assert_ne!(cell, cell_seed(seed, "belief", "rollout-64"));
    }

    #[test]
    fn baseline_field_cycles_components_across_games() {
        let mut seen = std::collections::BTreeSet::new();
        for game_idx in 0..3 {
            for hero_seat in 0..5 {
                for player in 0..5 {
                    if player == hero_seat {
                        continue;
                    }
                    seen.insert(scheduled_field_component(player, hero_seat, game_idx, 5, 6));
                }
            }
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn baseline_field_roster_entry_does_not_require_rebel() {
        let mut args = test_args();
        args.agents = Some(vec![BASELINE_FIELD_NAME.to_string(), "random".to_string()]);
        args.rebel = Some("none".to_string());
        let roster = build_roster(&args).unwrap();
        let baseline = roster
            .iter()
            .find(|entry| entry.name == BASELINE_FIELD_NAME)
            .unwrap();
        assert_eq!(baseline.setup.kind, "baseline_field");
        assert_eq!(baseline.setup.units, 6);
        assert_eq!(baseline.field_components.as_ref().map(Vec::len), Some(6));
    }

    #[test]
    fn baseline_field_rebel_component_is_explicit_only() {
        let mut args = test_args();
        args.rebel = None;
        assert!(baseline_rebel_path(&args).is_none());

        args.rebel = Some("none".to_string());
        assert!(baseline_rebel_path(&args).is_none());

        args.rebel = Some("/tmp/rebel.bin".to_string());
        assert_eq!(baseline_rebel_path(&args), Some("/tmp/rebel.bin"));
    }

    #[test]
    fn rollout_sweep_expands_multiple_budgets() {
        let mut args = test_args();
        args.agents = Some(vec![
            "belief".to_string(),
            "rollout-sweep".to_string(),
            "ab-rollout-sweep".to_string(),
        ]);
        args.rollout_sweep = vec![4, 8];
        args.ab_rollout_sweep = vec![4];
        let roster = build_roster(&args).unwrap();
        let names = roster
            .iter()
            .map(|entry| (entry.name.as_str(), entry.budget_per_move))
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                ("belief", 0),
                ("rollout-4", 4),
                ("rollout-8", 8),
                ("ab-rollout-4", 96),
            ]
        );
    }

    #[test]
    fn rollout_sweep_requires_unique_budgets() {
        assert!(parse_u32_list("4,8,4", "rollout_sweep").is_err());
    }

    #[test]
    fn resume_ignores_legacy_index_seeded_rows() {
        let args = test_args();
        let path = temp_metrics_path("legacy");
        std::fs::write(
            &path,
            format!(
                "{{\"event\":\"tournament_cell\",\"players\":{},\"dice\":{},\"faces\":{},\
                 \"games\":4,\"seed\":{},\"hero\":\"random\",\"field\":\"belief\",\
                 \"win_share\":0.25,\"mean_placement\":1.5,\"decisions\":8,\"wall_s\":0.1}}\n",
                args.players, args.dice, args.faces, args.seed
            ),
        )
        .unwrap();

        let existing = read_existing_metrics(path.to_str().unwrap(), &args).unwrap();
        assert!(existing.cells.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn resume_accepts_current_name_stable_cell_seed() {
        let args = test_args();
        let path = temp_metrics_path("current");
        let hero = "random";
        let field = "belief";
        std::fs::write(
            &path,
            format!(
                "{{\"event\":\"tournament_cell\",\"players\":{},\"dice\":{},\"faces\":{},\
                 \"games\":4,\"seed\":{},\"cell_seed\":{},\"seed_policy\":\"{}\",\
                 \"hero\":\"{}\",\"field\":\"{}\",\
                 \"win_share\":0.25,\"mean_placement\":1.5,\"decisions\":8,\"wall_s\":0.1}}\n",
                args.players,
                args.dice,
                args.faces,
                args.seed,
                cell_seed(args.seed, hero, field),
                SEED_POLICY,
                hero,
                field
            ),
        )
        .unwrap();

        let existing = read_existing_metrics(path.to_str().unwrap(), &args).unwrap();
        let cell = existing
            .cells
            .get(&(hero.to_string(), field.to_string()))
            .expect("current cell row should resume");
        assert_eq!(cell.games, 4);
        assert_eq!(cell.decisions, 8);
        let _ = std::fs::remove_file(path);
    }
}
