//! Held-out population selection for learned Liar's Dice checkpoints.
//!
//! This is the C11 meta-method in the bake-off plan: take a population of
//! learned checkpoints (Deep CFR/distill nets, R-NaD, PPO, ReBeL), plus optional
//! learned-search / learned-solve wrappers for standard MLP checkpoints,
//! evaluate each against a held-out league of styled baselines plus
//! search/exploiter fields, optionally include the peer checkpoint population as
//! additional fields, and select by measured win-share. It does not rank by tiny
//! exploitability.
//!
//!     cargo run --release -p liars-dice --example population -- \
//!         rnads=rnad:runs/ld_rnad/best.bin ppos=ppo:runs/ld_ppo/best.bin \
//!         fields=honest-bayes,belief,rollout,is-mcts,mccfr,blueprint-search games=200
//!
//! The metrics file records every candidate/field cell and one
//! `population_selection` row with the selected checkpoint source.

use std::fs::{File, OpenOptions};
use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use game_core::stats::wilson;
use game_core::{Agent, Game, RandomAgent, Rng, Turn, win_share};
use liars_dice::rebel::{PbsNet, RebelAgent};
use liars_dice::{
    AbstractedMccfrAgent, AbstractedQAgent, AbstractedRolloutAgent, ActionAbstractionConfig,
    BidConditioned, DeterminizedMctsAgent, DiceShareValue, HistoryNetAgent, LiarsDice, NetAgent,
    NetOnlineSolveAgent, NetTruncRollout, OnlineSolveAgent, OnlineSolveConfig, ProbConfig,
    ProbabilisticAgent,
};
use solvers::Rollout;

const NET_SEARCH_CAND_CAP: u64 = 8;
const SEED_POLICY: &str = "name-stable-population-cell-v1";
const HERO_SEAT_POLICY: &str = "game-index-mod-players";
const BASELINE_FIELD_NAME: &str = "baseline-field";

struct Args {
    players: u8,
    dice: u8,
    faces: u8,
    games: u32,
    seed: u64,
    metrics: Option<String>,
    append: bool,
    fields: Vec<String>,
    include_peers: bool,
    rollouts: u32,
    mccfr_iters: u64,
    mccfr_seed: u64,
    q_episodes: u64,
    q_seed: u64,
    mcts_worlds: u32,
    mcts_sims: u32,
    include_search: bool,
    include_solves: bool,
    net_search_rollouts: u32,
    net_search_plies: u32,
    solve_iters: u64,
    solve_max_iters: u64,
    solve_restarts: usize,
    solve_seed: u64,
    solve_flat_iters: Option<u64>,
    rebel_iters: usize,
    rebel_depth: u32,
    nets: Vec<CheckpointSpec>,
    rnads: Vec<CheckpointSpec>,
    ppos: Vec<CheckpointSpec>,
    histories: Vec<CheckpointSpec>,
    rebels: Vec<CheckpointSpec>,
}

#[derive(Clone, Debug)]
struct CheckpointSpec {
    label: String,
    path: String,
}

struct Entry {
    name: String,
    method: String,
    source: String,
    budget_per_move: u64,
    setup_wall_s: f64,
    agent: Box<dyn Agent<LiarsDice>>,
    field_components: Option<Vec<FieldComponent>>,
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

#[derive(Clone, Copy, Default)]
struct Cell {
    share_sum: f64,
    placement_sum: f64,
    games: u32,
    decisions: u64,
    wall_s: f64,
}

#[derive(Clone, Copy)]
struct Selection {
    idx: usize,
    score: f64,
    wilson_lo: f64,
    wilson_hi: f64,
    games: u32,
    mean_placement: f64,
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
    let candidates = build_candidates(&args)?;
    if candidates.is_empty() {
        return Err("population needs at least one learned checkpoint".to_string());
    }
    let fields = build_fields(&args)?;
    if fields.is_empty() && (!args.include_peers || candidates.len() < 2) {
        return Err(
            "population needs at least one held-out field or two peer candidates".to_string(),
        );
    }

    let mut metrics = open_metrics(args.metrics.as_deref(), args.append)?;
    write_config_metric(&mut metrics, &game, &args, &candidates, &fields)
        .map_err(|e| format!("failed to write population config: {e}"))?;
    for cand in &candidates {
        write_entry_metric(&mut metrics, &game, &args, "population_candidate", cand)
            .map_err(|e| format!("failed to write candidate metric: {e}"))?;
    }
    for field in &fields {
        write_entry_metric(&mut metrics, &game, &args, "population_field", field)
            .map_err(|e| format!("failed to write field metric: {e}"))?;
    }

    println!(
        "Liar's Dice population selection: {}p{}d{}f, {} games/cell, seed={:#x}",
        args.players, args.dice, args.faces, args.games, args.seed
    );
    println!("seed policy: {SEED_POLICY}; hero seats: {HERO_SEAT_POLICY}");
    println!(
        "candidates: {}",
        candidates
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "held-out fields: {}{}",
        fields
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        if args.include_peers {
            " + peer checkpoints"
        } else {
            ""
        }
    );

    let mut all_cells: Vec<Vec<(String, Cell)>> = vec![Vec::new(); candidates.len()];
    for (ci, cand) in candidates.iter().enumerate() {
        for field in &fields {
            let seed = population_cell_seed(args.seed, &cand.name, &field.name, "field");
            let cell = evaluate_cell(&game, cand, field, args.games, seed);
            write_cell_metric(&mut metrics, &game, &args, cand, field, "field", cell)
                .map_err(|e| format!("failed to write field cell metric: {e}"))?;
            all_cells[ci].push((field.name.clone(), cell));
        }
        if args.include_peers {
            for (pi, peer) in candidates.iter().enumerate() {
                if pi == ci {
                    continue;
                }
                let seed = population_cell_seed(args.seed, &cand.name, &peer.name, "peer");
                let cell = evaluate_cell(&game, cand, peer, args.games, seed);
                write_cell_metric(&mut metrics, &game, &args, cand, peer, "peer", cell)
                    .map_err(|e| format!("failed to write peer cell metric: {e}"))?;
                all_cells[ci].push((peer.name.clone(), cell));
            }
        }
    }

    let selection = select_candidate(&all_cells)
        .ok_or_else(|| "no population cells were evaluated".to_string())?;
    write_selection_metric(
        &mut metrics,
        &game,
        &args,
        &candidates[selection.idx],
        selection,
    )
    .map_err(|e| format!("failed to write selection metric: {e}"))?;
    print_summary(&candidates, &all_cells, selection);
    Ok(())
}

impl Args {
    fn parse() -> Result<Self, String> {
        let default_solve = OnlineSolveConfig::default();
        let mut args = Self {
            players: 5,
            dice: 5,
            faces: 6,
            games: 200,
            seed: 0xD15C_EA5E,
            metrics: Some("runs/ld_population_metrics.jsonl".to_string()),
            append: false,
            fields: parse_list(
                "random,honest-bayes,aggressive-bluffer,conservative-caller,belief,rollout,\
                 abstract-rollout,is-mcts,baseline-field,mccfr,qlearn,blueprint-search",
            ),
            include_peers: true,
            rollouts: 48,
            mccfr_iters: 256,
            mccfr_seed: 0xC0F5_D1CE,
            q_episodes: 1000,
            q_seed: 0xA11C_E5E5,
            mcts_worlds: 8,
            mcts_sims: 32,
            include_search: true,
            include_solves: true,
            net_search_rollouts: 48,
            net_search_plies: 3,
            solve_iters: default_solve.iters,
            solve_max_iters: default_solve.max_iters,
            solve_restarts: default_solve.restarts,
            solve_seed: default_solve.seed,
            solve_flat_iters: default_solve.flat_iters,
            rebel_iters: 96,
            rebel_depth: 2,
            nets: Vec::new(),
            rnads: Vec::new(),
            ppos: Vec::new(),
            histories: Vec::new(),
            rebels: Vec::new(),
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
                "seed" => args.seed = parse_u64(value, key)?,
                "metrics" => {
                    args.metrics = if value == "none" {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "append" => args.append = parse_bool(value, key)?,
                "fields" => args.fields = parse_list(value),
                "include_peers" | "peers" => args.include_peers = parse_bool(value, key)?,
                "rollouts" => args.rollouts = parse_num(value, key)?,
                "mccfr_iters" => args.mccfr_iters = parse_num(value, key)?,
                "mccfr_seed" => args.mccfr_seed = parse_u64(value, key)?,
                "q_episodes" => args.q_episodes = parse_num(value, key)?,
                "q_seed" => args.q_seed = parse_u64(value, key)?,
                "mcts_worlds" => args.mcts_worlds = parse_num(value, key)?,
                "mcts_sims" => args.mcts_sims = parse_num(value, key)?,
                "include_search" | "search_candidates" => {
                    args.include_search = parse_bool(value, key)?;
                }
                "include_solves" | "solve_candidates" => {
                    args.include_solves = parse_bool(value, key)?;
                }
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
                "rebel_iters" => args.rebel_iters = parse_num(value, key)?,
                "rebel_depth" => args.rebel_depth = parse_num(value, key)?,
                "nets" => args.nets = parse_checkpoint_specs(value, key)?,
                "rnads" => args.rnads = parse_checkpoint_specs(value, key)?,
                "ppos" => args.ppos = parse_checkpoint_specs(value, key)?,
                "histories" | "history_nets" => {
                    args.histories = parse_checkpoint_specs(value, key)?;
                }
                "rebels" => args.rebels = parse_checkpoint_specs(value, key)?,
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
        if args.rollouts == 0 {
            return Err("rollouts must be positive".to_string());
        }
        if args.mccfr_iters == 0 || args.q_episodes == 0 {
            return Err("mccfr_iters and q_episodes must be positive".to_string());
        }
        if args.mcts_worlds == 0 || args.mcts_sims == 0 {
            return Err("mcts_worlds and mcts_sims must be positive".to_string());
        }
        if args.net_search_rollouts == 0 || args.net_search_plies == 0 {
            return Err("net_search_rollouts and net_search_plies must be positive".to_string());
        }
        if args.solve_iters == 0 || args.solve_max_iters == 0 || args.solve_restarts == 0 {
            return Err("solve budgets must be positive".to_string());
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

fn build_candidates(args: &Args) -> Result<Vec<Entry>, String> {
    let mut out = Vec::new();
    for spec in &args.nets {
        out.push(load_net_candidate("net", spec, 1)?);
        if args.include_search {
            out.push(load_net_search_candidate("net", spec, args)?);
        }
        if args.include_solves {
            out.push(load_solve_candidate("net", spec, args)?);
        }
    }
    for spec in &args.rnads {
        out.push(load_net_candidate("rnad", spec, 1)?);
        if args.include_search {
            out.push(load_net_search_candidate("rnad", spec, args)?);
        }
        if args.include_solves {
            out.push(load_solve_candidate("rnad", spec, args)?);
        }
    }
    for spec in &args.ppos {
        out.push(load_net_candidate("ppo", spec, 1)?);
        if args.include_search {
            out.push(load_net_search_candidate("ppo", spec, args)?);
        }
        if args.include_solves {
            out.push(load_solve_candidate("ppo", spec, args)?);
        }
    }
    for spec in &args.histories {
        out.push(load_history_candidate("history", spec, 1)?);
    }
    for spec in &args.rebels {
        out.push(load_rebel_candidate("rebel", spec, args)?);
    }
    Ok(out)
}

fn build_fields(args: &Args) -> Result<Vec<Entry>, String> {
    let mut out = Vec::new();
    for name in &args.fields {
        match name.as_str() {
            "" | "none" => {}
            "random" | "random-legal" => out.push(Entry::field("random", Box::new(RandomAgent), 0)),
            "honest" | "honest-bayes" => out.push(Entry::field(
                "honest-bayes",
                Box::new(ProbabilisticAgent::new(ProbConfig::honest_bayes())),
                0,
            )),
            "aggressive" | "aggressive-bluffer" => out.push(Entry::field(
                "aggressive-bluffer",
                Box::new(ProbabilisticAgent::new(ProbConfig::aggressive_bluffer())),
                0,
            )),
            "conservative" | "conservative-caller" => out.push(Entry::field(
                "conservative-caller",
                Box::new(ProbabilisticAgent::new(ProbConfig::conservative_caller())),
                0,
            )),
            "belief" | "prob" | "probabilistic" => out.push(Entry::field(
                "belief",
                Box::new(ProbabilisticAgent::default_agent()),
                0,
            )),
            "rollout" => out.push(Entry::field(
                format!("rollout-{}", args.rollouts),
                Box::new(Rollout::new(
                    args.rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                )),
                u64::from(args.rollouts),
            )),
            "abstract-rollout" | "ab-rollout" => {
                let abstraction = ActionAbstractionConfig::default();
                let max_candidates = abstraction.max_candidates as u64;
                out.push(Entry::field(
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
            "is-mcts" | "det-mcts" => {
                let abstraction = ActionAbstractionConfig::default();
                let max_candidates = abstraction.max_candidates as u64;
                out.push(Entry::field(
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
                out.push(build_baseline_field(args));
            }
            "mccfr" | "abstract-mccfr" => {
                let train_game = LiarsDice::new(args.players, args.dice, args.faces);
                let start = Instant::now();
                let agent =
                    AbstractedMccfrAgent::train(train_game, args.mccfr_iters, args.mccfr_seed);
                let wall_s = start.elapsed().as_secs_f64();
                let infosets = agent.num_infosets();
                out.push(Entry::field_with_setup(
                    format!("mccfr-{}i-{}k", args.mccfr_iters, infosets / 1000),
                    "mccfr",
                    Box::new(agent),
                    args.mccfr_iters,
                    wall_s,
                ));
            }
            "qlearn" | "q-learning" | "q" => {
                let train_game = LiarsDice::new(args.players, args.dice, args.faces);
                let start = Instant::now();
                let agent = AbstractedQAgent::train(train_game, args.q_episodes, args.q_seed);
                let wall_s = start.elapsed().as_secs_f64();
                let rows = agent.table_size();
                out.push(Entry::field_with_setup(
                    format!("qlearn-{}e-{}k", args.q_episodes, rows / 1000),
                    "qlearn",
                    Box::new(agent),
                    args.q_episodes,
                    wall_s,
                ));
            }
            "online-solve" | "dice-share-solve" => {
                out.push(build_blueprint_search_field(args, "online-solve"));
            }
            "blueprint-search" | "pluribus" | "pluribus-search" => {
                out.push(build_blueprint_search_field(args, "blueprint-search"));
            }
            other => {
                return Err(format!(
                    "unknown field '{other}' (random,honest-bayes,aggressive-bluffer,\
                     conservative-caller,belief,rollout,abstract-rollout,is-mcts,mccfr,\
                     qlearn,baseline-field,online-solve,blueprint-search,none)"
                ));
            }
        }
    }
    Ok(out)
}

fn build_baseline_field(args: &Args) -> Entry {
    let start = Instant::now();
    let hero_components = baseline_components(args);
    let source = hero_components
        .iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let field_components = baseline_components(args);
    let budget_per_move = field_components
        .iter()
        .map(|c| component_budget(&c.name, args))
        .max()
        .unwrap_or(0);
    Entry::composite_field(
        BASELINE_FIELD_NAME,
        "baseline-field",
        source,
        Box::new(BaselineFieldHero {
            components: hero_components,
        }),
        field_components,
        budget_per_move,
        start.elapsed().as_secs_f64(),
    )
}

fn baseline_components(args: &Args) -> Vec<FieldComponent> {
    vec![
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
    ]
}

fn component_budget(name: &str, args: &Args) -> u64 {
    if name.starts_with("rollout-") {
        u64::from(args.rollouts)
    } else {
        0
    }
}

fn build_blueprint_search_field(args: &Args, method: &str) -> Entry {
    let cfg = args.solve_config();
    let total_dice = u32::from(args.players) * u32::from(args.dice);
    let (iters, restarts) = cfg.budget_for_total_dice(total_dice);
    Entry::field_with_setup(
        format!("{method}-{}x{}", iters, restarts),
        method,
        Box::new(OnlineSolveAgent::with_config(|| DiceShareValue, cfg)),
        iters * restarts as u64 * u64::from(args.players),
        0.0,
    )
}

impl Entry {
    fn field(
        name: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        budget_per_move: u64,
    ) -> Self {
        let name = name.into();
        Self {
            method: name.clone(),
            name,
            source: String::new(),
            budget_per_move,
            setup_wall_s: 0.0,
            agent,
            field_components: None,
        }
    }

    fn field_with_setup(
        name: impl Into<String>,
        method: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        budget_per_move: u64,
        setup_wall_s: f64,
    ) -> Self {
        Self {
            name: name.into(),
            method: method.into(),
            source: String::new(),
            budget_per_move,
            setup_wall_s,
            agent,
            field_components: None,
        }
    }

    fn composite_field(
        name: impl Into<String>,
        method: impl Into<String>,
        source: impl Into<String>,
        agent: Box<dyn Agent<LiarsDice>>,
        field_components: Vec<FieldComponent>,
        budget_per_move: u64,
        setup_wall_s: f64,
    ) -> Self {
        Self {
            name: name.into(),
            method: method.into(),
            source: source.into(),
            budget_per_move,
            setup_wall_s,
            agent,
            field_components: Some(field_components),
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

fn load_net_candidate(
    kind: &str,
    spec: &CheckpointSpec,
    budget_per_move: u64,
) -> Result<Entry, String> {
    let start = Instant::now();
    let agent = NetAgent::load(Path::new(&spec.path))
        .map_err(|e| format!("failed to load {kind} checkpoint '{}': {e}", spec.path))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Entry {
        name: format!("{kind}-{}", spec.label),
        method: kind.to_string(),
        source: spec.path.clone(),
        budget_per_move,
        setup_wall_s: wall_s,
        agent: Box::new(agent),
        field_components: None,
    })
}

fn load_net_search_candidate(
    kind: &str,
    spec: &CheckpointSpec,
    args: &Args,
) -> Result<Entry, String> {
    let start = Instant::now();
    let bytes = std::fs::read(&spec.path).map_err(|e| {
        format!(
            "failed to read {kind} search checkpoint '{}': {e}",
            spec.path
        )
    })?;
    let agent =
        NetTruncRollout::from_bytes(&bytes, args.net_search_rollouts, args.net_search_plies)
            .map_err(|e| {
                format!(
                    "failed to load {kind} search checkpoint '{}': {e}",
                    spec.path
                )
            })?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Entry {
        name: format!(
            "{}-{}-{}x{}",
            net_search_prefix(kind),
            spec.label,
            args.net_search_rollouts,
            args.net_search_plies
        ),
        method: format!("{kind}+net-search"),
        source: spec.path.clone(),
        budget_per_move: u64::from(args.net_search_rollouts)
            * u64::from(args.net_search_plies)
            * NET_SEARCH_CAND_CAP,
        setup_wall_s: wall_s,
        agent: Box::new(agent),
        field_components: None,
    })
}

fn load_solve_candidate(kind: &str, spec: &CheckpointSpec, args: &Args) -> Result<Entry, String> {
    let cfg = args.solve_config();
    let total_dice = u32::from(args.players) * u32::from(args.dice);
    let (iters, restarts) = cfg.budget_for_total_dice(total_dice);
    let start = Instant::now();
    let bytes = std::fs::read(&spec.path).map_err(|e| {
        format!(
            "failed to read {kind} solve checkpoint '{}': {e}",
            spec.path
        )
    })?;
    let agent = NetOnlineSolveAgent::from_bytes_with_config(&bytes, cfg).map_err(|e| {
        format!(
            "failed to load {kind} solve checkpoint '{}': {e}",
            spec.path
        )
    })?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Entry {
        name: format!("solve-{kind}-{}-{iters}x{restarts}", spec.label),
        method: format!("{kind}+solve"),
        source: spec.path.clone(),
        budget_per_move: iters * restarts as u64 * u64::from(args.players),
        setup_wall_s: wall_s,
        agent: Box::new(agent),
        field_components: None,
    })
}

fn net_search_prefix(kind: &str) -> String {
    if kind == "net" {
        "net-search".to_string()
    } else {
        format!("net-search-{kind}")
    }
}

fn load_history_candidate(
    kind: &str,
    spec: &CheckpointSpec,
    budget_per_move: u64,
) -> Result<Entry, String> {
    let start = Instant::now();
    let agent = HistoryNetAgent::load(Path::new(&spec.path))
        .map_err(|e| format!("failed to load {kind} checkpoint '{}': {e}", spec.path))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Entry {
        name: format!("{kind}-{}", spec.label),
        method: kind.to_string(),
        source: spec.path.clone(),
        budget_per_move,
        setup_wall_s: wall_s,
        agent: Box::new(agent),
        field_components: None,
    })
}

fn load_rebel_candidate(kind: &str, spec: &CheckpointSpec, args: &Args) -> Result<Entry, String> {
    let start = Instant::now();
    let net = PbsNet::load(Path::new(&spec.path))
        .map_err(|e| format!("failed to load {kind} checkpoint '{}': {e}", spec.path))?;
    let wall_s = start.elapsed().as_secs_f64();
    Ok(Entry {
        name: format!(
            "{kind}-{}-{}x{}",
            spec.label, args.rebel_iters, args.rebel_depth
        ),
        method: kind.to_string(),
        source: spec.path.clone(),
        budget_per_move: args.rebel_iters as u64,
        setup_wall_s: wall_s,
        agent: Box::new(RebelAgent::with_config(
            net,
            args.rebel_iters,
            args.rebel_depth,
        )),
        field_components: None,
    })
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
            return (i + 1 + j) as f64 / 2.0;
        }
        i = j;
    }
    unreachable!("player must appear in placement table");
}

fn select_candidate(cells: &[Vec<(String, Cell)>]) -> Option<Selection> {
    cells
        .iter()
        .enumerate()
        .filter_map(|(idx, rows)| {
            let games: u32 = rows.iter().map(|(_, c)| c.games).sum();
            if games == 0 {
                return None;
            }
            let share_sum: f64 = rows.iter().map(|(_, c)| c.share_sum).sum();
            let placement_sum: f64 = rows.iter().map(|(_, c)| c.placement_sum).sum();
            let score = share_sum / f64::from(games);
            let (lo, hi) = wilson(share_sum, f64::from(games), 1.96);
            Some(Selection {
                idx,
                score,
                wilson_lo: lo,
                wilson_hi: hi,
                games,
                mean_placement: placement_sum / f64::from(games),
            })
        })
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap()
                .then_with(|| a.wilson_lo.partial_cmp(&b.wilson_lo).unwrap())
        })
}

fn print_summary(candidates: &[Entry], cells: &[Vec<(String, Cell)>], selection: Selection) {
    println!();
    println!("population scores:");
    for (idx, rows) in cells.iter().enumerate() {
        let games: u32 = rows.iter().map(|(_, c)| c.games).sum();
        if games == 0 {
            continue;
        }
        let share_sum: f64 = rows.iter().map(|(_, c)| c.share_sum).sum();
        let placement_sum: f64 = rows.iter().map(|(_, c)| c.placement_sum).sum();
        let (lo, hi) = wilson(share_sum, f64::from(games), 1.96);
        println!(
            "  {:24} score {:.3} [{:.3},{:.3}] place {:.2} over {} games",
            candidates[idx].name,
            share_sum / f64::from(games),
            lo,
            hi,
            placement_sum / f64::from(games),
            games
        );
    }
    println!(
        "\nselected: {}  score {:.3} [{:.3},{:.3}] source={}",
        candidates[selection.idx].name,
        selection.score,
        selection.wilson_lo,
        selection.wilson_hi,
        candidates[selection.idx].source
    );
}

fn write_config_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    candidates: &[Entry],
    fields: &[Entry],
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    writeln!(
        out,
        "{{\"event\":\"population_config\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"games\":{},\"seed\":{},\"rollouts\":{},\"solve_iters\":{},\
         \"mccfr_iters\":{},\"q_episodes\":{},\"mcts_worlds\":{},\"mcts_sims\":{},\
         \"include_search\":{},\"include_solves\":{},\"net_search_rollouts\":{},\
         \"net_search_plies\":{},\
         \"solve_max_iters\":{},\"solve_restarts\":{},\"rebel_iters\":{},\
         \"rebel_depth\":{},\"include_peers\":{},\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"candidates\":[{}],\"fields\":[{}]}}",
        game.players,
        game.dice,
        game.faces,
        args.games,
        args.seed,
        args.rollouts,
        args.solve_iters,
        args.mccfr_iters,
        args.q_episodes,
        args.mcts_worlds,
        args.mcts_sims,
        args.include_search,
        args.include_solves,
        args.net_search_rollouts,
        args.net_search_plies,
        args.solve_max_iters,
        args.solve_restarts,
        args.rebel_iters,
        args.rebel_depth,
        args.include_peers,
        SEED_POLICY,
        HERO_SEAT_POLICY,
        candidates
            .iter()
            .map(|e| format!("\"{}\"", json_escape(&e.name)))
            .collect::<Vec<_>>()
            .join(","),
        fields
            .iter()
            .map(|e| format!("\"{}\"", json_escape(&e.name)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn write_entry_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    event: &str,
    entry: &Entry,
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    writeln!(
        out,
        "{{\"event\":\"{}\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"seed\":{},\"name\":\"{}\",\"method\":\"{}\",\"source\":\"{}\",\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"budget_per_move\":{},\"setup_wall_s\":{:.6}}}",
        json_escape(event),
        game.players,
        game.dice,
        game.faces,
        args.seed,
        json_escape(&entry.name),
        json_escape(&entry.method),
        json_escape(&entry.source),
        SEED_POLICY,
        HERO_SEAT_POLICY,
        entry.budget_per_move,
        entry.setup_wall_s,
    )
}

fn write_cell_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    candidate: &Entry,
    field: &Entry,
    cell_kind: &str,
    cell: Cell,
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    let share = cell.share_sum / f64::from(cell.games).max(1.0);
    let (lo, hi) = wilson(cell.share_sum, f64::from(cell.games), 1.96);
    let seed = population_cell_seed(args.seed, &candidate.name, &field.name, cell_kind);
    writeln!(
        out,
        "{{\"event\":\"population_cell\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"games\":{},\"seed\":{},\"cell_seed\":{},\"cell_kind\":\"{}\",\
         \"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\",\
         \"candidate\":\"{}\",\"candidate_method\":\"{}\",\
         \"candidate_source\":\"{}\",\"field\":\"{}\",\"field_method\":\"{}\",\
         \"field_source\":\"{}\",\"candidate_budget_per_move\":{},\
         \"field_budget_per_move\":{},\"win_share\":{:.6},\"wilson_lo\":{:.6},\
         \"wilson_hi\":{:.6},\"mean_placement\":{:.6},\"decisions\":{},\
         \"wall_s\":{:.6},\"games_per_s\":{:.3},\"decisions_per_s\":{:.3}}}",
        game.players,
        game.dice,
        game.faces,
        cell.games,
        args.seed,
        seed,
        json_escape(cell_kind),
        SEED_POLICY,
        HERO_SEAT_POLICY,
        json_escape(&candidate.name),
        json_escape(&candidate.method),
        json_escape(&candidate.source),
        json_escape(&field.name),
        json_escape(&field.method),
        json_escape(&field.source),
        candidate.budget_per_move,
        field.budget_per_move,
        share,
        lo,
        hi,
        cell.placement_sum / f64::from(cell.games).max(1.0),
        cell.decisions,
        cell.wall_s,
        f64::from(cell.games) / cell.wall_s.max(1e-9),
        cell.decisions as f64 / cell.wall_s.max(1e-9),
    )
}

fn write_selection_metric(
    out: &mut Option<File>,
    game: &LiarsDice,
    args: &Args,
    selected: &Entry,
    selection: Selection,
) -> io::Result<()> {
    let Some(out) = out else {
        return Ok(());
    };
    writeln!(
        out,
        "{{\"event\":\"population_selection\",\"players\":{},\"dice\":{},\"faces\":{},\
         \"games\":{},\"seed\":{},\"selected\":\"{}\",\"method\":\"{}\",\
         \"source\":\"{}\",\"budget_per_move\":{},\"win_share\":{:.6},\
         \"wilson_lo\":{:.6},\"wilson_hi\":{:.6},\"mean_placement\":{:.6},\
         \"eval_games\":{},\"seed_policy\":\"{}\",\"hero_seat_policy\":\"{}\"}}",
        game.players,
        game.dice,
        game.faces,
        args.games,
        args.seed,
        json_escape(&selected.name),
        json_escape(&selected.method),
        json_escape(&selected.source),
        selected.budget_per_move,
        selection.score,
        selection.wilson_lo,
        selection.wilson_hi,
        selection.mean_placement,
        selection.games,
        SEED_POLICY,
        HERO_SEAT_POLICY,
    )
}

fn open_metrics(path: Option<&str>, append: bool) -> Result<Option<File>, String> {
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
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    options
        .open(path)
        .map(Some)
        .map_err(|e| format!("failed to open metrics '{path}': {e}"))
}

fn parse_checkpoint_specs(value: &str, key: &str) -> Result<Vec<CheckpointSpec>, String> {
    if value == "none" {
        return Ok(Vec::new());
    }
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
    Ok(specs)
}

fn parse_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_num<T: std::str::FromStr>(value: &str, key: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {key}: '{value}'"))
}

fn parse_u64(value: &str, key: &str) -> Result<u64, String> {
    parse_num(value, key)
}

fn parse_optional_u64(value: &str, key: &str) -> Result<Option<u64>, String> {
    if value == "none" {
        Ok(None)
    } else {
        parse_u64(value, key).map(Some)
    }
}

fn parse_bool(value: &str, key: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid bool for {key}: '{value}'")),
    }
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

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn population_cell_seed(base: u64, candidate: &str, field: &str, cell_kind: &str) -> u64 {
    let mut key = game_core::hash::combine(base, game_core::hash::fnv1a(SEED_POLICY.as_bytes()));
    key = game_core::hash::combine(key, game_core::hash::fnv1a(cell_kind.as_bytes()));
    key = game_core::hash::combine(key, game_core::hash::fnv1a(candidate.as_bytes()));
    game_core::hash::combine(key, game_core::hash::fnv1a(field.as_bytes()))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        let default_solve = OnlineSolveConfig::default();
        Args {
            players: 5,
            dice: 5,
            faces: 6,
            games: 4,
            seed: 0xD15C_EA5E,
            metrics: None,
            append: false,
            fields: vec![BASELINE_FIELD_NAME.to_string()],
            include_peers: false,
            rollouts: 2,
            mccfr_iters: 1,
            mccfr_seed: 11,
            q_episodes: 1,
            q_seed: 13,
            mcts_worlds: 1,
            mcts_sims: 1,
            include_search: false,
            include_solves: false,
            net_search_rollouts: 1,
            net_search_plies: 1,
            solve_iters: default_solve.iters,
            solve_max_iters: default_solve.max_iters,
            solve_restarts: default_solve.restarts,
            solve_seed: default_solve.seed,
            solve_flat_iters: default_solve.flat_iters,
            rebel_iters: 1,
            rebel_depth: 1,
            nets: Vec::new(),
            rnads: Vec::new(),
            ppos: Vec::new(),
            histories: Vec::new(),
            rebels: Vec::new(),
        }
    }

    #[test]
    fn population_cell_seed_is_name_stable() {
        let seed = 0xD15C_EA5E;
        let cell = population_cell_seed(seed, "rnad-a", "rollout-48", "field");

        assert_eq!(
            cell,
            population_cell_seed(seed, "rnad-a", "rollout-48", "field")
        );
        assert_ne!(
            cell,
            population_cell_seed(seed, "rollout-48", "rnad-a", "field")
        );
        assert_ne!(
            cell,
            population_cell_seed(seed, "rnad-a", "rollout-48", "peer")
        );
        assert_ne!(
            cell,
            population_cell_seed(seed + 1, "rnad-a", "rollout-48", "field")
        );
        assert_ne!(
            cell,
            population_cell_seed(seed, "rnad-a", "belief", "field")
        );
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
    fn population_baseline_field_builds_as_composite() {
        let args = test_args();
        let fields = build_fields(&args).unwrap();
        assert_eq!(fields.len(), 1);
        let baseline = &fields[0];
        assert_eq!(baseline.name, BASELINE_FIELD_NAME);
        assert_eq!(baseline.method, "baseline-field");
        assert_eq!(baseline.budget_per_move, u64::from(args.rollouts));
        assert_eq!(baseline.field_components.as_ref().map(Vec::len), Some(6));
        assert!(baseline.source.contains("honest-bayes"));
    }
}
