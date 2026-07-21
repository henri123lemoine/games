//! The game/bot registry: every playable game registers an id, its options,
//! how to build a match against its bots and — when it supports bot-vs-bot
//! evaluation — one bot parser that both the play and compare paths share.
//! This is the single integration point a future web service reuses — it
//! serves whatever is registered here.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use game_core::{Agent, Game, NoSpec, hash};
use liars_dice::rebel::{PbsNet, RebelAgent};
use liars_dice::{
    AbstractedMccfrAgent, AbstractedQAgent, AbstractedRolloutAgent, ActionAbstractionConfig,
    BidConditioned, DeterminizedMctsAgent, DiceShareValue, HistoryNetAgent, LiarsDice, NetAgent,
    NetOnlineSolveAgent, NetTruncRollout, OnlineSolveAgent, OnlineSolveConfig, ProbConfig,
    ProbabilisticAgent,
};
use nn_infer::Net;
use poker::{HoleSampler, Poker, PokerBot};
use solvers::azero::{Gather, PuctConfig, Search};
use solvers::azero::{Mlp, Puct, PuctAgent};
use solvers::mcts::Mcts;
use solvers::{AlphaBeta, Rollout};
use stratego::game::Stratego;
use stratego::{HeuristicBot, NetBot as StrategoNetBot, State as StrategoState};
use twentyone::game::{Action as T21Action, T21State, TwentyOne};

use crate::compare::{
    BotBuilder, BotParser, BotSpec, BoxedAgent, CompareArgs, TourneyArgs, head_to_head, parse_spec,
    round_robin, run_field, run_pairs, split_specs, vs_field,
};
use crate::runner::{AnyMatch, SimultaneousTypedMatch, TypedMatch};
use crate::simultaneous_compare::{BoxedSimultaneousAgent, SimultaneousBotBuilder};

/// Loose `key=value` options from the command line. Lookups are recorded so a
/// client can reject typos after a build succeeds (see [`Opts::unused`]);
/// values that fail to parse are hard errors, never silent defaults. Clones
/// share the access record (a clone is the same logical option map, e.g. when
/// the whole map doubles as a bot spec), so reads through either count.
#[derive(Clone)]
pub struct Opts {
    map: HashMap<String, String>,
    accessed: Arc<Mutex<HashSet<String>>>,
}

impl Opts {
    pub fn new(map: HashMap<String, String>) -> Self {
        Self {
            map,
            accessed: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        self.accessed
            .lock()
            .expect("opts lock")
            .insert(key.to_string());
        match self.map.get(key) {
            Some(v) => v
                .parse()
                .map_err(|_| format!("could not parse option {key}={v}")),
            None => Ok(default),
        }
    }

    pub fn str(&self, key: &str, default: &str) -> String {
        self.accessed
            .lock()
            .expect("opts lock")
            .insert(key.to_string());
        self.map.get(key).cloned().unwrap_or_else(|| default.into())
    }

    /// Errors unless every option was looked up — the typo guard every
    /// entry point runs once its reads are done.
    pub fn ensure_consumed(&self, what: &str) -> Result<(), String> {
        let unused = self.unused();
        if unused.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unused option(s) for {what}: {}",
                unused.join(", ")
            ))
        }
    }

    /// Options that were never looked up — typos, or keys the chosen
    /// bot/config does not use. Empty when everything was consumed.
    pub fn unused(&self) -> Vec<String> {
        let accessed = self.accessed.lock().expect("opts lock");
        let mut unused: Vec<String> = self
            .map
            .keys()
            .filter(|k| !accessed.contains(*k))
            .cloned()
            .collect();
        unused.sort();
        unused
    }
}

pub type MakeFn = Box<dyn Fn(&Opts) -> Result<Box<dyn AnyMatch>, String>>;
pub type CompareFn = Box<dyn Fn(&CompareArgs) -> Result<(), String>>;
pub type TourneyFn = Box<dyn Fn(&TourneyArgs) -> Result<(), String>>;
/// Non-printing pair runner: `(opts, spec_a, spec_b, seed, pair_range)` →
/// W-D-L from A's perspective. Used by external drivers (the web engine).
pub type PairsFn =
    Box<dyn Fn(&Opts, &str, &str, u64, std::ops::Range<u64>) -> Result<(u64, u64, u64), String>>;
/// Field runner for N-player configurations: hero A vs a field of B →
/// (hero strict wins, non-wins).
pub type FieldFn =
    Box<dyn Fn(&Opts, &str, &str, u64, std::ops::Range<u64>) -> Result<(u64, u64), String>>;

/// One declared game option: key, default (as shown to users), and an
/// optional clarifying note. The single source for both the CLI help line
/// and the web manifest's structured schema — so a wording tweak can never
/// silently change what the web settings drawer offers.
pub struct OptSpec {
    pub key: &'static str,
    pub value: &'static str,
    pub note: &'static str,
    /// Values of the `bot` option this option configures; empty for
    /// game-level options that always apply. Rich clients use this to show
    /// only the chosen bot's knobs (and to drop the rest, which the
    /// unused-option guard would otherwise reject).
    pub bots: &'static [&'static str],
    /// Only meaningful on native builds (e.g. training knobs — the browser
    /// never trains); web clients omit it, and the wasm engine never reads
    /// it, so the unused-option guard rejects it loudly if supplied.
    pub native_only: bool,
}

const fn opt(key: &'static str, value: &'static str, note: &'static str) -> OptSpec {
    OptSpec {
        key,
        value,
        note,
        bots: &[],
        native_only: false,
    }
}

const fn bot_opt(
    key: &'static str,
    value: &'static str,
    note: &'static str,
    bots: &'static [&'static str],
) -> OptSpec {
    OptSpec {
        key,
        value,
        note,
        bots,
        native_only: false,
    }
}

const fn native_opt(key: &'static str, value: &'static str, note: &'static str) -> OptSpec {
    OptSpec {
        key,
        value,
        note,
        bots: &[],
        native_only: true,
    }
}

/// A registered game: how to play it, and (when it has a bot parser) how to
/// evaluate its bots against each other.
pub struct Entry {
    pub id: &'static str,
    /// Display name for rich clients.
    pub name: &'static str,
    pub summary: &'static str,
    /// Single-player game: no `seat` option; `bot=` decides play vs watch.
    pub solo: bool,
    /// Bot spec for watch mode on solo games (versus games use `seat=watch`).
    pub watch_bot: &'static str,
    pub opts: &'static [OptSpec],
    pub make: MakeFn,
    pub eval: Option<EvalEntry>,
}

impl Entry {
    /// The human-readable option help, derived from [`Entry::opts`].
    pub fn opts_help(&self) -> String {
        self.opts
            .iter()
            .map(|o| {
                let mut s = format!("{}={}", o.key, o.value);
                if !o.bots.is_empty() {
                    s.push_str(&format!(" [{}]", o.bots.join("/")));
                }
                if !o.note.is_empty() {
                    s.push_str(&format!(" {}", o.note));
                }
                s
            })
            .collect::<Vec<_>>()
            .join("  ")
    }
}

/// Bot-vs-bot evaluation surface, built once per game by [`eval_entry`]:
/// `compare` dispatches paired GSPRT (2-player) or hero-vs-field binomial
/// SPRT (more seats); `tourney` is a round-robin Elo table; `pairs`/`field`
/// are the non-printing runners external drivers slice up.
pub struct EvalEntry {
    pub bots_help: &'static str,
    /// Whether configurations with more than two seats exist (field mode).
    pub has_field: bool,
    pub compare: CompareFn,
    pub tourney: TourneyFn,
    pub pairs: PairsFn,
    pub field: FieldFn,
}

/// Builds a game's whole [`EvalEntry`] from its config parser and bot parser —
/// the duplication this kills is five hand-written closures per game.
fn eval_entry<G: Game + Sync + 'static>(
    bots_help: &'static str,
    default_open: u64,
    has_field: bool,
    game_of: fn(&Opts) -> Result<G, String>,
    parse: BotParser<G>,
) -> EvalEntry {
    EvalEntry {
        bots_help,
        has_field,
        compare: Box::new(move |a| {
            let game = game_of(&a.opts)?;
            if game.num_players() == 2 {
                head_to_head(&game, a, default_open, parse)
            } else {
                vs_field(&game, a, parse)
            }
        }),
        tourney: Box::new(move |a| round_robin(&game_of(&a.opts)?, a, default_open, parse)),
        pairs: Box::new(move |o, a, b, s, r| {
            run_pairs(&game_of(o)?, o, a, b, default_open, parse, s, r)
        }),
        field: Box::new(move |o, a, b, s, r| run_field(&game_of(o)?, o, a, b, parse, s, r)),
    }
}

/// Parses `seat=` — the human's seat index, or `watch` (`None`) to make
/// every seat a bot and spectate.
fn parse_seat(o: &Opts, seats: usize) -> Result<Option<usize>, String> {
    let s = o.str("seat", "0");
    if s == "watch" {
        return Ok(None);
    }
    match s.parse::<usize>() {
        Ok(i) if i < seats => Ok(Some(i)),
        _ => Err(format!("seat must be 0..={} or 'watch'", seats - 1)),
    }
}

/// Builds a human-vs-bots match where every bot seat runs the bot the shared
/// parser produces for `bot=` (so the play menu and the compare harness can
/// never drift apart). The whole option map doubles as the bot's spec options.
fn make_versus<G: game_core::GameUi + Sync + 'static>(
    o: &Opts,
    game: G,
    default_bot: &str,
    parse: BotParser<G>,
) -> Result<Box<dyn AnyMatch>, String> {
    let seats = game.num_players();
    let seat = parse_seat(o, seats)?;
    let seed = o.get("seed", default_seed())?;
    // `bots=` gives one spec per seat (the human seat's is ignored), so distinct
    // bots can share a board — e.g. watching AlphaBeta vs MCTS. Each spec is
    // parsed into its own isolated options, so per-bot knobs never collide.
    // `bot=` (one type for every bot seat) stays the default.
    let bots_list = o.str("bots", "");
    let bots: Vec<Option<BoxedAgent<G>>> = if bots_list.is_empty() {
        let spec = BotSpec {
            name: o.str("bot", default_bot),
            opts: o.clone(),
        };
        let builder = parse(&spec, o)?;
        (0..seats)
            .map(|p| (Some(p) != seat).then(|| builder(hash::combine(seed, p as u64))))
            .collect()
    } else {
        let specs = split_specs(&bots_list);
        if specs.len() != seats {
            return Err(format!(
                "bots= needs one spec per seat ({seats}), got {}",
                specs.len()
            ));
        }
        let builders: Vec<Option<BotBuilder<G>>> = specs
            .iter()
            .map(|s| {
                let spec = parse_spec(s)?;
                // The GPU AlphaZero seat is driven by the client (page-side
                // WebGPU), not the engine — leave it empty so step() yields to
                // the driver, exactly as make_external_versus does. This lets a
                // GPU seat share a board with an in-engine bot.
                if spec.name == "azero-gpu" {
                    return Ok::<_, String>(None);
                }
                let builder = parse(&spec, o)?;
                spec.opts.ensure_consumed(&format!("bot '{s}'"))?;
                Ok(Some(builder))
            })
            .collect::<Result<Vec<_>, _>>()?;
        (0..seats)
            .map(|p| {
                if Some(p) == seat {
                    return None;
                }
                builders[p]
                    .as_ref()
                    .map(|b| b(hash::combine(seed, p as u64)))
            })
            .collect()
    };
    Ok(TypedMatch::new(game, bots, seat, seed).boxed())
}

/// Builds a match where every non-human seat is externally driven — the
/// client computes those moves and feeds them through `apply_human` (the
/// browser does this for the WebGPU azero bot). `client_opts` names options
/// the client reads, so the unused-option check accepts them.
fn make_external_versus<G: game_core::GameUi + Sync + 'static>(
    o: &Opts,
    game: G,
    client_opts: &[&str],
) -> Result<Box<dyn AnyMatch>, String> {
    let seats = game.num_players();
    let seat = parse_seat(o, seats)?;
    let seed = o.get("seed", default_seed())?;
    for key in client_opts {
        let _ = o.str(key, "");
    }
    let bots = (0..seats).map(|_| None).collect();
    Ok(TypedMatch::new(game, bots, seat, seed).boxed())
}

/// Builds a versus match, except that the `azero-gpu` bot is driven
/// client-side (the page runs the in-wasm search + leaf forward and feeds the
/// moves back through `apply_human`): that seat is left externally driven so
/// `step()` yields to the page, like go/chess/pente/snake `azero-gpu`.
/// `client_opts` names the options the client reads itself, so the
/// unused-option guard accepts them.
fn make_versus_or_gpu<G: game_core::GameUi + Sync + 'static>(
    o: &Opts,
    game: G,
    default_bot: &str,
    client_opts: &[&str],
    parse: BotParser<G>,
) -> Result<Box<dyn AnyMatch>, String> {
    if o.str("bot", default_bot) == "azero-gpu" {
        return make_external_versus(o, game, client_opts);
    }
    make_versus(o, game, default_bot, parse)
}

const CHESS_OPTS: &[OptSpec] = &[
    opt("seat", "0|1|watch", "(0=White)"),
    opt(
        "bot",
        "azero-gpu|alphabeta|alphabeta-rich|azero",
        "(azero-gpu: browser only)",
    ),
    bot_opt("depth", "5", "", &["alphabeta", "alphabeta-rich"]),
    bot_opt("net", "data/azero/chess.bin", "", &["azero"]),
    bot_opt("sims", "256", "", &["azero", "azero-gpu"]),
    opt("seed", "...", ""),
];

const LIARS_DICE_OPTS: &[OptSpec] = &[
    opt("players", "5", ""),
    opt("dice", "5", ""),
    opt("faces", "6", ""),
    opt(
        "bot",
        "rollout|abstract-rollout|is-mcts|mccfr|qlearn|belief|honest-bayes|aggressive-bluffer|conservative-caller|online-solve|net|rnad|ppo|history|net-search|solve|rebel|random",
        "",
    ),
    bot_opt("rollouts", "768", "", &["rollout", "abstract-rollout"]),
    bot_opt("mccfr_iters", "256", "", &["mccfr", "abstract-mccfr"]),
    bot_opt("mccfr_seed", "...", "", &["mccfr", "abstract-mccfr"]),
    bot_opt("q_episodes", "1000", "", &["qlearn", "q-learning", "q"]),
    bot_opt("q_seed", "...", "", &["qlearn", "q-learning", "q"]),
    bot_opt("mcts_worlds", "8", "", &["is-mcts"]),
    bot_opt("mcts_sims", "32", "", &["is-mcts"]),
    bot_opt(
        "net_search_rollouts",
        "48",
        "",
        &["net-search", "trunc-net", "net-trunc-rollout"],
    ),
    bot_opt(
        "net_search_plies",
        "3",
        "",
        &["net-search", "trunc-net", "net-trunc-rollout"],
    ),
    bot_opt(
        "solve_iters",
        "8000",
        "",
        &["online-solve", "dice-share-solve", "pluribus", "solve"],
    ),
    bot_opt(
        "solve_max_iters",
        "8000",
        "",
        &["online-solve", "dice-share-solve", "pluribus", "solve"],
    ),
    bot_opt(
        "solve_restarts",
        "3",
        "",
        &["online-solve", "dice-share-solve", "pluribus", "solve"],
    ),
    bot_opt(
        "solve_seed",
        "...",
        "",
        &["online-solve", "dice-share-solve", "pluribus", "solve"],
    ),
    bot_opt(
        "solve_flat_iters",
        "none",
        "",
        &["online-solve", "dice-share-solve", "pluribus", "solve"],
    ),
    bot_opt(
        "net",
        "runs/ld_value/best.bin|runs/ld_deepcfr/best.bin",
        "(solve value net or policy checkpoint)",
        &[
            "solve",
            "net",
            "deepcfr-net",
            "distill-net",
            "net-search",
            "trunc-net",
            "net-trunc-rollout",
        ],
    ),
    bot_opt(
        "rnad_net",
        "runs/ld_rnad/best.bin",
        "(R-NaD/NeuRD policy checkpoint)",
        &["rnad", "rnad-net", "neurd"],
    ),
    bot_opt(
        "ppo_net",
        "runs/ld_ppo/best.bin",
        "(PPO policy checkpoint)",
        &["ppo", "ppo-net"],
    ),
    bot_opt(
        "history_net",
        "runs/ld_history/best.bin",
        "(history-attention policy checkpoint)",
        &["history", "history-net", "history-rnad", "transformer"],
    ),
    bot_opt("rebel_net", "runs/ld_rebel/best.bin", "", &["rebel"]),
    bot_opt("rebel_iters", "1024", "", &["rebel"]),
    bot_opt("rebel_depth", "2", "", &["rebel"]),
    opt("seat", "0|..|watch", ""),
    opt("seed", "...", ""),
];

const TWENTYONE_OPTS: &[OptSpec] = &[
    opt("hearts", "6", ""),
    native_opt(
        "iters",
        "50000",
        "(training iters/subgame, used only when no solver artifact exists)",
    ),
    opt("seat", "0|1|watch", ""),
    opt("seed", "...", ""),
];

const OTHELLO_OPTS: &[OptSpec] = &[
    opt("seat", "0|1|watch", "(0=Black)"),
    opt("bot", "alphabeta|mcts", ""),
    bot_opt("depth", "6", "", &["alphabeta"]),
    bot_opt("sims", "2000", "", &["mcts"]),
    opt("seed", "...", ""),
];

const CONNECT4_OPTS: &[OptSpec] = &[
    opt("seat", "0|1|watch", ""),
    opt("bot", "alphabeta|mcts", ""),
    bot_opt("depth", "9", "", &["alphabeta"]),
    bot_opt("sims", "2000", "", &["mcts"]),
    opt("seed", "...", ""),
];

const PENTE_OPTS: &[OptSpec] = &[
    opt("size", "13", "(13 or 15; tournament-standard)"),
    opt(
        "seat",
        "0|1|watch",
        "(0=Black, plays the forced center first)",
    ),
    opt(
        "bot",
        "alphabeta|mcts|azero|azero-gpu",
        "(azero-gpu: browser only)",
    ),
    bot_opt("depth", "4", "", &["alphabeta"]),
    bot_opt(
        "sims",
        "4000",
        "(azero default 400)",
        &["mcts", "azero", "azero-gpu"],
    ),
    bot_opt("net", "data/azpente/azero-pente.azweb", "", &["azero"]),
    bot_opt(
        "vcf-nodes",
        "4000",
        "(move-time VCF node budget)",
        &["azero", "azero-gpu"],
    ),
    bot_opt(
        "vcf-depth",
        "8",
        "(VCF max attacker plies)",
        &["azero", "azero-gpu"],
    ),
    opt("seed", "...", ""),
];

const POKER_OPTS: &[OptSpec] = &[
    opt("players", "6", "(2..=9 seats)"),
    opt("stack", "200", "(starting stack in big blinds × SB; chips)"),
    opt("bot", "equity|rollout|call|random", ""),
    bot_opt(
        "samples",
        "2000",
        "(equity Monte-Carlo samples)",
        &["equity"],
    ),
    bot_opt("rollouts", "300", "", &["rollout"]),
    opt("seat", "0|..|watch", ""),
    opt("seed", "...", ""),
];

const GO_OPTS: &[OptSpec] = &[
    opt("size", "9", ""),
    opt("seat", "0|1|watch", "(0=Black)"),
    opt(
        "bot",
        "azero-gpu|mcts|mcts-eval|mcts-spec",
        "(azero-gpu: browser only)",
    ),
    bot_opt(
        "sims",
        "6000",
        "",
        &["azero-gpu", "mcts", "mcts-eval", "mcts-spec"],
    ),
    bot_opt("depth", "...", "(default size²)", &["mcts-eval"]),
    opt("seed", "...", ""),
];

const SNAKE_OPTS: &[OptSpec] = &[
    opt("players", "2", "(2..=4)"),
    opt(
        "mode",
        "standard|royale|constrictor|wrapped|wrapped-constrictor",
        "",
    ),
    opt(
        "food",
        "standard|one",
        "(one keeps exactly one apple on the board)",
    ),
    opt("seat", "0|..|watch", "(0=Snake A)"),
    opt("bot", "bns|random", ""),
    bot_opt("millis", "440", "(search time per move)", &["bns"]),
    bot_opt("depth", "255", "", &["bns"]),
    bot_opt("qdepth", "3", "", &["bns"]),
    bot_opt("model", "mcs|brs+|full", "", &["bns"]),
    bot_opt("tt-bits", "19", "", &["bns"]),
    opt(
        "food-spawn",
        "15",
        "(official source has 14% effective rate)",
    ),
    opt("minimum-food", "1", ""),
    opt("hazard-damage", "14", ""),
    opt("shrink-every", "25", "(royale turns)"),
    opt("seed", "...", ""),
];

const STRATEGO_OPTS: &[OptSpec] = &[
    opt(
        "setup",
        "random|manual",
        "(random: pre-deployed; manual: place your 40 pieces)",
    ),
    opt("seat", "0|1|watch", "(0=red, moves first)"),
    opt(
        "bot",
        "ataraxios|heuristic|random",
        "(ataraxios: the trained transformer)",
    ),
    bot_opt(
        "net_path",
        "runs/stratego/ataraxios.bin",
        "(ATRX1 move+setup export)",
        &["ataraxios"],
    ),
    bot_opt("temp", "0.25", "(sampling temperature)", &["ataraxios"]),
    opt("seed", "...", ""),
];

pub fn entries() -> Vec<Entry> {
    vec![
        Entry {
            id: "chess",
            name: "Chess",
            solo: false,
            watch_bot: "",
            summary: "chess vs alpha-beta (perft-validated rules)",
            opts: CHESS_OPTS,
            make: Box::new(|o| {
                make_versus_or_gpu(o, chess::Chess, "alphabeta", &["sims"], chess_bot)
            }),
            eval: Some(eval_entry(
                "alphabeta[:depth=5] | alphabeta-rich[:depth=5] (rich eval) | \
                 azero[:net=data/azero/chess.bin,sims=256]",
                6,
                false,
                |_| Ok(chess::Chess),
                chess_bot,
            )),
        },
        Entry {
            id: "liars-dice",
            name: "Liar's Dice",
            solo: false,
            watch_bot: "",
            summary: "N-player Liar's Dice vs determinized-rollout bots",
            opts: LIARS_DICE_OPTS,
            make: Box::new(|o| make_versus(o, liars_dice_game(o)?, "rollout", liars_dice_bot)),
            eval: Some(eval_entry(
                "rollout[:rollouts=768] | abstract-rollout[:rollouts=768] | \
                 is-mcts[:mcts_worlds=8,mcts_sims=32] | mccfr[:mccfr_iters=256] | \
                 qlearn[:q_episodes=1000] | belief | honest-bayes | aggressive-bluffer | \
                 conservative-caller | online-solve[:solve_iters=8000,solve_restarts=3] | \
                 net[:net=runs/ld_deepcfr/best.bin] | rnad[:rnad_net=runs/ld_rnad/best.bin] | \
                 ppo[:ppo_net=runs/ld_ppo/best.bin] | \
                 history[:history_net=runs/ld_history/best.bin] | \
                 net-search[:net=runs/ld_deepcfr/best.bin] | \
                 solve[:net=runs/ld_value/best.bin] | rebel[:rebel_net=runs/ld_rebel/best.bin] | random",
                0,
                true,
                liars_dice_game,
                liars_dice_bot,
            )),
        },
        Entry {
            id: "poker",
            name: "Texas Hold'em",
            solo: false,
            watch_bot: "",
            summary: "No-Limit Texas Hold'em (6-max) vs equity-rollout bots",
            opts: POKER_OPTS,
            // Sitting down to play is a continuous cash-game session (hand after
            // hand, stacks carried, button rotating). The eval path below keeps
            // the bare one-hand game so the bb/hand metric is unchanged.
            make: Box::new(|o| {
                make_versus(o, poker_game(o)?.with_session(true), "equity", poker_bot)
            }),
            // `compare`/`tourney` here report win share, which understates a
            // poker bot badly: only one seat wins each pot, so a single rotated
            // hero can't beat the fair 1/N by much even when it dominates. The
            // honest strength metric is bb/100 — run `poker`'s `bot_eval` example.
            eval: Some(eval_entry(
                "equity[:samples=2000] | rollout[:rollouts=300] | call | random",
                0,
                true,
                poker_game,
                poker_bot,
            )),
        },
        Entry {
            id: "twentyone",
            name: "Twenty-One",
            solo: false,
            watch_bot: "",
            summary: "Twenty-One vs the decomposed CFR+ solver (artifact or train-at-startup)",
            opts: TWENTYONE_OPTS,
            make: Box::new(make_twentyone),
            eval: None,
        },
        Entry {
            id: "othello",
            name: "Othello",
            solo: false,
            watch_bot: "",
            summary: "Othello vs alpha-beta (weighted squares + mobility)",
            opts: OTHELLO_OPTS,
            make: Box::new(|o| make_versus(o, othello::Othello, "alphabeta", othello_bot)),
            eval: Some(eval_entry(
                "alphabeta[:depth=6] | mcts[:sims=2000]",
                4,
                false,
                |_| Ok(othello::Othello),
                othello_bot,
            )),
        },
        Entry {
            id: "connect4",
            name: "Connect 4",
            solo: false,
            watch_bot: "",
            summary: "Connect-4 vs alpha-beta",
            opts: CONNECT4_OPTS,
            make: Box::new(|o| make_versus(o, connect4::Connect4, "alphabeta", connect4_bot)),
            eval: Some(eval_entry(
                "alphabeta[:depth=9] | mcts[:sims=2000]",
                4,
                false,
                |_| Ok(connect4::Connect4),
                connect4_bot,
            )),
        },
        Entry {
            id: "go",
            name: "Go",
            solo: false,
            watch_bot: "",
            summary: "Go (area scoring, komi 7.5) vs MCTS",
            opts: GO_OPTS,
            make: Box::new(|o| {
                if o.str("bot", "mcts") == "azero-gpu" {
                    return make_external_versus(o, go_game(o)?, &["sims", "size"]);
                }
                // Play wants a stronger default than compare's quick 2000.
                let sims: u32 = o.get("sims", 6000)?;
                let mut spec_opts = o.clone();
                spec_opts.map.insert("sims".into(), sims.to_string());
                make_versus(&spec_opts, go_game(o)?, "mcts", go_bot)
            }),
            eval: Some(eval_entry(
                "mcts[:sims=2000] | mcts-eval[:sims=2000,depth=NxN] | mcts-spec[:sims=2000]",
                0,
                false,
                go_game,
                go_bot,
            )),
        },
        Entry {
            id: "pente",
            name: "Pente",
            solo: false,
            watch_bot: "",
            summary: "Pente (custodial capture + five-in-a-row) vs alpha-beta",
            opts: PENTE_OPTS,
            make: Box::new(|o| {
                make_versus_or_gpu(
                    o,
                    pente_game(o)?,
                    "alphabeta",
                    &["sims", "size", "vcf-nodes", "vcf-depth"],
                    pente_bot,
                )
            }),
            eval: Some(eval_entry(
                "alphabeta[:depth=4] | mcts[:sims=4000] | random",
                0,
                false,
                pente_game,
                pente_bot,
            )),
        },
        Entry {
            id: "snake",
            name: "Battlesnake",
            solo: false,
            watch_bot: "",
            summary: "Canonical simultaneous Battlesnake vs best-node search",
            opts: SNAKE_OPTS,
            make: Box::new(make_battlesnake),
            eval: Some(battlesnake_eval_entry()),
        },
        Entry {
            id: "stratego",
            name: "Stratego",
            solo: false,
            watch_bot: "",
            summary: "Classic Stratego (hidden ranks) vs ataraxios, the trained net",
            opts: STRATEGO_OPTS,
            make: Box::new(make_stratego),
            eval: Some(eval_entry(
                "ataraxios[:net_path=runs/stratego/ataraxios.bin] | heuristic | random",
                0,
                false,
                |_| Ok(Stratego),
                stratego_bot,
            )),
        },
    ]
}

/// Stratego play builder. `setup=random` (default) skips the 80-square
/// deployment by starting from a random *legal* pre-deployed board; `setup=manual`
/// begins in the deployment phase so the human places their own side square by
/// square (the bot deploys itself through its `Agent`).
fn make_stratego(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let seat = parse_seat(o, 2)?;
    let seed = o.get("seed", default_seed())?;
    let setup = o.str("setup", "random");
    let spec = BotSpec {
        name: o.str("bot", "heuristic"),
        opts: o.clone(),
    };
    let builder = stratego_bot(&spec, o)?;
    let bots: Vec<Option<BoxedAgent<Stratego>>> = (0..2)
        .map(|p| (Some(p) != seat).then(|| builder(hash::combine(seed, p as u64))))
        .collect();
    match setup.as_str() {
        "manual" => Ok(TypedMatch::new(Stratego, bots, seat, seed).boxed()),
        "random" => {
            let mut rng = game_core::Rng::new(seed);
            let state: StrategoState = Stratego::random_play_state(&mut rng);
            Ok(TypedMatch::from_state(Stratego, state, bots, seat, seed).boxed())
        }
        other => Err(format!(
            "stratego setup must be 'random' or 'manual', got '{other}'"
        )),
    }
}

fn liars_dice_game(o: &Opts) -> Result<LiarsDice, String> {
    Ok(LiarsDice::new(
        o.get("players", 5)?,
        o.get("dice", 5)?,
        o.get("faces", 6)?,
    ))
}

fn poker_game(o: &Opts) -> Result<Poker, String> {
    let seats: u8 = o.get("players", 6)?;
    if !(2..=poker::MAX_SEATS as u8).contains(&seats) {
        return Err("poker players must be in 2..=9".into());
    }
    let stack: u32 = o.get("stack", 200)?;
    if stack < 2 {
        return Err("poker stack must be at least one big blind (2 chips)".into());
    }
    Ok(Poker::new(seats).with_blinds(1, 2).with_stack(stack))
}

fn go_game(o: &Opts) -> Result<go::Go, String> {
    Ok(go::Go::new(o.get("size", 9)?))
}

fn pente_game(o: &Opts) -> Result<pente::Pente, String> {
    let size: usize = o.get("size", 13)?;
    if !(5..=19).contains(&size) {
        return Err("pente size must be in 5..=19 (13 or 15 are standard)".into());
    }
    Ok(pente::Pente::new(size))
}

fn battlesnake_rules(o: &Opts, seed: u64) -> Result<snake::battlesnake::Rules, String> {
    use snake::battlesnake::{InitialFood, Mode, Rules};

    let mode = match o.str("mode", "standard").as_str() {
        "standard" => Mode::Standard,
        "royale" => Mode::Royale,
        "constrictor" => Mode::Constrictor,
        "wrapped" => Mode::Wrapped,
        "wrapped-constrictor" => Mode::WrappedConstrictor,
        other => {
            return Err(format!(
                "snake mode must be standard|royale|constrictor|wrapped|wrapped-constrictor, got '{other}'"
            ));
        }
    };
    // Parse the advanced knobs even when the one-apple preset overrides them,
    // so changing the preset in the web drawer cannot leave rejected leftovers.
    let configured_spawn = o.get("food-spawn", 15)?;
    let configured_minimum = o.get("minimum-food", 1)?;
    let (initial_food, food_spawn_chance, minimum_food) = match o.str("food", "standard").as_str() {
        "standard" => (InitialFood::Official, configured_spawn, configured_minimum),
        "one" => (InitialFood::One, 0, 1),
        other => return Err(format!("snake food must be standard|one, got '{other}'")),
    };
    Ok(Rules {
        mode,
        initial_food,
        food_spawn_chance,
        minimum_food,
        hazard_damage: o.get("hazard-damage", 14)?,
        shrink_every_n_turns: o.get("shrink-every", 25)?,
        seed,
    })
}

fn battlesnake_bot<const N: usize>(
    spec: &BotSpec,
    _o: &Opts,
) -> Result<SimultaneousBotBuilder<snake::battlesnake::Battlesnake<N>>, String> {
    use snake::battlesnake::search::{OpponentModel, SearchAgent, SearchConfig};

    match spec.name.as_str() {
        "random" => Ok(Box::new(|_| {
            Box::new(game_core::RandomSimultaneousAgent)
                as BoxedSimultaneousAgent<snake::battlesnake::Battlesnake<N>>
        })),
        "bns" => {
            let model = match spec.opts.str("model", "mcs").as_str() {
                "full" => OpponentModel::Full,
                "mcs" => OpponentModel::MoveCombination,
                "brs+" | "brs-plus" => OpponentModel::BestReplyPlus,
                other => {
                    return Err(format!("snake model must be full|mcs|brs+, got '{other}'"));
                }
            };
            let config = SearchConfig {
                time_limit: std::time::Duration::from_millis(spec.opts.get("millis", 440)?),
                max_depth: spec.opts.get("depth", u8::MAX)?,
                quiescence_depth: spec.opts.get("qdepth", 3)?,
                opponent_model: model,
                tt_bits: spec.opts.get("tt-bits", 19)?,
                ..SearchConfig::default()
            };
            Ok(Box::new(move |_| {
                Box::new(SearchAgent::<N>::new(config))
                    as BoxedSimultaneousAgent<snake::battlesnake::Battlesnake<N>>
            }))
        }
        other => Err(format!("unknown snake bot '{other}' (bns|random)")),
    }
}

fn make_battlesnake_n<const N: usize>(o: &Opts, seed: u64) -> Result<Box<dyn AnyMatch>, String> {
    use snake::battlesnake::Battlesnake;

    let game = Battlesnake::<N>::new(battlesnake_rules(o, seed)?);
    let seat = parse_seat(o, N)?;
    let bots_list = o.str("bots", "");
    let bots: Vec<Option<BoxedSimultaneousAgent<Battlesnake<N>>>> = if bots_list.is_empty() {
        let spec = BotSpec {
            name: o.str("bot", "bns"),
            opts: o.clone(),
        };
        let builder = battlesnake_bot::<N>(&spec, o)?;
        (0..N)
            .map(|player| {
                (Some(player) != seat).then(|| builder(hash::combine(seed, player as u64)))
            })
            .collect()
    } else {
        let specs = split_specs(&bots_list);
        if specs.len() != N {
            return Err(format!(
                "bots= needs one spec per seat ({N}), got {}",
                specs.len()
            ));
        }
        let builders: Vec<_> = specs
            .iter()
            .map(|text| {
                let spec = parse_spec(text)?;
                let builder = battlesnake_bot::<N>(&spec, o)?;
                spec.opts.ensure_consumed(&format!("bot '{text}'"))?;
                Ok::<_, String>(builder)
            })
            .collect::<Result<_, _>>()?;
        (0..N)
            .map(|player| {
                (Some(player) != seat).then(|| builders[player](hash::combine(seed, player as u64)))
            })
            .collect()
    };
    Ok(SimultaneousTypedMatch::new(game, bots, seat, seed).boxed())
}

fn make_battlesnake(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let players: usize = o.get("players", 2)?;
    let seed = o.get("seed", default_seed())?;
    match players {
        2 => make_battlesnake_n::<2>(o, seed),
        3 => make_battlesnake_n::<3>(o, seed),
        4 => make_battlesnake_n::<4>(o, seed),
        _ => Err("snake players must be 2..=4".into()),
    }
}

fn battlesnake_eval_entry() -> EvalEntry {
    use crate::simultaneous_compare as sim;
    use snake::battlesnake::Battlesnake;

    fn game<const N: usize>(o: &Opts, seed: u64) -> Result<Battlesnake<N>, String> {
        Ok(Battlesnake::new(battlesnake_rules(o, seed)?))
    }
    fn players(o: &Opts) -> Result<usize, String> {
        match o.get("players", 2)? {
            count @ 2..=4 => Ok(count),
            _ => Err("snake players must be 2..=4".into()),
        }
    }

    EvalEntry {
        bots_help: "bns[:millis=440,depth=255,qdepth=3,model=mcs,tt-bits=19] | random",
        has_field: true,
        compare: Box::new(move |args| match players(&args.opts)? {
            2 => sim::head_to_head(
                &game::<2>(&args.opts, args.seed)?,
                args,
                0,
                battlesnake_bot::<2>,
            ),
            3 => sim::vs_field(
                &game::<3>(&args.opts, args.seed)?,
                args,
                battlesnake_bot::<3>,
            ),
            4 => sim::vs_field(
                &game::<4>(&args.opts, args.seed)?,
                args,
                battlesnake_bot::<4>,
            ),
            _ => unreachable!(),
        }),
        tourney: Box::new(move |args| match players(&args.opts)? {
            2 => sim::round_robin(
                &game::<2>(&args.opts, args.seed)?,
                args,
                0,
                battlesnake_bot::<2>,
            ),
            _ => Err("snake tourney requires players=2; use compare field mode for 3-4".into()),
        }),
        pairs: Box::new(move |o, a, b, seed, range| match players(o)? {
            2 => sim::run_pairs(
                &game::<2>(o, seed)?,
                o,
                a,
                b,
                0,
                battlesnake_bot::<2>,
                seed,
                range,
            ),
            _ => Err("paired Battlesnake games require players=2".into()),
        }),
        field: Box::new(move |o, a, b, seed, range| match players(o)? {
            2 => sim::run_field(
                &game::<2>(o, seed)?,
                o,
                a,
                b,
                battlesnake_bot::<2>,
                seed,
                range,
            ),
            3 => sim::run_field(
                &game::<3>(o, seed)?,
                o,
                a,
                b,
                battlesnake_bot::<3>,
                seed,
                range,
            ),
            4 => sim::run_field(
                &game::<4>(o, seed)?,
                o,
                a,
                b,
                battlesnake_bot::<4>,
                seed,
                range,
            ),
            _ => unreachable!(),
        }),
    }
}

/// Shares the net (compare builders clone it per game) and runs a fresh PUCT
/// search per move.
struct AzeroBot {
    net: std::sync::Arc<Mlp>,
    sims: usize,
}

fn load_azero_net(path: &str) -> Result<std::sync::Arc<Mlp>, String> {
    let bytes = crate::artifacts::read(path)?;
    Mlp::from_bytes(&bytes)
        .map(std::sync::Arc::new)
        .map_err(|e| format!("failed to load azero net '{path}': {e}"))
}

impl Agent<chess::Chess> for AzeroBot {
    fn act(
        &self,
        game: &chess::Chess,
        state: &chess::Board,
        player: usize,
        rng: &mut game_core::Rng,
    ) -> usize {
        PuctAgent(Puct::new(
            game,
            &chess::encode::FlatEncoder,
            &self.net,
            self.sims,
        ))
        .act(game, state, player, rng)
    }
}

/// Plays the solved strategy greedily via the solver's draw probability.
struct SolverBot(std::sync::Arc<twentyone::Solver>);

impl Agent<TwentyOne> for SolverBot {
    fn act(
        &self,
        game: &TwentyOne,
        state: &T21State,
        player: usize,
        _rng: &mut game_core::Rng,
    ) -> usize {
        use game_core::Game;
        let actions = game.legal_actions(state);
        let draw = self.0.play_draw_prob(state.env(), player) > 0.5;
        actions
            .iter()
            .position(|a| matches!(a, T21Action::Draw) == draw)
            .unwrap_or(0)
    }
}

fn make_twentyone(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let hearts: u8 = o.get("hearts", 6)?;
    // A pre-trained artifact (shipped on the web, written back after a native
    // train-at-startup) is the only way to play; an artifact that exists but
    // fails to parse or was trained for different rules is a hard error,
    // never a silent retrain. Only native builds train at all.
    let artifact = format!("data/twentyone/solver-h{hearts}.bin");
    let solver = match crate::artifacts::read(&artifact) {
        Ok(bytes) => {
            let s =
                twentyone::Solver::from_bytes(&bytes).map_err(|e| format!("{artifact}: {e}"))?;
            if s.start_hearts() != hearts {
                return Err(format!(
                    "{artifact} was trained for hearts={}, not hearts={hearts}",
                    s.start_hearts()
                ));
            }
            s
        }
        Err(missing) => train_twentyone(o, hearts, &artifact, &missing)?,
    };
    let solver = std::sync::Arc::new(solver);
    let seat = parse_seat(o, 2)?;
    let game = TwentyOne::new(hearts);
    let bots: Vec<Option<Box<dyn Agent<TwentyOne>>>> = (0..2)
        .map(|p| {
            if Some(p) == seat {
                None
            } else {
                Some(Box::new(SolverBot(solver.clone())) as Box<dyn Agent<TwentyOne>>)
            }
        })
        .collect();
    Ok(TypedMatch::new(game, bots, seat, o.get("seed", default_seed())?).boxed())
}

/// Native train-at-startup when no artifact exists yet — the lab producing
/// its own cache, announced loudly and persisted for the next launch.
#[cfg(not(target_arch = "wasm32"))]
fn train_twentyone(
    o: &Opts,
    hearts: u8,
    artifact: &str,
    _missing: &str,
) -> Result<twentyone::Solver, String> {
    let iters: u64 = o.get("iters", 50_000)?;
    let mut solver = if hearts <= 2 {
        twentyone::Solver::with_hearts(0xD1CE, hearts)
    } else {
        twentyone::Solver::abstracted(0xD1CE, hearts)
    };
    eprintln!("training the Twenty-One solver ({iters} iters/subgame)...");
    solver.solve(iters);
    persist_twentyone(&solver, artifact);
    Ok(solver)
}

/// The browser never trains: a missing artifact is the host's bug to surface,
/// not something to paper over with an undertrained stand-in.
#[cfg(target_arch = "wasm32")]
fn train_twentyone(
    _o: &Opts,
    hearts: u8,
    artifact: &str,
    missing: &str,
) -> Result<twentyone::Solver, String> {
    Err(format!(
        "no solver shipped for hearts={hearts} ({missing}); the browser never trains — \
         solve it natively (`cargo run --release -p twentyone --example solve {hearts} 50000 {artifact}`) \
         and ship the artifact"
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn persist_twentyone(solver: &twentyone::Solver, artifact: &str) {
    if let Some(dir) = std::path::Path::new(artifact).parent()
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        eprintln!("note: could not create {}: {e}", dir.display());
        return;
    }
    match solver.save(artifact) {
        Ok(()) => eprintln!("saved the trained solver to {artifact} (reused next launch)"),
        Err(e) => eprintln!("note: could not save {artifact}: {e}"),
    }
}

fn chess_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<chess::Chess>, String> {
    let depth: u32 = spec.opts.get("depth", 5)?;
    Ok(match spec.name.as_str() {
        "alphabeta" => Box::new(move |_| {
            Box::new(AlphaBeta::new(depth, chess::MaterialEval, chess::ChessSpec))
                as BoxedAgent<chess::Chess>
        }),
        "alphabeta-rich" => Box::new(move |_| {
            Box::new(AlphaBeta::new(depth, chess::RichEval, chess::ChessSpec))
                as BoxedAgent<chess::Chess>
        }),
        "azero" => {
            let net = load_azero_net(&spec.opts.str("net", "data/azero/chess.bin"))?;
            let sims: usize = spec.opts.get("sims", 256)?;
            Box::new(move |_| {
                Box::new(AzeroBot {
                    net: net.clone(),
                    sims,
                }) as BoxedAgent<chess::Chess>
            })
        }
        other => {
            return Err(format!(
                "unknown chess bot '{other}' (alphabeta|alphabeta-rich|azero; \
                 azero-gpu plays only in the browser)"
            ));
        }
    })
}

/// The `alphabeta|mcts` parser the perfect-information games share; `ab`
/// builds the game's alpha-beta from a depth.
fn ab_or_mcts_bot<G: Game + 'static>(
    spec: &BotSpec,
    default_depth: u32,
    ab: fn(u32) -> BoxedAgent<G>,
    game_name: &str,
) -> Result<BotBuilder<G>, String> {
    Ok(match spec.name.as_str() {
        "alphabeta" => {
            let depth: u32 = spec.opts.get("depth", default_depth)?;
            Box::new(move |_| ab(depth))
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 2000)?;
            Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<G>)
        }
        other => {
            return Err(format!(
                "unknown {game_name} bot '{other}' (alphabeta|mcts)"
            ));
        }
    })
}

fn othello_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<othello::Othello>, String> {
    ab_or_mcts_bot(
        spec,
        6,
        |d| {
            Box::new(AlphaBeta::new(
                d,
                othello::OthelloEval,
                othello::OthelloSpec,
            ))
        },
        "othello",
    )
}

fn connect4_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<connect4::Connect4>, String> {
    ab_or_mcts_bot(
        spec,
        9,
        |d| Box::new(AlphaBeta::new(d, connect4::Connect4Eval, NoSpec)),
        "connect4",
    )
}

fn go_bot(spec: &BotSpec, o: &Opts) -> Result<BotBuilder<go::Go>, String> {
    let sims: u32 = spec.opts.get("sims", 2000)?;
    let size: usize = o.get("size", 9)?;
    Ok(match spec.name.as_str() {
        "mcts" => Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<go::Go>),
        "mcts-eval" => {
            let depth: u32 = spec.opts.get("depth", (size * size) as u32)?;
            Box::new(move |_| {
                Box::new(Mcts::with_eval(sims, go::GoEval, depth)) as BoxedAgent<go::Go>
            })
        }
        "mcts-spec" => {
            Box::new(move |_| Box::new(Mcts::with_spec(sims, go::GoSpec)) as BoxedAgent<go::Go>)
        }
        other => {
            return Err(format!(
                "unknown go bot '{other}' (mcts|mcts-eval|mcts-spec)"
            ));
        }
    })
}

/// The native Pente bot: an AlphaZero net-guided PUCT search with the
/// capture-aware VCF+VCT forcing solver wired in as the search's
/// [`game_core::TerminalProver`]. The solver proves a forced win at every leaf
/// the search expands (a cheap per-leaf budget), and the MCTS-solver backs that
/// proof up as an exact ±1 — so the tactical knowledge flows through the whole
/// tree, not just a root pre-check. Per move it runs `sims` of `Search<Pente>`
/// against the reference `nn-infer` forward, then plays the root's proven win
/// when the solver proves one, else the most-visited root move. The net is
/// shared across compare workers via `Arc`.
struct AzeroPenteBot {
    net: std::sync::Arc<Net>,
    enc: pente::PenteEncoder,
    sims: u32,
    vcf: pente::VcfConfig,
}

impl Agent<pente::Pente> for AzeroPenteBot {
    fn act(
        &self,
        game: &pente::Pente,
        state: &pente::PenteState,
        _player: usize,
        rng: &mut game_core::Rng,
    ) -> usize {
        self.search_move(game, state, rng)
    }
}

impl AzeroPenteBot {
    /// Runs the whole PUCT search to its visit budget on the CPU — the forcing
    /// solver proving every expanded leaf — evaluating each parked leaf with the
    /// reference forward (mirrors the wasm CPU driver). Returns the index (into
    /// `legal_actions`) of the root's proven win if the search proved one, else
    /// the most-visited root move. No root noise: deterministic full-strength
    /// play, not self-play.
    fn search_move(
        &self,
        game: &pente::Pente,
        state: &pente::PenteState,
        rng: &mut game_core::Rng,
    ) -> usize {
        let cfg = PuctConfig {
            sims: self.sims,
            root_noise: 0.0,
            ..PuctConfig::default()
        };
        let prover = pente::PenteProver { cfg: self.vcf };
        let mut search = Search::new(None);
        let mut results = Vec::new();
        while let Gather::Requests(reqs) = search.advance(
            game,
            &self.enc,
            state,
            &cfg,
            rng,
            std::mem::take(&mut results),
            &|_| false,
            Some(&prover),
        ) {
            results = reqs
                .iter()
                .map(|r| {
                    let (priors, value) = self.net.forward_support(&r.features, &[], &r.support);
                    solvers::azero::EvalResult { priors, value }
                })
                .collect();
        }
        // A solver-proven root win is exact — play the proven move over the
        // visit argmax. `best_proven_action` is correct for both a proof bubbled
        // up from a winning child and a root the prover proves *directly* (its
        // witnessing move pins the edge in `resolve`). The index is into
        // `root_actions` — the `legal_actions` order the search expanded — so it
        // is the action index `Agent::act` owes its caller.
        search
            .best_proven_action()
            .unwrap_or_else(|| solvers::azero::argmax(search.root_visits()))
    }
}

fn load_pente_net(path: &str) -> Result<std::sync::Arc<Net>, String> {
    let bytes = crate::artifacts::read(path)?;
    Net::parse(&bytes)
        .map(std::sync::Arc::new)
        .map_err(|e| format!("failed to load pente net '{path}': {e}"))
}

fn pente_bot(spec: &BotSpec, o: &Opts) -> Result<BotBuilder<pente::Pente>, String> {
    Ok(match spec.name.as_str() {
        "alphabeta" => {
            let depth: u32 = spec.opts.get("depth", 4)?;
            Box::new(move |_| {
                Box::new(AlphaBeta::new(depth, pente::PenteEval, pente::PenteSpec))
                    as BoxedAgent<pente::Pente>
            })
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 4000)?;
            Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<pente::Pente>)
        }
        "azero" => {
            let net = load_pente_net(&spec.opts.str("net", "data/azpente/azero-pente.azweb"))?;
            let sims: u32 = spec.opts.get("sims", 400)?;
            // A *per-leaf* forcing budget: the solver runs at every MCTS leaf as
            // the search's `TerminalProver`, so the default 200k-node depth-12
            // search (tuned for offline analysis) would tank throughput. A small
            // budget still proves the short forcing wins (open fours,
            // double-fours, fifth-pair captures, and at VCT double-threes) that
            // matter while staying cheap per leaf. VCT (continuous open-three +
            // capture threats) is the default; `vct=0` narrows to VCF-only
            // (fours/captures) as a speed lever.
            let vcf_nodes: u64 = spec.opts.get("vcf-nodes", 1500)?;
            let vcf_depth: u32 = spec.opts.get("vcf-depth", 7)?;
            let vct: u32 = spec.opts.get("vct", 1)?;
            let vcf = pente::VcfConfig::for_leaf(vcf_depth, vcf_nodes, vct != 0);
            let size: usize = o.get("size", 13)?;
            Box::new(move |_| {
                Box::new(AzeroPenteBot {
                    net: net.clone(),
                    enc: pente::PenteEncoder::new(size),
                    sims,
                    vcf,
                }) as BoxedAgent<pente::Pente>
            })
        }
        "random" => Box::new(|_| Box::new(game_core::RandomAgent) as BoxedAgent<pente::Pente>),
        other => {
            return Err(format!(
                "unknown pente bot '{other}' (alphabeta|mcts|azero|random)"
            ));
        }
    })
}

fn liars_dice_bot(spec: &BotSpec, o: &Opts) -> Result<BotBuilder<LiarsDice>, String> {
    Ok(match spec.name.as_str() {
        "rollout" => {
            let rollouts: u32 = spec.opts.get("rollouts", 768)?;
            Box::new(move |_| {
                Box::new(Rollout::new(
                    rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                )) as BoxedAgent<LiarsDice>
            })
        }
        "abstract-rollout" | "ab-rollout" => {
            let rollouts: u32 = spec.opts.get("rollouts", 768)?;
            Box::new(move |_| {
                Box::new(AbstractedRolloutAgent::with_config(
                    rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                    ActionAbstractionConfig::default(),
                )) as BoxedAgent<LiarsDice>
            })
        }
        "is-mcts" | "det-mcts" => {
            let worlds: u32 = spec.opts.get("mcts_worlds", 8)?;
            let sims: u32 = spec.opts.get("mcts_sims", 32)?;
            Box::new(move |_| {
                Box::new(DeterminizedMctsAgent::with_config(
                    worlds,
                    sims,
                    BidConditioned::default(),
                    ActionAbstractionConfig::default(),
                )) as BoxedAgent<LiarsDice>
            })
        }
        "mccfr" | "abstract-mccfr" => {
            let iters: u64 = spec.opts.get("mccfr_iters", 256)?;
            let seed: u64 = spec.opts.get("mccfr_seed", 0xC0F5_D1CE)?;
            let players: u8 = o.get("players", 5)?;
            let dice: u8 = o.get("dice", 5)?;
            let faces: u8 = o.get("faces", 6)?;
            Box::new(move |_| {
                let game = LiarsDice::new(players, dice, faces);
                Box::new(AbstractedMccfrAgent::train(game, iters, seed)) as BoxedAgent<LiarsDice>
            })
        }
        "qlearn" | "q-learning" | "q" => {
            let episodes: u64 = spec.opts.get("q_episodes", 1000)?;
            let seed: u64 = spec.opts.get("q_seed", 0xA11C_E5E5)?;
            let players: u8 = o.get("players", 5)?;
            let dice: u8 = o.get("dice", 5)?;
            let faces: u8 = o.get("faces", 6)?;
            Box::new(move |_| {
                let game = LiarsDice::new(players, dice, faces);
                Box::new(AbstractedQAgent::train(game, episodes, seed)) as BoxedAgent<LiarsDice>
            })
        }
        "online-solve" | "dice-share-solve" | "pluribus" => {
            let cfg = liars_dice_solve_config(spec)?;
            Box::new(move |_| {
                Box::new(OnlineSolveAgent::with_config(|| DiceShareValue, cfg))
                    as BoxedAgent<LiarsDice>
            })
        }
        "belief" => {
            Box::new(|_| Box::new(ProbabilisticAgent::default_agent()) as BoxedAgent<LiarsDice>)
        }
        "honest" | "honest-bayes" => Box::new(|_| {
            Box::new(ProbabilisticAgent::new(ProbConfig::honest_bayes())) as BoxedAgent<LiarsDice>
        }),
        "aggressive" | "aggressive-bluffer" => Box::new(|_| {
            Box::new(ProbabilisticAgent::new(ProbConfig::aggressive_bluffer()))
                as BoxedAgent<LiarsDice>
        }),
        "conservative" | "conservative-caller" => Box::new(|_| {
            Box::new(ProbabilisticAgent::new(ProbConfig::conservative_caller()))
                as BoxedAgent<LiarsDice>
        }),
        "solve" => {
            // DeepStack-style online subgame solving against the trained value
            // head. Load the net bytes once; each bot seat gets its own agent
            // (with its own inference cache) parsed from the shared bytes. The
            // continuation is rebuilt for the live game's (players, faces) on
            // every move, so one bot plays any config.
            let path = spec.opts.str("net", "runs/ld_value/best.bin");
            let cfg = liars_dice_solve_config(spec)?;
            let bytes = crate::artifacts::read(&path)?;
            // Fail loudly at build time if the checkpoint is unreadable, rather
            // than per-seat at the first move.
            NetOnlineSolveAgent::from_bytes_with_config(&bytes, cfg)
                .map_err(|e| format!("failed to load liars-dice value net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    NetOnlineSolveAgent::from_bytes_with_config(&bytes, cfg)
                        .expect("value net bytes already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "net" | "deepcfr-net" | "distill-net" => {
            // A single-forward policy checkpoint produced by the distillation
            // or Deep CFR trainers. Validate once, then give each seat its own
            // inference cache from the shared checkpoint bytes.
            let path = spec.opts.str("net", "runs/ld_deepcfr/best.bin");
            let bytes = crate::artifacts::read(&path)?;
            NetAgent::from_bytes(&bytes)
                .map_err(|e| format!("failed to load liars-dice policy net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    NetAgent::from_bytes(&bytes)
                        .expect("policy net bytes already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "rnad" | "rnad-net" | "neurd" => {
            // The R-NaD/NeuRD-family trainer emits the same MLP policy/value
            // artifact as the distillation and Deep CFR trainers, but keep it
            // named separately so bake-off reports don't collapse methods.
            let path = spec.opts.str("rnad_net", "runs/ld_rnad/best.bin");
            let bytes = crate::artifacts::read(&path)?;
            NetAgent::from_bytes(&bytes)
                .map_err(|e| format!("failed to load liars-dice R-NaD net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    NetAgent::from_bytes(&bytes)
                        .expect("R-NaD net bytes already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "ppo" | "ppo-net" => {
            // The PPO trainer exports the same MLP policy/value artifact as the
            // other learned policy nets; keep a named alias for bake-off reports.
            let path = spec.opts.str("ppo_net", "runs/ld_ppo/best.bin");
            let bytes = crate::artifacts::read(&path)?;
            NetAgent::from_bytes(&bytes)
                .map_err(|e| format!("failed to load liars-dice PPO net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    NetAgent::from_bytes(&bytes)
                        .expect("PPO net bytes already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "history" | "history-net" | "history-rnad" | "transformer" => {
            // C13 architecture variant: same policy vocabulary, but a wider
            // input with the compact public bid-history attention encoder.
            let path = spec.opts.str("history_net", "runs/ld_history/best.bin");
            let bytes = crate::artifacts::read(&path)?;
            HistoryNetAgent::from_bytes(&bytes)
                .map_err(|e| format!("failed to load liars-dice history net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    HistoryNetAgent::from_bytes(&bytes)
                        .expect("history net bytes already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "net-search" | "trunc-net" | "net-trunc-rollout" => {
            let path = spec.opts.str("net", "runs/ld_deepcfr/best.bin");
            let rollouts: u32 = spec.opts.get("net_search_rollouts", 48)?;
            let plies: u32 = spec.opts.get("net_search_plies", 3)?;
            let bytes = crate::artifacts::read(&path)?;
            NetTruncRollout::from_bytes(&bytes, rollouts, plies).map_err(|e| {
                format!("failed to load liars-dice net-search checkpoint '{path}': {e}")
            })?;
            Box::new(move |_| {
                Box::new(
                    NetTruncRollout::from_bytes(&bytes, rollouts, plies)
                        .expect("net-search checkpoint already validated at build time"),
                ) as BoxedAgent<LiarsDice>
            })
        }
        "rebel" => {
            let path = spec.opts.str("rebel_net", "runs/ld_rebel/best.bin");
            let iters: usize = spec.opts.get("rebel_iters", 1024)?;
            let depth: u32 = spec.opts.get("rebel_depth", 2)?;
            PbsNet::load(std::path::Path::new(&path))
                .map_err(|e| format!("failed to load liars-dice ReBeL checkpoint '{path}': {e}"))?;
            Box::new(move |_| {
                let net = PbsNet::load(std::path::Path::new(&path))
                    .expect("ReBeL checkpoint already validated at build time");
                Box::new(RebelAgent::with_config(net, iters, depth)) as BoxedAgent<LiarsDice>
            })
        }
        "random" | "random-legal" => {
            Box::new(|_| Box::new(game_core::RandomAgent) as BoxedAgent<LiarsDice>)
        }
        other => {
            return Err(format!(
                "unknown liars-dice bot '{other}' (rollout|abstract-rollout|is-mcts|mccfr|\
                 qlearn|belief|honest-bayes|aggressive-bluffer|conservative-caller|\
                 online-solve|net|deepcfr-net|distill-net|rnad|ppo|history|net-search|solve|rebel|random)"
            ));
        }
    })
}

fn liars_dice_solve_config(spec: &BotSpec) -> Result<OnlineSolveConfig, String> {
    let flat = spec.opts.str("solve_flat_iters", "none");
    let flat_iters = if flat.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(
            flat.parse()
                .map_err(|_| format!("could not parse option solve_flat_iters={flat}"))?,
        )
    };
    Ok(OnlineSolveConfig {
        iters: spec.opts.get("solve_iters", 8_000)?,
        max_iters: spec.opts.get("solve_max_iters", 8_000)?,
        restarts: spec.opts.get("solve_restarts", 3)?,
        seed: spec.opts.get("solve_seed", 0xA5_0117_0E50_17E5)?,
        flat_iters,
    })
}

fn poker_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<Poker>, String> {
    Ok(match spec.name.as_str() {
        "equity" => {
            let samples: u32 = spec.opts.get("samples", 2000)?;
            Box::new(move |_| {
                Box::new(PokerBot::new(poker::PokerStyle {
                    samples,
                    ..Default::default()
                })) as BoxedAgent<Poker>
            })
        }
        "rollout" => {
            let rollouts: u32 = spec.opts.get("rollouts", 300)?;
            Box::new(move |_| {
                Box::new(Rollout::new(rollouts, PokerBot::default_bot(), HoleSampler))
                    as BoxedAgent<Poker>
            })
        }
        "call" => Box::new(|_| Box::new(poker::AlwaysCall) as BoxedAgent<Poker>),
        "random" => Box::new(|_| Box::new(game_core::RandomAgent) as BoxedAgent<Poker>),
        other => {
            return Err(format!(
                "unknown poker bot '{other}' (equity|rollout|call|random)"
            ));
        }
    })
}

fn stratego_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<Stratego>, String> {
    Ok(match spec.name.as_str() {
        "ataraxios" => {
            let path = spec.opts.str("net_path", "runs/stratego/ataraxios.bin");
            let temp: f64 = spec.opts.get("temp", 0.25)?;
            let bytes = crate::artifacts::read(&path)?;
            StrategoNetBot::from_bytes(&bytes)
                .map_err(|e| format!("failed to load stratego net '{path}': {e}"))?;
            Box::new(move |_| {
                Box::new(
                    StrategoNetBot::from_bytes(&bytes)
                        .expect("stratego net bytes already validated at build time")
                        .with_temperature(temp as f32),
                ) as BoxedAgent<Stratego>
            })
        }
        "heuristic" => Box::new(|_| Box::new(HeuristicBot) as BoxedAgent<Stratego>),
        "random" => Box::new(|_| Box::new(game_core::RandomAgent) as BoxedAgent<Stratego>),
        other => {
            return Err(format!(
                "unknown stratego bot '{other}' (ataraxios|heuristic|random)"
            ));
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
        | 1
}

/// Wasm hosts always pass `seed=` explicitly (replays stay shareable); this
/// fallback only keeps seedless option maps from panicking.
#[cfg(target_arch = "wasm32")]
fn default_seed() -> u64 {
    0x5EED_BA5E_D00D | 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Game;

    /// A synthetic `AZNET1` Pente net of the right shape (8 planes,
    /// global-pool-spatial head, no ownership/score head): a tiny uniform fill,
    /// just enough to forward. The uniform priors/value make the *search* an
    /// uninformative baseline, so the forced win in the end-to-end test below
    /// can only come from the wired-in forcing solver, not the net.
    fn synth_pente_net(blocks: usize, c: usize, size: usize) -> Vec<u8> {
        let planes = pente::encode::PLANES;
        let floats = c * planes * 9
            + c
            + blocks * 2 * (c * c * 9 + c)
            + (c * c + c)
            + (3 * c * c + c)
            + c
            + (3 * c + 1)
            + (c * c + c)
            + (128 * 3 * c + 128)
            + (128 + 1);
        let arch = nn_infer::Arch {
            blocks,
            channels: c,
            planes,
            size,
            scalars: 0,
            head: nn_infer::HeadKind::GlobalPoolSpatial,
            policy_len: 0,
            flags: nn_infer::HeadFlags(0),
        };
        let mut b = arch.header_bytes();
        for _ in 0..floats {
            b.extend_from_slice(&0.02f32.to_le_bytes());
        }
        b
    }

    #[test]
    fn azero_pente_bot_plays_a_solver_proven_forced_win() {
        // The end-to-end native path: build the `bot=azero` agent with the
        // forcing solver wired in as the search's prover, and confirm it plays
        // a proven forced win. The position is an open four (both ends free),
        // which the solver proves at the root; with only a uniform synthetic
        // net the search alone could not reliably pick the exact win, so the
        // bot returning the winning completion shows the integrated
        // solver path drove the move.
        let game = pente::Pente::new(9);
        let state = game.parse_state(
            &[
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . X X X X . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
                ". . . . . . . . .",
            ],
            0,
            [0, 0],
        );
        let net = Net::parse(&synth_pente_net(2, 6, 9)).expect("synthetic pente net");
        let bot = AzeroPenteBot {
            net: std::sync::Arc::new(net),
            enc: pente::PenteEncoder::new(9),
            sims: 64,
            vcf: pente::VcfConfig {
                max_depth: 7,
                max_nodes: 1500,
                ..pente::VcfConfig::default()
            },
        };
        let mut rng = game_core::Rng::new(7);
        let idx = bot.act(&game, &state, 0, &mut rng);
        let action = game.legal_actions(&state)[idx];
        // Either open end completes the five — the only proven wins here.
        let wins = [
            pente::PenteAction(game.point("b5").unwrap()),
            pente::PenteAction(game.point("g5").unwrap()),
        ];
        assert!(
            wins.contains(&action),
            "the azero bot must play the solver-proven winning move, got {action:?}"
        );
    }
}
