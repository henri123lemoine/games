//! The game/bot registry: every playable game registers an id, its options,
//! how to build a match against its bots and — when it supports bot-vs-bot
//! evaluation — one bot parser that both the play and compare paths share.
//! This is the single integration point a future web service reuses — it
//! serves whatever is registered here.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use game_core::{Agent, Game, NoSpec, hash};
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent};
use solvers::azero::{Mlp, Puct, PuctAgent};
use solvers::mcts::Mcts;
use solvers::{AlphaBeta, Rollout};
use twentyone::game::{Action as T21Action, T21State, TwentyOne};

use crate::compare::{
    BotBuilder, BotParser, BotSpec, BoxedAgent, CompareArgs, TourneyArgs, head_to_head,
    round_robin, run_field, run_pairs, vs_field,
};
use crate::runner::{AnyMatch, TypedMatch};

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
}

const fn opt(key: &'static str, value: &'static str, note: &'static str) -> OptSpec {
    OptSpec { key, value, note }
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
                if o.note.is_empty() {
                    format!("{}={}", o.key, o.value)
                } else {
                    format!("{}={} {}", o.key, o.value, o.note)
                }
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
    let spec = BotSpec {
        name: o.str("bot", default_bot),
        opts: o.clone(),
    };
    let builder = parse(&spec, o)?;
    let bots = (0..seats)
        .map(|p| (Some(p) != seat).then(|| builder(hash::combine(seed, p as u64))))
        .collect();
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

const CHESS_OPTS: &[OptSpec] = &[
    opt("depth", "5", ""),
    opt("seat", "0|1|watch", "(0=White)"),
    opt(
        "bot",
        "alphabeta|alphabeta-rich|azero|azero-gpu",
        "(azero-gpu: browser only)",
    ),
    opt("net", "data/azero/chess.bin", ""),
    opt("sims", "256", ""),
    opt("seed", "...", ""),
];

const LIARS_DICE_OPTS: &[OptSpec] = &[
    opt("players", "5", ""),
    opt("dice", "5", ""),
    opt("faces", "6", ""),
    opt("rollouts", "1000", ""),
    opt("bot", "rollout|belief|random", ""),
    opt("seat", "0|..|watch", ""),
    opt("seed", "...", ""),
];

const TWENTYONE_OPTS: &[OptSpec] = &[
    opt("hearts", "6", ""),
    opt("iters", "50000", "(training iters/subgame)"),
    opt("seat", "0|1|watch", ""),
    opt("seed", "...", ""),
];

const OTHELLO_OPTS: &[OptSpec] = &[
    opt("depth", "6", ""),
    opt("seat", "0|1|watch", "(0=Black)"),
    opt("bot", "alphabeta|mcts", ""),
    opt("seed", "...", ""),
];

const CONNECT4_OPTS: &[OptSpec] = &[
    opt("depth", "9", ""),
    opt("seat", "0|1|watch", ""),
    opt("bot", "alphabeta|mcts", ""),
    opt("seed", "...", ""),
];

const GO_OPTS: &[OptSpec] = &[
    opt("size", "9", ""),
    opt("sims", "6000", ""),
    opt("bot", "mcts|mcts-eval|mcts-spec", ""),
    opt("seat", "0|1|watch", "(0=Black)"),
    opt("seed", "...", ""),
];

const G2048_OPTS: &[OptSpec] = &[
    opt("bot", "mcts|mcts-eval", "(omit to play yourself)"),
    opt("sims", "200", ""),
    opt("depth", "8", ""),
    opt("seed", "...", ""),
];

const SNAKE_OPTS: &[OptSpec] = &[
    opt("width", "10", ""),
    opt("height", "10", ""),
    opt("bot", "mcts|mcts-eval", "(omit to play yourself)"),
    opt("sims", "200", ""),
    opt("depth", "12", ""),
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
                if o.str("bot", "alphabeta") == "azero-gpu" {
                    return make_external_versus(o, chess::Chess, &["sims"]);
                }
                make_versus(o, chess::Chess, "alphabeta", chess_bot)
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
                "rollout[:rollouts=1000] | belief | random",
                0,
                true,
                liars_dice_game,
                liars_dice_bot,
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
            id: "2048",
            name: "2048",
            solo: true,
            watch_bot: "mcts-eval",
            summary: "2048 (single-player) — play it, or watch an MCTS bot",
            opts: G2048_OPTS,
            make: Box::new(|o| make_solo(o, g2048::G2048, g2048_bot)),
            eval: None,
        },
        Entry {
            id: "snake",
            name: "Snake",
            solo: true,
            watch_bot: "mcts-eval",
            summary: "Snake (single-player) — play it, or watch an MCTS bot",
            opts: SNAKE_OPTS,
            make: Box::new(|o| {
                let game = snake::Snake::new(o.get("width", 10)?, o.get("height", 10)?);
                make_solo(o, game, snake_bot)
            }),
            eval: None,
        },
    ]
}

fn liars_dice_game(o: &Opts) -> Result<LiarsDice, String> {
    Ok(LiarsDice::new(
        o.get("players", 5)?,
        o.get("dice", 5)?,
        o.get("faces", 6)?,
    ))
}

fn go_game(o: &Opts) -> Result<go::Go, String> {
    Ok(go::Go::new(o.get("size", 9)?))
}

/// Single-player entry: with `bot=` set, that bot plays and you watch;
/// without it, you play. Bots come through the same [`BotParser`] convention
/// as the versus games, so the next single-player game (or a future solo
/// compare mode) adds a parser instead of copying this scaffold.
fn make_solo<G: game_core::GameUi + Sync + 'static>(
    o: &Opts,
    game: G,
    parse: BotParser<G>,
) -> Result<Box<dyn AnyMatch>, String> {
    let seed = o.get("seed", default_seed())?;
    let name = o.str("bot", "");
    let bot = if name.is_empty() {
        None
    } else {
        let spec = BotSpec {
            name,
            opts: o.clone(),
        };
        Some(parse(&spec, o)?(seed))
    };
    let human = if bot.is_some() { None } else { Some(0) };
    Ok(TypedMatch::new(game, vec![bot], human, seed).boxed())
}

/// The `mcts|mcts-eval` parser the single-player games share; `with_eval`
/// builds the eval-guided variant from `(sims, depth)`.
fn mcts_solo_bot<G: Game + 'static>(
    spec: &BotSpec,
    default_depth: u32,
    with_eval: fn(u32, u32) -> BoxedAgent<G>,
    game_name: &str,
) -> Result<BotBuilder<G>, String> {
    let sims: u32 = spec.opts.get("sims", 200)?;
    Ok(match spec.name.as_str() {
        "mcts" => Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<G>),
        "mcts-eval" => {
            let depth: u32 = spec.opts.get("depth", default_depth)?;
            Box::new(move |_| with_eval(sims, depth))
        }
        other => {
            return Err(format!(
                "unknown {game_name} bot '{other}' (mcts|mcts-eval)"
            ));
        }
    })
}

fn g2048_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<g2048::G2048>, String> {
    mcts_solo_bot(
        spec,
        8,
        |sims, depth| Box::new(Mcts::with_eval(sims, g2048::Heuristic2048, depth)),
        "2048",
    )
}

fn snake_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<snake::Snake>, String> {
    mcts_solo_bot(
        spec,
        12,
        |sims, depth| Box::new(Mcts::with_eval(sims, snake::SnakeEval, depth)),
        "snake",
    )
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
    let iters: u64 = o.get("iters", 50_000)?;
    // A pre-trained artifact (shipped on the web, written back after a native
    // train-at-startup) beats re-solving on every launch.
    let artifact = format!("data/twentyone/solver-h{hearts}.bin");
    let solver = match crate::artifacts::read(&artifact)
        .ok()
        .and_then(|b| twentyone::Solver::from_bytes(&b).ok())
        .filter(|s| s.start_hearts() == hearts)
    {
        Some(s) => s,
        None => {
            let mut solver = if hearts <= 2 {
                twentyone::Solver::with_hearts(0xD1CE, hearts)
            } else {
                twentyone::Solver::abstracted(0xD1CE, hearts)
            };
            eprintln!("training the Twenty-One solver ({iters} iters/subgame)...");
            solver.solve(iters);
            persist_twentyone(&solver, &artifact);
            solver
        }
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

/// Wasm matches are ephemeral; artifacts arrive via the host instead.
#[cfg(target_arch = "wasm32")]
fn persist_twentyone(_solver: &twentyone::Solver, _artifact: &str) {}

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

fn liars_dice_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<LiarsDice>, String> {
    Ok(match spec.name.as_str() {
        "rollout" => {
            let rollouts: u32 = spec.opts.get("rollouts", 1000)?;
            Box::new(move |_| {
                Box::new(Rollout::new(
                    rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                )) as BoxedAgent<LiarsDice>
            })
        }
        "belief" => {
            Box::new(|_| Box::new(ProbabilisticAgent::default_agent()) as BoxedAgent<LiarsDice>)
        }
        "random" => Box::new(|_| Box::new(game_core::RandomAgent) as BoxedAgent<LiarsDice>),
        other => {
            return Err(format!(
                "unknown liars-dice bot '{other}' (rollout|belief|random)"
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
