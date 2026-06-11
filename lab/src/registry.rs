//! The game/bot registry: every playable game registers an id, its options,
//! how to build a match against its bots and — when it supports bot-vs-bot
//! evaluation — one bot parser that both the play and compare paths share.
//! This is the single integration point a future web service reuses — it
//! serves whatever is registered here.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use game_core::{Agent, Game, NoSpec, hash};
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent};
use solvers::azero::{Mlp, PolicyValueEncoder, Puct, PuctAgent};
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

/// A registered game: how to play it, and (when it has a bot parser) how to
/// evaluate its bots against each other.
pub struct Entry {
    pub id: &'static str,
    pub summary: &'static str,
    pub opts_help: &'static str,
    pub make: MakeFn,
    pub eval: Option<EvalEntry>,
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

/// Parses `seat=` — the human's seat index, or `watch` to make every seat a
/// bot and spectate.
fn parse_seat(o: &Opts, seats: usize) -> Result<usize, String> {
    let s = o.str("seat", "0");
    if s == "watch" {
        return Ok(usize::MAX);
    }
    match s.parse::<usize>() {
        Ok(i) if i < seats => Ok(i),
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
        .map(|p| (p != seat).then(|| builder(hash::combine(seed, p as u64))))
        .collect();
    Ok(TypedMatch::new(game, bots, seat, seed).boxed())
}

pub fn entries() -> Vec<Entry> {
    vec![
        Entry {
            id: "chess",
            summary: "chess vs alpha-beta (perft-validated rules)",
            opts_help: "depth=5  seat=0|1|watch (0=White)  bot=alphabeta|alphabeta-rich|azero  \
                        net=data/azero/chess.bin  sims=256  seed=...",
            make: Box::new(|o| make_versus(o, chess::Chess, "alphabeta", chess_bot)),
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
            summary: "N-player Liar's Dice vs determinized-rollout bots",
            opts_help: "players=5 dice=5 faces=6 rollouts=1000 bot=rollout|belief|random \
                        seat=0|..|watch seed=...",
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
            summary: "Twenty-One vs the decomposed CFR+ solver (artifact or train-at-startup)",
            opts_help: "hearts=6 iters=50000 (training iters/subgame)  seat=0|1|watch  seed=...",
            make: Box::new(make_twentyone),
            eval: None,
        },
        Entry {
            id: "othello",
            summary: "Othello vs alpha-beta (weighted squares + mobility)",
            opts_help: "depth=6  seat=0|1|watch (0=Black)  bot=alphabeta|mcts  seed=...",
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
            summary: "Connect-4 vs alpha-beta",
            opts_help: "depth=9  seat=0|1|watch  bot=alphabeta|mcts  seed=...",
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
            summary: "Go (area scoring, komi 7.5) vs MCTS",
            opts_help: "size=9  sims=6000  bot=mcts|mcts-eval|mcts-spec  seat=0|1|watch (0=Black)  \
                        seed=...",
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
            summary: "2048 (single-player) — play it, or watch an MCTS bot",
            opts_help: "bot=mcts|mcts-eval (omit to play yourself)  sims=200  depth=8  seed=...",
            make: Box::new(make_2048),
            eval: None,
        },
        Entry {
            id: "snake",
            summary: "Snake (single-player) — play it, or watch an MCTS bot",
            opts_help: "width=10 height=10  bot=mcts|mcts-eval (omit to play yourself)  sims=200  \
                        depth=12  seed=...",
            make: Box::new(make_snake),
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

fn make_2048(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let seed = o.get("seed", default_seed())?;
    let sims: u32 = o.get("sims", 200)?;
    let bot: Option<Box<dyn Agent<g2048::G2048>>> = match o.str("bot", "").as_str() {
        "" => None,
        "mcts" => Some(Box::new(Mcts::new(sims))),
        "mcts-eval" => Some(Box::new(Mcts::with_eval(
            sims,
            g2048::Heuristic2048,
            o.get("depth", 8)?,
        ))),
        other => return Err(format!("unknown bot '{other}' (mcts|mcts-eval)")),
    };
    let human = if bot.is_some() { usize::MAX } else { 0 };
    Ok(TypedMatch::new(g2048::G2048, vec![bot], human, seed).boxed())
}

fn make_snake(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let game = snake::Snake::new(o.get("width", 10)?, o.get("height", 10)?);
    let seed = o.get("seed", default_seed())?;
    let sims: u32 = o.get("sims", 200)?;
    let bot: Option<Box<dyn Agent<snake::Snake>>> = match o.str("bot", "").as_str() {
        "" => None,
        "mcts" => Some(Box::new(Mcts::new(sims))),
        "mcts-eval" => Some(Box::new(Mcts::with_eval(
            sims,
            snake::SnakeEval,
            o.get("depth", 12)?,
        ))),
        other => return Err(format!("unknown bot '{other}' (mcts|mcts-eval)")),
    };
    let human = if bot.is_some() { usize::MAX } else { 0 };
    Ok(TypedMatch::new(game, vec![bot], human, seed).boxed())
}

/// Binds the chess crate's encoding to the azero trait (lab depends on both;
/// the chess crate itself stays solver-free).
struct ChessEnc;

impl PolicyValueEncoder<chess::Chess> for ChessEnc {
    fn input_len(&self) -> usize {
        chess::encode::INPUT_LEN
    }
    fn policy_len(&self) -> usize {
        chess::encode::POLICY_LEN
    }
    fn encode_state(&self, _g: &chess::Chess, s: &chess::Board) -> Vec<f32> {
        chess::encode::encode_board(s)
    }
    fn action_index(&self, _g: &chess::Chess, _s: &chess::Board, m: chess::Move) -> usize {
        chess::encode::move_index(m)
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
        PuctAgent(Puct::new(game, &ChessEnc, &self.net, self.sims)).act(game, state, player, rng)
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
            if p == seat {
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
                "unknown chess bot '{other}' (alphabeta|alphabeta-rich|azero)"
            ));
        }
    })
}

fn othello_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<othello::Othello>, String> {
    Ok(match spec.name.as_str() {
        "alphabeta" => {
            let depth: u32 = spec.opts.get("depth", 6)?;
            Box::new(move |_| {
                Box::new(AlphaBeta::new(
                    depth,
                    othello::OthelloEval,
                    othello::OthelloSpec,
                )) as BoxedAgent<othello::Othello>
            })
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 2000)?;
            Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<othello::Othello>)
        }
        other => return Err(format!("unknown othello bot '{other}' (alphabeta|mcts)")),
    })
}

fn connect4_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<connect4::Connect4>, String> {
    Ok(match spec.name.as_str() {
        "alphabeta" => {
            let depth: u32 = spec.opts.get("depth", 9)?;
            Box::new(move |_| {
                Box::new(AlphaBeta::new(depth, connect4::Connect4Eval, NoSpec))
                    as BoxedAgent<connect4::Connect4>
            })
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 2000)?;
            Box::new(move |_| Box::new(Mcts::new(sims)) as BoxedAgent<connect4::Connect4>)
        }
        other => return Err(format!("unknown connect4 bot '{other}' (alphabeta|mcts)")),
    })
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
        .subsec_nanos() as u64
        | 1
}

/// Wasm hosts always pass `seed=` explicitly (replays stay shareable); this
/// fallback only keeps seedless option maps from panicking.
#[cfg(target_arch = "wasm32")]
fn default_seed() -> u64 {
    0x5EED_BA5E_D00D | 1
}
