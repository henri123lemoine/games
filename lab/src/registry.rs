//! The game/bot registry: every playable game registers an id, its options,
//! and how to build a match against its bots. This is the single integration
//! point a future web service reuses — it serves whatever is registered here.

use std::collections::HashMap;

use game_core::{Agent, NoSpec};
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent, RandomAgent};
use solvers::azero::{Mlp, PolicyValueEncoder, Puct, PuctAgent};
use solvers::mcts::Mcts;
use solvers::{AlphaBeta, Rollout};
use twentyone::game::{Action as T21Action, T21State, TwentyOne};

use crate::compare::{
    BotBuilder, BotSpec, BoxedAgent, CompareArgs, TourneyArgs, head_to_head, round_robin,
    run_field, run_pairs, vs_field,
};
use crate::runner::{AnyMatch, TypedMatch};

/// Loose `key=value` options from the command line.
pub struct Opts(pub HashMap<String, String>);

impl Opts {
    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.0
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    pub fn str(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.into())
    }
}

pub struct Entry {
    pub id: &'static str,
    pub summary: &'static str,
    pub opts_help: &'static str,
    pub make: fn(&Opts) -> Result<Box<dyn AnyMatch>, String>,
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

pub fn entries() -> Vec<Entry> {
    vec![
        Entry {
            id: "chess",
            summary: "chess vs alpha-beta (perft-validated rules)",
            opts_help: "depth=5  seat=0|1|watch (0=White)  bot=alphabeta|azero  net=data/azero/chess.bin  sims=256  seed=...",
            make: make_chess,
        },
        Entry {
            id: "liars-dice",
            summary: "N-player Liar's Dice vs determinized-rollout bots",
            opts_help: "players=5 dice=5 faces=6 rollouts=1000 bot=rollout|belief|random seat=0|..|watch seed=...",
            make: make_liars_dice,
        },
        Entry {
            id: "twentyone",
            summary: "Twenty-One vs the decomposed CFR+ solver (trains at startup)",
            opts_help: "hearts=6 iters=50000 (training iters/subgame)  seat=0|1|watch  seed=...",
            make: make_twentyone,
        },
        Entry {
            id: "othello",
            summary: "Othello vs alpha-beta (weighted squares + mobility)",
            opts_help: "depth=6  seat=0|1|watch (0=Black)  seed=...",
            make: make_othello,
        },
        Entry {
            id: "connect4",
            summary: "Connect-4 vs alpha-beta",
            opts_help: "depth=9  seat=0|1|watch  seed=...",
            make: make_connect4,
        },
        Entry {
            id: "go",
            summary: "Go (area scoring, komi 7.5) vs MCTS",
            opts_help: "size=9  sims=6000  seat=0|1|watch (0=Black)  seed=...",
            make: make_go,
        },
        Entry {
            id: "2048",
            summary: "2048 (single-player) — play it, or watch an MCTS bot",
            opts_help: "bot=mcts|mcts-eval (omit to play yourself)  sims=200  depth=8  seed=...",
            make: make_2048,
        },
        Entry {
            id: "snake",
            summary: "Snake (single-player) — play it, or watch an MCTS bot",
            opts_help: "width=10 height=10  bot=mcts|mcts-eval (omit to play yourself)  sims=200  depth=12  seed=...",
            make: make_snake,
        },
    ]
}

fn make_2048(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let seed = o.get("seed", default_seed());
    let sims: u32 = o.get("sims", 200);
    let bot: Option<Box<dyn Agent<g2048::G2048>>> = match o.str("bot", "").as_str() {
        "" => None,
        "mcts" => Some(Box::new(Mcts::new(sims, seed ^ 0x2048))),
        "mcts-eval" => Some(Box::new(Mcts::with_eval(
            sims,
            g2048::Heuristic2048,
            o.get("depth", 8),
            seed ^ 0x2048,
        ))),
        other => return Err(format!("unknown bot '{other}' (mcts|mcts-eval)")),
    };
    let human = if bot.is_some() { usize::MAX } else { 0 };
    Ok(TypedMatch::new(g2048::G2048, vec![bot], human, seed).boxed())
}

fn make_snake(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let game = snake::Snake::new(o.get("width", 10), o.get("height", 10));
    let seed = o.get("seed", default_seed());
    let sims: u32 = o.get("sims", 200);
    let bot: Option<Box<dyn Agent<snake::Snake>>> = match o.str("bot", "").as_str() {
        "" => None,
        "mcts" => Some(Box::new(Mcts::new(sims, seed ^ 0x57AE))),
        "mcts-eval" => Some(Box::new(Mcts::with_eval(
            sims,
            snake::SnakeEval,
            o.get("depth", 12),
            seed ^ 0x57AE,
        ))),
        other => return Err(format!("unknown bot '{other}' (mcts|mcts-eval)")),
    };
    let human = if bot.is_some() { usize::MAX } else { 0 };
    Ok(TypedMatch::new(game, vec![bot], human, seed).boxed())
}

fn make_othello(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 6);
    let seat = parse_seat(o, 2)?;
    let bot = || -> Box<dyn Agent<othello::Othello>> {
        Box::new(AlphaBeta::new(
            depth,
            othello::OthelloEval,
            othello::OthelloSpec,
        ))
    };
    let bots = (0..2)
        .map(|p| if p == seat { None } else { Some(bot()) })
        .collect();
    Ok(TypedMatch::new(othello::Othello, bots, seat, o.get("seed", default_seed())).boxed())
}

fn make_connect4(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 9);
    let seat = parse_seat(o, 2)?;
    let bot = || -> Box<dyn Agent<connect4::Connect4>> {
        Box::new(AlphaBeta::new(depth, connect4::Connect4Eval, NoSpec))
    };
    let bots = (0..2)
        .map(|p| if p == seat { None } else { Some(bot()) })
        .collect();
    Ok(TypedMatch::new(
        connect4::Connect4,
        bots,
        seat,
        o.get("seed", default_seed()),
    )
    .boxed())
}

fn make_go(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let size: usize = o.get("size", 9);
    let sims: u32 = o.get("sims", 6000);
    let seat = parse_seat(o, 2)?;
    let seed = o.get("seed", default_seed());
    let bots = (0..2)
        .map(|p| {
            if p == seat {
                None
            } else {
                Some(Box::new(Mcts::new(sims, seed ^ 0xC0)) as Box<dyn Agent<go::Go>>)
            }
        })
        .collect();
    Ok(TypedMatch::new(go::Go::new(size), bots, seat, seed).boxed())
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
    fn act(&self, game: &chess::Chess, state: &chess::Board, player: usize, r: f64) -> usize {
        PuctAgent(Puct::new(game, &ChessEnc, &self.net, self.sims)).act(game, state, player, r)
    }
}

fn make_chess(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 5);
    let seat = parse_seat(o, 2)?;
    let bot_kind = o.str("bot", "alphabeta");
    let bot = || -> Result<Box<dyn Agent<chess::Chess>>, String> {
        Ok(match bot_kind.as_str() {
            "alphabeta" => Box::new(AlphaBeta::new(depth, chess::MaterialEval, chess::ChessSpec)),
            "azero" => Box::new(AzeroBot {
                net: load_azero_net(&o.str("net", "data/azero/chess.bin"))?,
                sims: o.get("sims", 256),
            }),
            other => return Err(format!("unknown bot '{other}' (alphabeta|azero)")),
        })
    };
    let mut bots: Vec<Option<Box<dyn Agent<chess::Chess>>>> = Vec::new();
    for p in 0..2 {
        bots.push(if p == seat { None } else { Some(bot()?) });
    }
    Ok(TypedMatch::new(chess::Chess, bots, seat, o.get("seed", default_seed())).boxed())
}

fn make_liars_dice(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let players: u8 = o.get("players", 5);
    let game = LiarsDice::new(players, o.get("dice", 5), o.get("faces", 6));
    let rollouts: u32 = o.get("rollouts", 1000);
    let bot_kind = o.str("bot", "rollout");
    let seed = o.get("seed", default_seed());
    let bot = |p: usize| -> Result<Box<dyn Agent<LiarsDice>>, String> {
        Ok(match bot_kind.as_str() {
            "rollout" => Box::new(Rollout::new(
                rollouts,
                ProbabilisticAgent::default_agent(),
                BidConditioned::default(),
                seed ^ (p as u64) << 8,
            )),
            "belief" => Box::new(ProbabilisticAgent::default_agent()),
            "random" => Box::new(RandomAgent),
            other => return Err(format!("unknown bot '{other}'")),
        })
    };
    let seat = parse_seat(o, players as usize)?;
    let mut bots: Vec<Option<Box<dyn Agent<LiarsDice>>>> = Vec::new();
    for p in 0..players as usize {
        bots.push(if p == seat { None } else { Some(bot(p)?) });
    }
    Ok(TypedMatch::new(game, bots, seat, seed).boxed())
}

/// Plays the solved strategy greedily via the solver's draw probability.
struct SolverBot(std::sync::Arc<twentyone::Solver>);

impl Agent<TwentyOne> for SolverBot {
    fn act(&self, game: &TwentyOne, state: &T21State, player: usize, _r: f64) -> usize {
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
    let hearts: u8 = o.get("hearts", 6);
    let iters: u64 = o.get("iters", 50_000);
    let mut solver = if hearts <= 2 {
        twentyone::Solver::with_hearts(0xD1CE, hearts)
    } else {
        twentyone::Solver::abstracted(0xD1CE, hearts)
    };
    eprintln!("training the Twenty-One solver ({iters} iters/subgame)...");
    solver.solve(iters);
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
    Ok(TypedMatch::new(game, bots, seat, o.get("seed", default_seed())).boxed())
}

/// A game registered for bot-vs-bot evaluation: `compare` runs two specs
/// against each other (paired GSPRT for 2-player configurations, hero-vs-field
/// binomial SPRT otherwise); `tourney` runs a round-robin Elo table.
/// Non-printing pair runner: `(opts, spec_a, spec_b, seed, pair_range)` →
/// W-D-L from A's perspective. Used by external drivers (the web engine).
pub type PairsFn =
    fn(&Opts, &str, &str, u64, std::ops::Range<u64>) -> Result<(u64, u64, u64), String>;
/// Field runner for N-player configurations: hero A vs a field of B →
/// (hero wins, losses).
pub type FieldFn = fn(&Opts, &str, &str, u64, std::ops::Range<u64>) -> Result<(u64, u64), String>;

pub struct CompareEntry {
    pub id: &'static str,
    pub bots_help: &'static str,
    pub compare: fn(&CompareArgs) -> Result<(), String>,
    pub tourney: fn(&TourneyArgs) -> Result<(), String>,
    pub pairs: PairsFn,
    pub field: Option<FieldFn>,
}

pub fn compare_entries() -> Vec<CompareEntry> {
    vec![
        CompareEntry {
            id: "chess",
            bots_help: "alphabeta[:depth=5] | alphabeta-rich[:depth=5] (rich eval) | \
                        azero[:net=data/azero/chess.bin,sims=256]",
            compare: |a| head_to_head(&chess::Chess, a, 6, chess_bot),
            tourney: |a| round_robin(&chess::Chess, a, 6, chess_bot),
            pairs: |o, a, b, s, r| run_pairs(&chess::Chess, o, a, b, 6, chess_bot, s, r),
            field: None,
        },
        CompareEntry {
            id: "othello",
            bots_help: "alphabeta[:depth=6] | mcts[:sims=2000]",
            compare: |a| head_to_head(&othello::Othello, a, 4, othello_bot),
            tourney: |a| round_robin(&othello::Othello, a, 4, othello_bot),
            pairs: |o, a, b, s, r| run_pairs(&othello::Othello, o, a, b, 4, othello_bot, s, r),
            field: None,
        },
        CompareEntry {
            id: "connect4",
            bots_help: "alphabeta[:depth=9] | mcts[:sims=2000]",
            compare: |a| head_to_head(&connect4::Connect4, a, 4, connect4_bot),
            tourney: |a| round_robin(&connect4::Connect4, a, 4, connect4_bot),
            pairs: |o, a, b, s, r| run_pairs(&connect4::Connect4, o, a, b, 4, connect4_bot, s, r),
            field: None,
        },
        CompareEntry {
            id: "go",
            bots_help: "mcts[:sims=2000] | mcts-eval[:sims=2000,depth=NxN] | mcts-spec[:sims=2000]",
            compare: |a| head_to_head(&go::Go::new(a.opts.get("size", 9)), a, 0, go_bot),
            tourney: |a| round_robin(&go::Go::new(a.opts.get("size", 9)), a, 0, go_bot),
            pairs: |o, a, b, s, r| {
                run_pairs(&go::Go::new(o.get("size", 9)), o, a, b, 0, go_bot, s, r)
            },
            field: None,
        },
        CompareEntry {
            id: "liars-dice",
            bots_help: "rollout[:rollouts=1000] | belief | random",
            compare: |a| {
                let game = liars_dice_game(&a.opts);
                use game_core::Game;
                if game.num_players() == 2 {
                    head_to_head(&game, a, 0, liars_dice_bot)
                } else {
                    vs_field(&game, a, liars_dice_bot)
                }
            },
            tourney: |a| round_robin(&liars_dice_game(&a.opts), a, 0, liars_dice_bot),
            pairs: |o, a, b, s, r| run_pairs(&liars_dice_game(o), o, a, b, 0, liars_dice_bot, s, r),
            field: Some(|o, a, b, s, r| {
                run_field(&liars_dice_game(o), o, a, b, liars_dice_bot, s, r)
            }),
        },
    ]
}

fn liars_dice_game(o: &Opts) -> LiarsDice {
    LiarsDice::new(o.get("players", 5), o.get("dice", 5), o.get("faces", 6))
}

fn chess_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<chess::Chess>, String> {
    let depth: u32 = spec.opts.get("depth", 5);
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
            let sims: usize = spec.opts.get("sims", 256);
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
            let depth: u32 = spec.opts.get("depth", 6);
            Box::new(move |_| {
                Box::new(AlphaBeta::new(
                    depth,
                    othello::OthelloEval,
                    othello::OthelloSpec,
                )) as BoxedAgent<othello::Othello>
            })
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 2000);
            Box::new(move |seed| Box::new(Mcts::new(sims, seed)) as BoxedAgent<othello::Othello>)
        }
        other => return Err(format!("unknown othello bot '{other}' (alphabeta|mcts)")),
    })
}

fn connect4_bot(spec: &BotSpec, _o: &Opts) -> Result<BotBuilder<connect4::Connect4>, String> {
    Ok(match spec.name.as_str() {
        "alphabeta" => {
            let depth: u32 = spec.opts.get("depth", 9);
            Box::new(move |_| {
                Box::new(AlphaBeta::new(depth, connect4::Connect4Eval, NoSpec))
                    as BoxedAgent<connect4::Connect4>
            })
        }
        "mcts" => {
            let sims: u32 = spec.opts.get("sims", 2000);
            Box::new(move |seed| Box::new(Mcts::new(sims, seed)) as BoxedAgent<connect4::Connect4>)
        }
        other => return Err(format!("unknown connect4 bot '{other}' (alphabeta|mcts)")),
    })
}

fn go_bot(spec: &BotSpec, o: &Opts) -> Result<BotBuilder<go::Go>, String> {
    let sims: u32 = spec.opts.get("sims", 2000);
    let size: usize = o.get("size", 9);
    Ok(match spec.name.as_str() {
        "mcts" => Box::new(move |seed| Box::new(Mcts::new(sims, seed)) as BoxedAgent<go::Go>),
        "mcts-eval" => {
            let depth: u32 = spec.opts.get("depth", (size * size) as u32);
            Box::new(move |seed| {
                Box::new(Mcts::with_eval(sims, go::GoEval, depth, seed)) as BoxedAgent<go::Go>
            })
        }
        "mcts-spec" => Box::new(move |seed| {
            Box::new(Mcts::with_spec(sims, go::GoSpec, seed)) as BoxedAgent<go::Go>
        }),
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
            let rollouts: u32 = spec.opts.get("rollouts", 1000);
            Box::new(move |seed| {
                Box::new(Rollout::new(
                    rollouts,
                    ProbabilisticAgent::default_agent(),
                    BidConditioned::default(),
                    seed,
                )) as BoxedAgent<LiarsDice>
            })
        }
        "belief" => {
            Box::new(|_| Box::new(ProbabilisticAgent::default_agent()) as BoxedAgent<LiarsDice>)
        }
        "random" => Box::new(|_| Box::new(RandomAgent) as BoxedAgent<LiarsDice>),
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
