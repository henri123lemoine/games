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

pub fn entries() -> Vec<Entry> {
    vec![
        Entry {
            id: "chess",
            summary: "chess vs alpha-beta (perft-validated rules)",
            opts_help: "depth=5  seat=0 (0=White)  bot=alphabeta|azero  net=data/azero/chess.bin  sims=256  seed=...",
            make: make_chess,
        },
        Entry {
            id: "liars-dice",
            summary: "N-player Liar's Dice vs determinized-rollout bots",
            opts_help: "players=5 dice=5 faces=6 rollouts=1000 bot=rollout|belief|random seed=...",
            make: make_liars_dice,
        },
        Entry {
            id: "twentyone",
            summary: "Twenty-One vs the decomposed CFR+ solver (trains at startup)",
            opts_help: "hearts=6 iters=50000 (training iters/subgame)  seed=...",
            make: make_twentyone,
        },
        Entry {
            id: "othello",
            summary: "Othello vs alpha-beta (weighted squares + mobility)",
            opts_help: "depth=6  seat=0|1 (0=Black)  seed=...",
            make: make_othello,
        },
        Entry {
            id: "connect4",
            summary: "Connect-4 vs alpha-beta",
            opts_help: "depth=9  seat=0|1  seed=...",
            make: make_connect4,
        },
        Entry {
            id: "go",
            summary: "Go (area scoring, komi 7.5) vs MCTS",
            opts_help: "size=9  sims=6000  seat=0|1 (0=Black)  seed=...",
            make: make_go,
        },
    ]
}

fn make_othello(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 6);
    let seat: usize = o.get("seat", 0);
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
    let seat: usize = o.get("seat", 0);
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
    let seat: usize = o.get("seat", 0);
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

/// Owns the net and runs a fresh PUCT search per move.
struct AzeroBot {
    net: Mlp,
    sims: usize,
}

impl Agent<chess::Chess> for AzeroBot {
    fn act(&self, game: &chess::Chess, state: &chess::Board, player: usize, r: f64) -> usize {
        PuctAgent(Puct::new(game, &ChessEnc, &self.net, self.sims)).act(game, state, player, r)
    }
}

fn make_chess(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 5);
    let seat: usize = o.get("seat", 0);
    if seat > 1 {
        return Err("seat must be 0 or 1".into());
    }
    let bot_kind = o.str("bot", "alphabeta");
    let bot = || -> Result<Box<dyn Agent<chess::Chess>>, String> {
        Ok(match bot_kind.as_str() {
            "alphabeta" => Box::new(AlphaBeta::new(depth, chess::MaterialEval, chess::ChessSpec)),
            "azero" => {
                let path = o.str("net", "data/azero/chess.bin");
                let net = Mlp::load(std::path::Path::new(&path))
                    .map_err(|e| format!("failed to load azero net '{path}': {e}"))?;
                Box::new(AzeroBot {
                    net,
                    sims: o.get("sims", 256),
                })
            }
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
    let mut bots: Vec<Option<Box<dyn Agent<LiarsDice>>>> = vec![None];
    for p in 1..players as usize {
        bots.push(Some(bot(p)?));
    }
    Ok(TypedMatch::new(game, bots, 0, seed).boxed())
}

/// Plays the solved strategy greedily via the solver's draw probability.
struct SolverBot(twentyone::Solver);

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
    let game = TwentyOne::new(hearts);
    let bots: Vec<Option<Box<dyn Agent<TwentyOne>>>> =
        vec![None, Some(Box::new(SolverBot(solver)))];
    Ok(TypedMatch::new(game, bots, 0, o.get("seed", default_seed())).boxed())
}

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64
        | 1
}
