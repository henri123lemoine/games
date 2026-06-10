//! The game/bot registry: every playable game registers an id, its options,
//! and how to build a match against its bots. This is the single integration
//! point a future web service reuses — it serves whatever is registered here.

use std::collections::HashMap;

use game_core::Agent;
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent, RandomAgent};
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
            opts_help: "depth=5 (search depth)  seat=0 (0=White, 1=Black)  seed=...",
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
    ]
}

fn make_chess(o: &Opts) -> Result<Box<dyn AnyMatch>, String> {
    let depth: u32 = o.get("depth", 5);
    let seat: usize = o.get("seat", 0);
    if seat > 1 {
        return Err("seat must be 0 or 1".into());
    }
    let bot = || -> Box<dyn Agent<chess::Chess>> {
        Box::new(AlphaBeta::new(depth, chess::MaterialEval, chess::ChessSpec))
    };
    let bots = (0..2)
        .map(|p| if p == seat { None } else { Some(bot()) })
        .collect();
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
