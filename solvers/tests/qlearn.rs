//! Tabular Q-learning / SARSA correctness: two-player zero-sum self-play on
//! tic-tac-toe produces near-perfect greedy play against random, and a
//! single-player chain MDP with chance converges to the known optimal value.

use game_core::{Agent, Game, RandomAgent, Rng, Turn};

mod common;
use common::Ttt;
use solvers::qlearn::{QConfig, QLearner};

fn play_pair(game: &Ttt, agents: [&dyn Agent<Ttt>; 2], rng: &mut Rng) -> f64 {
    let mut s = game.initial_state();
    while !game.is_terminal(&s) {
        let Turn::Player(p) = game.turn(&s) else {
            unreachable!("tic-tac-toe has no chance nodes")
        };
        let actions = game.legal_actions(&s);
        let i = agents[p].act(game, &s, p, rng);
        game.apply(&mut s, actions[i]);
    }
    game.returns(&s, 0)
}

/// Wins + half-draws per game for the trained greedy policy against uniform
/// random, seats alternating.
fn score_vs_random(learner: &QLearner<Ttt>, games: u32, seed: u64) -> f64 {
    let game = Ttt;
    let greedy = learner.greedy();
    let mut rng = Rng::new(seed);
    let mut score = 0.0;
    for g in 0..games {
        let hero_seat = (g % 2) as usize;
        let agents: [&dyn Agent<Ttt>; 2] = if hero_seat == 0 {
            [&greedy, &RandomAgent]
        } else {
            [&RandomAgent, &greedy]
        };
        let r0 = play_pair(&game, agents, &mut rng);
        let hero = if hero_seat == 0 { r0 } else { -r0 };
        score += if hero > 0.0 {
            1.0
        } else if hero < 0.0 {
            0.0
        } else {
            0.5
        };
    }
    score / games as f64
}

const TTT_NONTERMINAL_STATES: usize = 4520;

#[test]
fn ttt_selfplay_greedy_beats_random() {
    let cfg = QConfig {
        epsilon_end: 0.2,
        decay_episodes: 150_000,
        ..QConfig::default()
    };
    let mut learner = QLearner::new(Ttt, cfg, 1);
    learner.train_episodes(200_000);

    let score = score_vs_random(&learner, 200, 99);
    eprintln!("ttt greedy score vs random: {score}");
    assert!(score >= 0.95, "greedy Q scored only {score} vs random");

    let size = learner.table_size();
    eprintln!("ttt table size: {size}");
    assert!(
        (2_000..=TTT_NONTERMINAL_STATES).contains(&size),
        "table size {size} outside (2000, {TTT_NONTERMINAL_STATES})"
    );
}

/// A stop-or-risk chain: at positions 0..3 the player may stop (reward 0.25)
/// or continue through a chance node that advances with probability 0.8 and
/// busts (reward 0) otherwise; reaching position 3 pays 1.0. With gamma = 0.9
/// the optimal policy always continues and V(start) = (0.8 * 0.9)^3 = 0.373248.
#[derive(Clone, Copy)]
enum Chain {
    Decide(u8),
    Risk(u8),
    Done(f64),
}

struct ChainMdp;

const CHAIN_LEN: u8 = 3;
const P_ADVANCE: f64 = 0.8;
const STOP_REWARD: f64 = 0.25;
const FINAL_REWARD: f64 = 1.0;

impl Game for ChainMdp {
    type State = Chain;
    type Action = u8;

    fn num_players(&self) -> usize {
        1
    }

    fn initial_state(&self) -> Chain {
        Chain::Decide(0)
    }

    fn turn(&self, s: &Chain) -> Turn {
        match s {
            Chain::Risk(_) => Turn::Chance,
            _ => Turn::Player(0),
        }
    }

    fn is_terminal(&self, s: &Chain) -> bool {
        matches!(s, Chain::Done(_))
    }

    fn returns(&self, s: &Chain, _player: usize) -> f64 {
        match s {
            Chain::Done(r) => *r,
            _ => unreachable!("returns on non-terminal"),
        }
    }

    fn legal_actions(&self, s: &Chain) -> Vec<u8> {
        match s {
            Chain::Decide(pos) if *pos < CHAIN_LEN => vec![0, 1],
            Chain::Decide(_) => vec![0],
            _ => Vec::new(),
        }
    }

    fn chance_outcomes(&self, _s: &Chain) -> Vec<(u8, f64)> {
        vec![(0, P_ADVANCE), (1, 1.0 - P_ADVANCE)]
    }

    fn apply(&self, s: &mut Chain, a: u8) {
        *s = match (*s, a) {
            (Chain::Decide(pos), 0) => Chain::Done(if pos == CHAIN_LEN {
                FINAL_REWARD
            } else {
                STOP_REWARD
            }),
            (Chain::Decide(pos), _) => Chain::Risk(pos),
            (Chain::Risk(pos), 0) => Chain::Decide(pos + 1),
            (Chain::Risk(_), _) => Chain::Done(0.0),
            (Chain::Done(_), _) => unreachable!("apply on terminal"),
        };
    }

    fn infoset_key(&self, s: &Chain, _player: usize) -> u64 {
        match s {
            Chain::Decide(pos) => *pos as u64,
            Chain::Risk(pos) => 100 + *pos as u64,
            Chain::Done(_) => u64::MAX,
        }
    }
}

fn chain_optimal_start_value() -> f64 {
    let mut v = FINAL_REWARD;
    for _ in 0..CHAIN_LEN {
        v *= P_ADVANCE * 0.9;
    }
    v
}

fn chain_config(sarsa: bool) -> QConfig {
    QConfig {
        gamma: 0.9,
        alpha_end: 0.005,
        epsilon_end: 0.01,
        decay_episodes: 60_000,
        sarsa,
        ..QConfig::default()
    }
}

fn assert_chain_converges(sarsa: bool, seed: u64) {
    let mut learner = QLearner::new(ChainMdp, chain_config(sarsa), seed);
    learner.train_episodes(80_000);

    assert_eq!(
        learner.table_size(),
        CHAIN_LEN as usize + 1,
        "one row per decision position"
    );

    let greedy = learner.greedy();
    for pos in 0..CHAIN_LEN {
        let i = greedy.act(&ChainMdp, &Chain::Decide(pos), 0, &mut Rng::new(1));
        assert_eq!(i, 1, "greedy stops at position {pos} instead of continuing");
    }

    let optimal = chain_optimal_start_value();
    let row = learner.q_values(0, 0).expect("start infoset visited");
    let learned = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("sarsa={sarsa}: learned start value {learned} (optimal {optimal})");
    assert!(
        (learned - optimal).abs() < 0.05,
        "sarsa={sarsa}: learned start value {learned} vs optimal {optimal}"
    );
}

#[test]
fn chain_mdp_qlearning_converges_to_optimal_value() {
    assert_chain_converges(false, 7);
}

#[test]
fn chain_mdp_sarsa_converges_to_optimal_value() {
    assert_chain_converges(true, 11);
}
