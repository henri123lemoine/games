//! REINFORCE tests on an inline tic-tac-toe: self-play training beats a
//! random player, with the mean return vs random improving across training.

use game_core::{Agent, Game, Rng, Turn, play, win_rate};
use solvers::azero::PolicyValueEncoder;
use solvers::pg::{PgConfig, Reinforce};

struct Ttt;

#[derive(Clone)]
struct TttState {
    cells: [u8; 9],
    to_move: usize,
}

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

fn winner(cells: &[u8; 9]) -> Option<usize> {
    LINES.iter().find_map(|l| {
        let v = cells[l[0]];
        (v != 0 && cells[l[1]] == v && cells[l[2]] == v).then(|| v as usize - 1)
    })
}

impl Game for Ttt {
    type State = TttState;
    type Action = usize;

    fn initial_state(&self) -> TttState {
        TttState {
            cells: [0; 9],
            to_move: 0,
        }
    }

    fn turn(&self, s: &TttState) -> Turn {
        Turn::Player(s.to_move)
    }

    fn is_terminal(&self, s: &TttState) -> bool {
        winner(&s.cells).is_some() || s.cells.iter().all(|&c| c != 0)
    }

    fn returns(&self, s: &TttState, p: usize) -> f64 {
        match winner(&s.cells) {
            Some(w) if w == p => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, s: &TttState) -> Vec<usize> {
        (0..9).filter(|&i| s.cells[i] == 0).collect()
    }

    fn chance_outcomes(&self, _s: &TttState) -> Vec<(usize, f64)> {
        vec![]
    }

    fn apply(&self, s: &mut TttState, a: usize) {
        s.cells[a] = s.to_move as u8 + 1;
        s.to_move ^= 1;
    }

    fn infoset_key(&self, s: &TttState, _p: usize) -> u64 {
        s.cells.iter().fold(0u64, |k, &c| k * 3 + u64::from(c))
    }
}

struct TttEnc;

impl PolicyValueEncoder<Ttt> for TttEnc {
    fn input_len(&self) -> usize {
        19
    }

    fn policy_len(&self) -> usize {
        9
    }

    fn encode_state(&self, _g: &Ttt, s: &TttState) -> Vec<f32> {
        let mut x = vec![0.0; 19];
        for (i, &c) in s.cells.iter().enumerate() {
            match c {
                1 => x[i] = 1.0,
                2 => x[9 + i] = 1.0,
                _ => {}
            }
        }
        x[18] = s.to_move as f32;
        x
    }

    fn action_index(&self, _g: &Ttt, _s: &TttState, a: usize) -> usize {
        a
    }
}

fn random_agent(g: &Ttt, s: &TttState, _p: usize, r: f64) -> usize {
    let n = g.legal_actions(s).len();
    ((r * n as f64) as usize).min(n - 1)
}

/// Mean return vs random over `games`, seats swapped to cancel X's edge.
fn mean_return_vs_random(agent: &impl Agent<Ttt>, games: u32, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for g in 0..games {
        total += if g % 2 == 0 {
            play(&Ttt, agent, &random_agent, &mut rng)
        } else {
            -play(&Ttt, &random_agent, agent, &mut rng)
        };
    }
    total / f64::from(games)
}

#[test]
fn reinforce_learns_tictactoe() {
    let mut tr = Reinforce::new(&Ttt, &TttEnc, PgConfig::default(), 42);

    let first = tr.train_episodes(3_000);
    let initial_return = mean_return_vs_random(&tr.agent(), 200, 99);
    eprintln!(
        "initial window: vs-random return {initial_return:.3} self-play return {:.3} entropy {:.3} mse {:.3}",
        first.mean_return, first.mean_entropy, first.value_mse
    );

    let mut last = first;
    for _ in 0..9 {
        last = tr.train_episodes(3_000);
    }
    let final_return = mean_return_vs_random(&tr.agent(), 200, 99);
    eprintln!(
        "final window:   vs-random return {final_return:.3} self-play return {:.3} entropy {:.3} mse {:.3}",
        last.mean_return, last.mean_entropy, last.value_mse
    );

    assert!(last.mean_return.is_finite() && last.value_mse.is_finite());
    assert!(
        final_return > initial_return,
        "mean return vs random did not improve: first {initial_return:.3} last {final_return:.3}"
    );

    let score = win_rate(&Ttt, &tr.greedy_agent(), &random_agent, 200, 7);
    eprintln!("greedy vs random over 200 games: {score:.3}");
    assert!(
        score >= 0.75,
        "greedy policy scored only {score:.3} vs random"
    );
}

#[test]
fn agents_return_legal_indices_untrained() {
    let tr = Reinforce::new(&Ttt, &TttEnc, PgConfig::default(), 5);
    let game = &Ttt;
    for greedy in [false, true] {
        let agent = if greedy {
            tr.greedy_agent()
        } else {
            tr.agent()
        };
        let mut s = game.initial_state();
        let mut rng = game_core::Rng::new(11);
        while !game.is_terminal(&s) {
            let actions = game.legal_actions(&s);
            let Turn::Player(p) = game.turn(&s) else {
                unreachable!()
            };
            let i = game_core::Agent::act(&agent, game, &s, p, rng.unit());
            assert!(i < actions.len(), "illegal index {i} of {}", actions.len());
            game.apply(&mut s, actions[i]);
        }
    }
}
