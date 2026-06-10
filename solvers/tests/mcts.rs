//! UCT MCTS correctness on tic-tac-toe: never loses to random, wins most
//! games outright, and blocks immediate threats (with and without an eval
//! cutoff).

use game_core::{Agent, Eval, Game, Rng, Turn};
use solvers::mcts::Mcts;

#[derive(Clone)]
struct Board {
    cells: [u8; 9],
    to_move: usize,
}

struct Ttt;

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

fn winner(b: &Board) -> Option<usize> {
    LINES.iter().find_map(|l| {
        let v = b.cells[l[0]];
        (v != 0 && b.cells[l[1]] == v && b.cells[l[2]] == v).then(|| (v - 1) as usize)
    })
}

impl Game for Ttt {
    type State = Board;
    type Action = usize;

    fn initial_state(&self) -> Board {
        Board {
            cells: [0; 9],
            to_move: 0,
        }
    }

    fn turn(&self, s: &Board) -> Turn {
        Turn::Player(s.to_move)
    }

    fn is_terminal(&self, s: &Board) -> bool {
        winner(s).is_some() || s.cells.iter().all(|&c| c != 0)
    }

    fn returns(&self, s: &Board, player: usize) -> f64 {
        match winner(s) {
            Some(w) if w == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    fn legal_actions(&self, s: &Board) -> Vec<usize> {
        (0..9).filter(|&i| s.cells[i] == 0).collect()
    }

    fn chance_outcomes(&self, _s: &Board) -> Vec<(usize, f64)> {
        Vec::new()
    }

    fn apply(&self, s: &mut Board, a: usize) {
        s.cells[a] = s.to_move as u8 + 1;
        s.to_move ^= 1;
    }

    fn infoset_key(&self, s: &Board, _player: usize) -> u64 {
        s.cells
            .iter()
            .fold(s.to_move as u64, |k, &c| k * 4 + c as u64)
    }
}

fn random_agent(game: &Ttt, state: &Board, _p: usize, r: f64) -> usize {
    let n = game.legal_actions(state).len();
    ((r * n as f64) as usize).min(n - 1)
}

fn play_pair(game: &Ttt, agents: [&dyn Agent<Ttt>; 2], rng: &mut Rng) -> f64 {
    let mut s = game.initial_state();
    while !game.is_terminal(&s) {
        let Turn::Player(p) = game.turn(&s) else {
            unreachable!("tic-tac-toe has no chance nodes")
        };
        let actions = game.legal_actions(&s);
        let i = agents[p].act(game, &s, p, rng.unit());
        game.apply(&mut s, actions[i]);
    }
    game.returns(&s, 0)
}

#[test]
fn never_loses_to_random_and_mostly_wins() {
    let game = Ttt;
    let mcts: Mcts<Ttt> = Mcts::new(2000, 42);
    let mut rng = Rng::new(7);
    let (mut wins, mut losses) = (0u32, 0u32);
    for g in 0..50 {
        let mcts_seat = g % 2;
        let agents: [&dyn Agent<Ttt>; 2] = if mcts_seat == 0 {
            [&mcts, &random_agent]
        } else {
            [&random_agent, &mcts]
        };
        let r0 = play_pair(&game, agents, &mut rng);
        let m = if mcts_seat == 0 { r0 } else { -r0 };
        if m > 0.0 {
            wins += 1;
        } else if m < 0.0 {
            losses += 1;
        }
    }
    assert_eq!(losses, 0, "mcts lost {losses}/50 games to random");
    assert!(wins >= 40, "mcts won only {wins}/50 games against random");
}

/// X . .
/// X O .      O to move; X threatens 0-3-6. O must play cell 6.
/// . . .
fn threat_position() -> Board {
    Board {
        cells: [1, 0, 0, 1, 2, 0, 0, 0, 0],
        to_move: 1,
    }
}

#[test]
fn blocks_immediate_losing_threat() {
    let game = Ttt;
    let s = threat_position();
    let actions = game.legal_actions(&s);
    for seed in 1..=5 {
        let mcts: Mcts<Ttt> = Mcts::new(2000, seed);
        let i = mcts.act(&game, &s, 1, 0.5);
        assert_eq!(
            actions[i], 6,
            "seed {seed}: played {} instead of 6",
            actions[i]
        );
    }
}

struct ZeroEval;
impl Eval<Ttt> for ZeroEval {
    fn eval(&self, _game: &Ttt, _state: &Board, _player: usize) -> f64 {
        0.0
    }
}

#[test]
fn eval_cutoff_still_blocks_immediate_threat() {
    let game = Ttt;
    let s = threat_position();
    let actions = game.legal_actions(&s);
    let mcts = Mcts::with_eval(2000, ZeroEval, 2, 9);
    let i = mcts.act(&game, &s, 1, 0.5);
    assert_eq!(actions[i], 6, "played {} instead of 6", actions[i]);
}
