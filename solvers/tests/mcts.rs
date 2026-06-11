//! UCT MCTS correctness on tic-tac-toe: never loses to random, wins most
//! games outright, and blocks immediate threats (with and without an eval
//! cutoff).

use game_core::{Agent, Eval, Game, RandomAgent, Rng, Turn};
use solvers::mcts::Mcts;

mod common;
use common::{Ttt, TttState as Board};

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

#[test]
fn never_loses_to_random_and_mostly_wins() {
    let game = Ttt;
    let mcts: Mcts<Ttt> = Mcts::new(2000);
    let mut rng = Rng::new(7);
    let (mut wins, mut losses) = (0u32, 0u32);
    for g in 0..50 {
        let mcts_seat = g % 2;
        let agents: [&dyn Agent<Ttt>; 2] = if mcts_seat == 0 {
            [&mcts, &RandomAgent]
        } else {
            [&RandomAgent, &mcts]
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
        let mcts: Mcts<Ttt> = Mcts::new(2000);
        let i = mcts.act(&game, &s, 1, &mut Rng::new(seed));
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
    let mcts = Mcts::with_eval(2000, ZeroEval, 2);
    let i = mcts.act(&game, &s, 1, &mut Rng::new(9));
    assert_eq!(actions[i], 6, "played {} instead of 6", actions[i]);
}

/// X O X
/// . . .      X to move. Anything but 4 loses on the spot (O completes 1-4-7);
/// . O .      4 blocks and forks 0-4-8 / 2-4-6 — a forced win two X-moves deep.
fn mate_in_two_position() -> Board {
    Board {
        cells: [1, 2, 1, 0, 0, 0, 0, 2, 0],
        to_move: 0,
    }
}

#[test]
fn solver_proves_mate_in_two_at_tiny_sims() {
    let game = Ttt;
    let s = mate_in_two_position();
    let actions = game.legal_actions(&s);
    for seed in 1..=5 {
        let mcts: Mcts<Ttt> = Mcts::new(100);
        let i = mcts.act(&game, &s, 0, &mut Rng::new(seed));
        assert_eq!(
            actions[i], 4,
            "seed {seed}: played {} instead of 4",
            actions[i]
        );
    }
}

/// . X .
/// X O O      X to move, no threats pending either way. Only 0 wins (fork
/// . . .      0-1-2 / 0-3-6, minimax-verified unique; 7 even loses).
fn quiet_fork_position() -> Board {
    Board {
        cells: [0, 1, 0, 1, 2, 2, 0, 0, 0],
        to_move: 0,
    }
}

#[test]
fn solver_finds_quiet_fork_win() {
    let game = Ttt;
    let s = quiet_fork_position();
    let actions = game.legal_actions(&s);
    for seed in 1..=5 {
        let mcts: Mcts<Ttt> = Mcts::new(200);
        let i = mcts.act(&game, &s, 0, &mut Rng::new(seed));
        assert_eq!(
            actions[i], 0,
            "seed {seed}: played {} instead of 0",
            actions[i]
        );
    }
}

#[test]
fn rave_enabled_still_blocks_immediate_threat() {
    let game = Ttt;
    let s = threat_position();
    let actions = game.legal_actions(&s);
    let mut mcts: Mcts<Ttt> = Mcts::new(2000);
    mcts.rave = true;
    let i = mcts.act(&game, &s, 1, &mut Rng::new(11));
    assert_eq!(actions[i], 6, "played {} instead of 6", actions[i]);
}

#[test]
fn solver_on_beats_solver_off_head_to_head() {
    let game = Ttt;
    let on: Mcts<Ttt> = Mcts::new(32);
    let mut off: Mcts<Ttt> = Mcts::new(32);
    off.solver = false;
    let mut rng = Rng::new(99);
    let (mut on_wins, mut off_wins, mut draws) = (0u32, 0u32, 0u32);
    for g in 0..100 {
        let on_seat = g % 2;
        let agents: [&dyn Agent<Ttt>; 2] = if on_seat == 0 {
            [&on, &off]
        } else {
            [&off, &on]
        };
        let r0 = play_pair(&game, agents, &mut rng);
        let r_on = if on_seat == 0 { r0 } else { -r0 };
        if r_on > 0.0 {
            on_wins += 1;
        } else if r_on < 0.0 {
            off_wins += 1;
        } else {
            draws += 1;
        }
    }
    println!("solver-on {on_wins} wins / {draws} draws / solver-off {off_wins} wins");
    assert!(
        on_wins > off_wins,
        "solver-on {on_wins} vs solver-off {off_wins} ({draws} draws)"
    );
}
