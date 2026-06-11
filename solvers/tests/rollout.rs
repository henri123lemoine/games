//! Determinized-rollout behavior: the agent maximizes *expected return*, not
//! win frequency — margins and draws count — and is deterministic given the
//! arena's rng.

use game_core::{Agent, Game, Identity, RandomAgent, Rng, Turn};
use solvers::Rollout;

/// One decision, then one chance step. Action 0 wins 60% of the time but the
/// stakes are terrible (+0.1 / -1.0, EV -0.34); action 1 is a fair ±1 coin
/// (EV 0). A win-frequency maximizer takes action 0; an expected-return
/// maximizer takes action 1.
#[derive(Clone, Copy, PartialEq)]
enum S {
    Pick,
    Flip(u8),
    Done(i8), // payoff to player 0, scaled by 10
}

struct Stakes;

impl Game for Stakes {
    type State = S;
    type Action = u8;

    fn initial_state(&self) -> S {
        S::Pick
    }

    fn turn(&self, s: &S) -> Turn {
        match s {
            S::Flip(_) => Turn::Chance,
            _ => Turn::Player(0),
        }
    }

    fn is_terminal(&self, s: &S) -> bool {
        matches!(s, S::Done(_))
    }

    fn returns(&self, s: &S, player: usize) -> f64 {
        let S::Done(v) = s else { unreachable!() };
        let v0 = f64::from(*v) / 10.0;
        if player == 0 { v0 } else { -v0 }
    }

    fn legal_actions(&self, _s: &S) -> Vec<u8> {
        vec![0, 1]
    }

    fn chance_outcomes(&self, s: &S) -> Vec<(u8, f64)> {
        match s {
            S::Flip(0) => vec![(0, 0.6), (1, 0.4)],
            _ => vec![(0, 0.5), (1, 0.5)],
        }
    }

    fn apply(&self, s: &mut S, a: u8) {
        *s = match (*s, a) {
            (S::Pick, k) => S::Flip(k),
            (S::Flip(0), 0) => S::Done(1),   // +0.1: a narrow win
            (S::Flip(0), _) => S::Done(-10), // -1.0: a blowout loss
            (S::Flip(_), 0) => S::Done(10),
            (S::Flip(_), _) => S::Done(-10),
            (S::Done(_), _) => unreachable!("apply on terminal"),
        };
    }

    fn infoset_key(&self, s: &S, _player: usize) -> u64 {
        match s {
            S::Pick => 0,
            S::Flip(k) => 1 + u64::from(*k),
            S::Done(v) => 100 + (*v as i64 + 50) as u64,
        }
    }
}

#[test]
fn maximizes_expected_return_not_win_rate() {
    let game = Stakes;
    let rollout = Rollout::new(4000, RandomAgent, Identity);
    let mut rng = Rng::new(42);
    let i = rollout.act(&game, &S::Pick, 0, &mut rng);
    assert_eq!(
        game.legal_actions(&S::Pick)[i],
        1,
        "EV maximization must prefer the fair coin over frequent narrow wins"
    );
}

#[test]
fn deterministic_given_the_arena_rng() {
    let game = Stakes;
    let rollout = Rollout::new(200, RandomAgent, Identity);
    let a = rollout.act(&game, &S::Pick, 0, &mut Rng::new(7));
    let b = rollout.act(&game, &S::Pick, 0, &mut Rng::new(7));
    assert_eq!(a, b);
}
