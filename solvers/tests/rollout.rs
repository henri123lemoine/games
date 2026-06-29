//! Determinized-rollout behavior: the agent maximizes *expected return*, not
//! win frequency — margins and draws count — and is deterministic given the
//! arena's rng.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use game_core::{Agent, Determinizer, Game, Identity, RandomAgent, Rng, Turn};
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

#[derive(Clone, Copy, PartialEq)]
enum IndexedState {
    Pick,
    Done(i8),
}

struct IndexedOnly;

impl Game for IndexedOnly {
    type State = IndexedState;
    type Action = u8;

    fn initial_state(&self) -> IndexedState {
        IndexedState::Pick
    }

    fn turn(&self, _s: &IndexedState) -> Turn {
        Turn::Player(0)
    }

    fn is_terminal(&self, s: &IndexedState) -> bool {
        matches!(s, IndexedState::Done(_))
    }

    fn returns(&self, s: &IndexedState, player: usize) -> f64 {
        let IndexedState::Done(v) = s else {
            unreachable!()
        };
        let v = f64::from(*v);
        if player == 0 { v } else { -v }
    }

    fn legal_actions(&self, _s: &IndexedState) -> Vec<u8> {
        panic!("rollout should use num_actions/action_at for root candidates")
    }

    fn num_actions(&self, _s: &IndexedState) -> usize {
        2
    }

    fn action_at(&self, _s: &IndexedState, i: usize) -> u8 {
        assert!(i < 2);
        i as u8
    }

    fn chance_outcomes(&self, _s: &IndexedState) -> Vec<(u8, f64)> {
        Vec::new()
    }

    fn apply(&self, s: &mut IndexedState, a: u8) {
        *s = match (*s, a) {
            (IndexedState::Pick, 0) => IndexedState::Done(-1),
            (IndexedState::Pick, 1) => IndexedState::Done(1),
            (IndexedState::Done(_), _) => unreachable!("apply on terminal"),
            (_, _) => unreachable!("invalid action"),
        }
    }

    fn infoset_key(&self, s: &IndexedState, _player: usize) -> u64 {
        match s {
            IndexedState::Pick => 0,
            IndexedState::Done(v) => 100 + (*v as i64 + 2) as u64,
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
fn rollout_uses_indexed_action_lookup_for_root_candidates() {
    let game = IndexedOnly;
    let rollout = Rollout::new(10, RandomAgent, Identity);
    let mut rng = Rng::new(9);
    let i = rollout.act(&game, &IndexedState::Pick, 0, &mut rng);
    assert_eq!(i, 1);
}

#[test]
fn deterministic_given_the_arena_rng() {
    let game = Stakes;
    let rollout = Rollout::new(200, RandomAgent, Identity);
    let a = rollout.act(&game, &S::Pick, 0, &mut Rng::new(7));
    let b = rollout.act(&game, &S::Pick, 0, &mut Rng::new(7));
    assert_eq!(a, b);
}

struct CountingDet(Arc<AtomicUsize>);

impl Determinizer<Stakes> for CountingDet {
    fn determinize(&self, _game: &Stakes, _state: &mut S, _observer: usize, _rng: &mut Rng) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn shared_worlds_are_determinized_once_per_rollout() {
    let game = Stakes;
    let calls = Arc::new(AtomicUsize::new(0));
    let rollouts = 37;
    let rollout = Rollout::new(rollouts, RandomAgent, CountingDet(Arc::clone(&calls)));
    let _ = rollout.act(&game, &S::Pick, 0, &mut Rng::new(11));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        rollouts as usize,
        "common-random rollout worlds should be shared across candidate actions"
    );
}
