//! MCTS variant coverage on inline toy games: chance-node valuation (with and
//! without solver backup), prior-guided PUCT selection, and transposition
//! merging.

use game_core::{Agent, Game, Rng, SearchSpec, Turn};
use solvers::mcts::Mcts;

/// Decision-then-chance toy: SAFE banks a guaranteed 0.2, GAMBLE flips a
/// `p_win` coin worth +1/-1 (EV `2·p_win − 1`). Player 1 never moves.
#[derive(Clone, Copy, PartialEq, Debug)]
enum GState {
    Pick,
    Flip,
    Safe,
    Won,
    Lost,
}

struct Gamble {
    p_win: f64,
}

impl Game for Gamble {
    type State = GState;
    type Action = u8;

    fn initial_state(&self) -> GState {
        GState::Pick
    }

    fn turn(&self, s: &GState) -> Turn {
        match s {
            GState::Flip => Turn::Chance,
            _ => Turn::Player(0),
        }
    }

    fn is_terminal(&self, s: &GState) -> bool {
        matches!(s, GState::Safe | GState::Won | GState::Lost)
    }

    fn returns(&self, s: &GState, player: usize) -> f64 {
        let v0 = match s {
            GState::Safe => 0.2,
            GState::Won => 1.0,
            GState::Lost => -1.0,
            _ => 0.0,
        };
        if player == 0 { v0 } else { -v0 }
    }

    fn legal_actions(&self, _s: &GState) -> Vec<u8> {
        vec![0, 1]
    }

    fn chance_outcomes(&self, _s: &GState) -> Vec<(u8, f64)> {
        vec![(0, self.p_win), (1, 1.0 - self.p_win)]
    }

    fn apply(&self, s: &mut GState, a: u8) {
        *s = match (*s, a) {
            (GState::Pick, 0) => GState::Safe,
            (GState::Pick, _) => GState::Flip,
            (GState::Flip, 0) => GState::Won,
            _ => GState::Lost,
        };
    }

    fn infoset_key(&self, s: &GState, _player: usize) -> u64 {
        *s as u64
    }
}

#[test]
fn chance_nodes_valued_correctly() {
    for (p_win, want) in [(0.75, 1u8), (0.25, 0u8)] {
        let game = Gamble { p_win };
        let actions = game.legal_actions(&GState::Pick);
        for seed in 1..=3 {
            let mcts: Mcts<Gamble> = Mcts::new(2000);
            let i = mcts.act(&game, &GState::Pick, 0, &mut Rng::new(seed));
            assert_eq!(actions[i], want, "solver-on p_win {p_win} seed {seed}");

            let mut plain: Mcts<Gamble> = Mcts::new(2000);
            plain.solver = false;
            let i = plain.act(&game, &GState::Pick, 0, &mut Rng::new(seed));
            assert_eq!(actions[i], want, "solver-off p_win {p_win} seed {seed}");
        }
    }
}

/// Five identical-value arms; only a prior can break the symmetry.
struct Bandit;

impl Game for Bandit {
    type State = Option<u8>;
    type Action = u8;

    fn initial_state(&self) -> Option<u8> {
        None
    }

    fn turn(&self, _s: &Option<u8>) -> Turn {
        Turn::Player(0)
    }

    fn is_terminal(&self, s: &Option<u8>) -> bool {
        s.is_some()
    }

    fn returns(&self, _s: &Option<u8>, _player: usize) -> f64 {
        0.0
    }

    fn legal_actions(&self, _s: &Option<u8>) -> Vec<u8> {
        (0..5).collect()
    }

    fn chance_outcomes(&self, _s: &Option<u8>) -> Vec<(u8, f64)> {
        Vec::new()
    }

    fn apply(&self, s: &mut Option<u8>, a: u8) {
        *s = Some(a);
    }

    fn infoset_key(&self, s: &Option<u8>, _player: usize) -> u64 {
        s.map_or(9, u64::from)
    }
}

struct HintThird;
impl SearchSpec<Bandit> for HintThird {
    fn order_hint(&self, _game: &Bandit, _state: &Option<u8>, a: u8) -> i64 {
        if a == 3 { 100 } else { 0 }
    }
}

#[test]
fn prior_spec_concentrates_root_visits_on_hinted_move() {
    let game = Bandit;
    let mut mcts = Mcts::with_spec(500, HintThird);
    mcts.solver = false;
    let visits = mcts.root_visits(&game, &None, 0, &mut Rng::new(9));
    let total: u32 = visits.iter().sum();
    assert!(
        visits[3] * 2 > total,
        "hinted arm got {} of {total} visits: {visits:?}",
        visits[3]
    );
    assert_eq!(mcts.act(&game, &None, 0, &mut Rng::new(10)), 3);
}

/// Subtraction Nim: take 1–3 from the pile, taking the last stone wins. The
/// state graph transposes heavily (different move orders, same pile).
#[derive(Clone)]
struct NimState {
    left: u8,
    to_move: usize,
}

struct Nim;

impl Game for Nim {
    type State = NimState;
    type Action = u8;

    fn initial_state(&self) -> NimState {
        NimState {
            left: 9,
            to_move: 0,
        }
    }

    fn turn(&self, s: &NimState) -> Turn {
        Turn::Player(s.to_move)
    }

    fn is_terminal(&self, s: &NimState) -> bool {
        s.left == 0
    }

    fn returns(&self, s: &NimState, player: usize) -> f64 {
        if player == s.to_move { -1.0 } else { 1.0 }
    }

    fn legal_actions(&self, s: &NimState) -> Vec<u8> {
        (1..=s.left.min(3)).collect()
    }

    fn chance_outcomes(&self, _s: &NimState) -> Vec<(u8, f64)> {
        Vec::new()
    }

    fn apply(&self, s: &mut NimState, a: u8) {
        s.left -= a;
        s.to_move ^= 1;
    }

    fn infoset_key(&self, s: &NimState, _player: usize) -> u64 {
        u64::from(s.left) * 2 + s.to_move as u64
    }

    fn state_key(&self, s: &NimState) -> Option<u64> {
        Some(u64::from(s.left) * 2 + s.to_move as u64)
    }
}

#[test]
fn transposition_merging_stays_correct() {
    let game = Nim;
    for start in [5u8, 9, 13] {
        let s = NimState {
            left: start,
            to_move: 0,
        };
        let actions = game.legal_actions(&s);
        let want = start % 4;
        for seed in [2u64, 8] {
            let mut mcts: Mcts<Nim> = Mcts::new(2000);
            mcts.transpositions = true;
            let i = mcts.act(&game, &s, 0, &mut Rng::new(seed));
            assert_eq!(actions[i], want, "start {start} seed {seed}");
        }
    }
}
