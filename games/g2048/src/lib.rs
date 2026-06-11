//! 2048 as a [`game_core::Game`].
//!
//! Single-player 4x4 sliding-tile game. Decision actions are the four shift
//! directions, legal only when they change the board. Tile spawns are chance
//! nodes — 90% a 2, 10% a 4, uniform over empty cells — and the initial state
//! starts in chance for the first two spawns. Terminal when no shift changes
//! the board.
//!
//! **Not zero-sum.** [`Game::returns`] for the lone player is the normalized
//! score `(ln(1+score) / ln(1+80000)).min(1.0)` — a monotone map of the raw
//! score into `[0, 1]` so single-player utilities live on the scale that
//! eval-truncated search expects. Two-player solvers (CFR, alpha-beta) do not
//! apply; MCTS and rollout methods do.

mod ui;

use game_core::{Eval, Game, Turn};

pub const SIZE: usize = 4;
const CELLS: usize = SIZE * SIZE;
const SCORE_CAP: f64 = 80_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

pub const DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2048Action {
    Shift(Dir),
    Spawn { cell: u8, four: bool },
}

/// Board cells hold exponents (`0` = empty, `k` = tile `2^k`), row-major from
/// the top-left. `pending_spawns` counts chance spawns still owed (2 at the
/// start of a game, 1 after every board-changing shift).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G2048State {
    cells: [u8; CELLS],
    score: u64,
    pending_spawns: u8,
}

impl G2048State {
    /// Cumulative score: every merge adds the value of the tile it creates.
    pub fn score(&self) -> u64 {
        self.score
    }

    /// Value of the largest tile on the board (`0` if the board is empty).
    pub fn max_tile(&self) -> u32 {
        match self.cells.iter().max() {
            Some(&0) | None => 0,
            Some(&e) => 1u32 << e,
        }
    }

    /// Tile value at `(row, col)`, `0` if empty. Row 0 is the top.
    pub fn tile(&self, row: usize, col: usize) -> u32 {
        match self.cells[row * SIZE + col] {
            0 => 0,
            e => 1u32 << e,
        }
    }

    /// Build a decision-node state from tile values (each `0` or a power of
    /// two ≥ 2) — handy for tests and position setup.
    pub fn from_tiles(tiles: [[u32; SIZE]; SIZE], score: u64) -> Self {
        let mut cells = [0u8; CELLS];
        for (r, row) in tiles.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                cells[r * SIZE + c] = match v {
                    0 => 0,
                    v => {
                        assert!(v.is_power_of_two() && v >= 2, "invalid tile {v}");
                        v.trailing_zeros() as u8
                    }
                };
            }
        }
        Self {
            cells,
            score,
            pending_spawns: 0,
        }
    }

    fn key(&self) -> u64 {
        self.cells
            .iter()
            .chain([self.pending_spawns].iter())
            .fold(0, |h, &b| game_core::hash::combine(h, b as u64))
    }
}

/// Cell indices of lane `lane` for a shift toward `dir`, front first.
fn lane_indices(dir: Dir, lane: usize) -> [usize; SIZE] {
    std::array::from_fn(|k| match dir {
        Dir::Left => lane * SIZE + k,
        Dir::Right => lane * SIZE + (SIZE - 1 - k),
        Dir::Up => k * SIZE + lane,
        Dir::Down => (SIZE - 1 - k) * SIZE + lane,
    })
}

/// Slide a line of exponents toward index 0, merging equal neighbours once
/// each (`[2,2,4,4] -> [4,8,_,_]`, never re-merging a freshly made tile).
/// Returns the new line and the score gained (sum of created tile values).
fn merge_line(line: [u8; SIZE]) -> ([u8; SIZE], u64) {
    let mut out = [0u8; SIZE];
    let mut gained = 0u64;
    let mut len = 0usize;
    let mut open: Option<u8> = None;
    for v in line.into_iter().filter(|&v| v != 0) {
        if open == Some(v) {
            out[len - 1] = v + 1;
            gained += 1u64 << (v + 1);
            open = None;
        } else {
            out[len] = v;
            len += 1;
            open = Some(v);
        }
    }
    (out, gained)
}

fn shifted(cells: &[u8; CELLS], dir: Dir) -> ([u8; CELLS], u64, bool) {
    let mut out = *cells;
    let mut gained = 0u64;
    let mut changed = false;
    for l in 0..SIZE {
        let idx = lane_indices(dir, l);
        let line = idx.map(|i| cells[i]);
        let (merged, pts) = merge_line(line);
        gained += pts;
        changed |= merged != line;
        for (k, &i) in idx.iter().enumerate() {
            out[i] = merged[k];
        }
    }
    (out, gained, changed)
}

pub struct G2048;

impl Game for G2048 {
    type State = G2048State;
    type Action = G2048Action;

    fn num_players(&self) -> usize {
        1
    }

    fn initial_state(&self) -> G2048State {
        G2048State {
            cells: [0; CELLS],
            score: 0,
            pending_spawns: 2,
        }
    }

    fn turn(&self, state: &G2048State) -> Turn {
        if state.pending_spawns > 0 {
            Turn::Chance
        } else {
            Turn::Player(0)
        }
    }

    fn is_terminal(&self, state: &G2048State) -> bool {
        state.pending_spawns == 0 && DIRS.iter().all(|&d| !shifted(&state.cells, d).2)
    }

    /// Normalized single-player utility in `[0, 1]` (see crate docs); **not**
    /// zero-sum.
    fn returns(&self, state: &G2048State, player: usize) -> f64 {
        debug_assert_eq!(player, 0);
        ((1.0 + state.score as f64).ln() / (1.0 + SCORE_CAP).ln()).min(1.0)
    }

    /// Shifts in the fixed order up, down, left, right, keeping only those
    /// that change the board.
    fn legal_actions(&self, state: &G2048State) -> Vec<G2048Action> {
        DIRS.iter()
            .filter(|&&d| shifted(&state.cells, d).2)
            .map(|&d| G2048Action::Shift(d))
            .collect()
    }

    fn chance_outcomes(&self, state: &G2048State) -> Vec<(G2048Action, f64)> {
        debug_assert!(state.pending_spawns > 0);
        let empties: Vec<u8> = (0..CELLS as u8)
            .filter(|&c| state.cells[c as usize] == 0)
            .collect();
        let n = empties.len() as f64;
        empties
            .into_iter()
            .flat_map(|cell| {
                [
                    (G2048Action::Spawn { cell, four: false }, 0.9 / n),
                    (G2048Action::Spawn { cell, four: true }, 0.1 / n),
                ]
            })
            .collect()
    }

    fn apply(&self, state: &mut G2048State, action: G2048Action) {
        match action {
            G2048Action::Shift(dir) => {
                let (cells, gained, changed) = shifted(&state.cells, dir);
                debug_assert!(changed, "no-op shift {dir:?} applied");
                state.cells = cells;
                state.score += gained;
                if changed {
                    state.pending_spawns = 1;
                }
            }
            G2048Action::Spawn { cell, four } => {
                debug_assert!(state.pending_spawns > 0, "spawn outside chance");
                debug_assert_eq!(state.cells[cell as usize], 0, "spawn on occupied cell");
                state.cells[cell as usize] = if four { 2 } else { 1 };
                state.pending_spawns -= 1;
            }
        }
    }

    fn infoset_key(&self, state: &G2048State, _player: usize) -> u64 {
        state.key()
    }

    fn state_key(&self, state: &G2048State) -> Option<u64> {
        Some(state.key())
    }

    fn action_id(&self, action: &G2048Action) -> u64 {
        match action {
            G2048Action::Shift(d) => *d as u64,
            G2048Action::Spawn { cell, four } => 4 + 2 * u64::from(*cell) + u64::from(*four),
        }
    }
}

/// Static evaluation on the same `[0, 1]` scale as [`Game::returns`]: the
/// normalized score plus small bonuses for empty cells, monotone rows and
/// columns, and keeping the maximum tile in a corner — the standard 2048
/// heuristics — capped at 1. Unlocks eval-truncated MCTS.
pub struct Heuristic2048;

impl Eval<G2048> for Heuristic2048 {
    fn eval(&self, game: &G2048, state: &G2048State, player: usize) -> f64 {
        let base = game.returns(state, player);
        let empty = state.cells.iter().filter(|&&c| c == 0).count() as f64 / CELLS as f64;
        let bonus = 0.10 * empty + 0.06 * monotonicity(&state.cells) + 0.04 * corner(&state.cells);
        (base + bonus).min(1.0)
    }
}

/// Mean over the 8 lines (4 rows + 4 columns) of how monotone each line's
/// exponents are: fraction of adjacent pairs ordered in the line's dominant
/// direction, so a fully sorted line scores 1.
fn monotonicity(cells: &[u8; CELLS]) -> f64 {
    let mut total = 0.0;
    for dir in [Dir::Left, Dir::Up] {
        for l in 0..SIZE {
            let line = lane_indices(dir, l).map(|i| cells[i]);
            let inc = (0..SIZE - 1).filter(|&k| line[k] <= line[k + 1]).count();
            let dec = (0..SIZE - 1).filter(|&k| line[k] >= line[k + 1]).count();
            total += inc.max(dec) as f64 / (SIZE - 1) as f64;
        }
    }
    total / (2 * SIZE) as f64
}

fn corner(cells: &[u8; CELLS]) -> f64 {
    let max = *cells.iter().max().unwrap();
    let corners = [0, SIZE - 1, CELLS - SIZE, CELLS - 1];
    if max > 0 && corners.iter().any(|&i| cells[i] == max) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Rng;

    fn shift(game: &G2048, s: &mut G2048State, dir: Dir) {
        game.apply(s, G2048Action::Shift(dir));
    }

    fn row(s: &G2048State, r: usize) -> [u32; SIZE] {
        std::array::from_fn(|c| s.tile(r, c))
    }

    #[test]
    fn merge_pairs_left() {
        let game = G2048;
        let mut s = G2048State::from_tiles([[2, 2, 4, 4], [0; 4], [0; 4], [0; 4]], 0);
        shift(&game, &mut s, Dir::Left);
        assert_eq!(row(&s, 0), [4, 8, 0, 0]);
        assert_eq!(s.score(), 12);
    }

    #[test]
    fn no_double_merge() {
        let game = G2048;
        let mut s = G2048State::from_tiles([[4, 4, 8, 0], [0; 4], [0; 4], [0; 4]], 0);
        shift(&game, &mut s, Dir::Left);
        assert_eq!(row(&s, 0), [8, 8, 0, 0]);
        assert_eq!(s.score(), 8);

        let mut s = G2048State::from_tiles([[2, 2, 2, 2], [0; 4], [0; 4], [0; 4]], 0);
        shift(&game, &mut s, Dir::Left);
        assert_eq!(row(&s, 0), [4, 4, 0, 0]);
        assert_eq!(s.score(), 8);
    }

    #[test]
    fn merge_works_in_all_directions() {
        let game = G2048;
        let tiles = [[2, 0, 0, 2], [2, 0, 0, 2], [0; 4], [0; 4]];

        let mut s = G2048State::from_tiles(tiles, 0);
        shift(&game, &mut s, Dir::Up);
        assert_eq!((s.tile(0, 0), s.tile(0, 3)), (4, 4));

        let mut s = G2048State::from_tiles(tiles, 0);
        shift(&game, &mut s, Dir::Down);
        assert_eq!((s.tile(3, 0), s.tile(3, 3)), (4, 4));

        let mut s = G2048State::from_tiles(tiles, 0);
        shift(&game, &mut s, Dir::Right);
        assert_eq!(row(&s, 0), [0, 0, 0, 4]);
    }

    #[test]
    fn noop_moves_excluded_from_legal_actions() {
        let game = G2048;
        let s = G2048State::from_tiles([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [0; 4]], 0);
        assert_eq!(
            game.legal_actions(&s),
            vec![
                G2048Action::Shift(Dir::Down),
                G2048Action::Shift(Dir::Right)
            ]
        );

        let full_left_column =
            G2048State::from_tiles([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]], 0);
        assert_eq!(
            game.legal_actions(&full_left_column),
            vec![G2048Action::Shift(Dir::Right)]
        );
    }

    #[test]
    fn full_board_without_merges_is_terminal() {
        let game = G2048;
        let s = G2048State::from_tiles(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
            100,
        );
        assert!(game.is_terminal(&s));
        assert!(game.legal_actions(&s).is_empty());
        assert!(game.returns(&s, 0) > 0.0 && game.returns(&s, 0) < 1.0);
    }

    #[test]
    fn initial_state_owes_two_spawns() {
        let game = G2048;
        let mut s = game.initial_state();
        assert_eq!(game.turn(&s), Turn::Chance);
        assert!(!game.is_terminal(&s));
        let first = game.chance_outcomes(&s)[0].0;
        game.apply(&mut s, first);
        assert_eq!(game.turn(&s), Turn::Chance);
        let second = game.chance_outcomes(&s)[0].0;
        game.apply(&mut s, second);
        assert_eq!(game.turn(&s), Turn::Player(0));
    }

    #[test]
    fn chance_distribution_sums_to_one() {
        let game = G2048;
        let assert_dist = |s: &G2048State, expected_empties: usize| {
            let outs = game.chance_outcomes(s);
            assert_eq!(outs.len(), 2 * expected_empties);
            let total: f64 = outs.iter().map(|(_, p)| p).sum();
            assert!((total - 1.0).abs() < 1e-12, "total {total}");
            let twos: f64 = outs
                .iter()
                .filter(|(a, _)| matches!(a, G2048Action::Spawn { four: false, .. }))
                .map(|(_, p)| p)
                .sum();
            assert!((twos - 0.9).abs() < 1e-12, "P(spawn 2) = {twos}");
        };
        let mut s = game.initial_state();
        for expected_empties in [16usize, 15] {
            assert_dist(&s, expected_empties);
            let first = game.chance_outcomes(&s)[0].0;
            game.apply(&mut s, first);
        }
        // Both opening spawns are consumed now (it is the player's move), so
        // a third board size needs a hand-built chance node.
        s.pending_spawns = 1;
        assert_dist(&s, 14);
    }

    #[test]
    fn score_accounting_matches_merges() {
        let game = G2048;
        let mut s = G2048State::from_tiles([[2, 2, 2, 2], [0; 4], [0; 4], [0; 4]], 0);
        shift(&game, &mut s, Dir::Left);
        assert_eq!(s.score(), 8);
        game.apply(
            &mut s,
            G2048Action::Spawn {
                cell: 15,
                four: false,
            },
        );
        shift(&game, &mut s, Dir::Left);
        assert_eq!(s.score(), 16);
        assert_eq!(s.tile(0, 0), 8);
        assert_eq!(s.tile(3, 0), 2);
        assert_eq!(s.max_tile(), 8);
    }

    #[test]
    fn random_playthroughs_terminate() {
        let game = G2048;
        let mut best = 0u64;
        for seed in 0..30u64 {
            let mut rng = Rng::new(seed * 7 + 1);
            let mut s = game.initial_state();
            let mut steps = 0u32;
            while !game.is_terminal(&s) {
                steps += 1;
                assert!(steps < 100_000, "seed {seed} did not terminate");
                match game.turn(&s) {
                    Turn::Chance => {
                        let outs = game.chance_outcomes(&s);
                        let r = rng.unit();
                        let mut acc = 0.0;
                        let mut chosen = outs[outs.len() - 1].0;
                        for &(a, p) in &outs {
                            acc += p;
                            if r < acc {
                                chosen = a;
                                break;
                            }
                        }
                        game.apply(&mut s, chosen);
                    }
                    Turn::Player(_) => {
                        let acts = game.legal_actions(&s);
                        let i = (rng.unit() * acts.len() as f64) as usize;
                        game.apply(&mut s, acts[i.min(acts.len() - 1)]);
                    }
                }
            }
            let r = game.returns(&s, 0);
            assert!((0.0..=1.0).contains(&r), "returns {r} out of range");
            best = best.max(s.score());
        }
        assert!(best > 0, "30 random games never scored");
    }

    #[test]
    fn eval_stays_on_returns_scale_and_rewards_structure() {
        let game = G2048;
        let messy = G2048State::from_tiles(
            [[2, 64, 4, 2], [32, 2, 16, 4], [2, 8, 2, 8], [4, 2, 4, 2]],
            500,
        );
        let tidy = G2048State::from_tiles([[64, 32, 16, 8], [4, 2, 0, 0], [0; 4], [0; 4]], 500);
        let (em, et) = (
            Heuristic2048.eval(&game, &messy, 0),
            Heuristic2048.eval(&game, &tidy, 0),
        );
        assert!((0.0..=1.0).contains(&em) && (0.0..=1.0).contains(&et));
        assert!(et > em, "tidy {et} should beat messy {em}");
    }
}
