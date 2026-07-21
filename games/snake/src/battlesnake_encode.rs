//! Perspective-correct neural features for simultaneous Battlesnake.

use std::collections::VecDeque;

use game_core::SimultaneousPolicyValueEncoder;

use crate::battlesnake::{Battlesnake, BoardState, CELLS, Direction, MAX_HEALTH, bit, next_cell};

const MY_HEAD: usize = 0;
const MY_BODY: usize = 1;
const ENEMY_HEADS: usize = 2;
const ENEMY_BODIES: usize = 3;
const FOOD: usize = 4;
const HAZARDS: usize = 5;
const MY_LENGTH: usize = 6;
const ENEMY_LENGTH: usize = 7;
const MY_HEALTH: usize = 8;
const ENEMY_HEALTH: usize = 9;
const MY_REACH: usize = 10;
const ENEMY_REACH: usize = 11;
const MY_VORONOI: usize = 12;
const ENEMY_VORONOI: usize = 13;
const VORONOI_DIFF: usize = 14;
const MY_TAIL: usize = 15;
const ENEMY_TAILS: usize = 16;
const MY_MOVES: usize = 17;
const ENEMY_MOVES: usize = 18;
const TURN: usize = 19;
const LIVE_PLAYERS: usize = 20;
const WRAPPED: usize = 21;
const CONSTRICTOR: usize = 22;
const ONES: usize = 23;

pub const PLANES: usize = ONES + 1;

#[derive(Clone, Copy, Debug, Default)]
pub struct BattlesnakeEncoder;

impl<const N: usize> SimultaneousPolicyValueEncoder<Battlesnake<N>> for BattlesnakeEncoder {
    fn input_len(&self) -> usize {
        PLANES * CELLS
    }

    fn policy_len(&self) -> usize {
        4
    }

    fn encode_state(
        &self,
        game: &Battlesnake<N>,
        state: &BoardState<N>,
        player: usize,
    ) -> Vec<f32> {
        assert!(player < N);
        let mut out = vec![0.0; PLANES * CELLS];
        let me = state.snake(player);
        for (index, position) in me.cells().enumerate() {
            let plane = if index == 0 { MY_HEAD } else { MY_BODY };
            out[plane * CELLS + position as usize] = 1.0;
        }

        let mut largest_enemy = 0usize;
        let mut lowest_enemy_health = MAX_HEALTH;
        for enemy in 0..N {
            if enemy == player || !state.snake(enemy).is_alive() {
                continue;
            }
            let snake = state.snake(enemy);
            largest_enemy = largest_enemy.max(snake.len());
            lowest_enemy_health = lowest_enemy_health.min(snake.health());
            for (index, position) in snake.cells().enumerate() {
                let plane = if index == 0 {
                    ENEMY_HEADS
                } else {
                    ENEMY_BODIES
                };
                out[plane * CELLS + position as usize] = 1.0;
            }
            out[ENEMY_TAILS * CELLS + snake.tail() as usize] = 1.0;
        }
        mark_bits(&mut out[FOOD * CELLS..(FOOD + 1) * CELLS], state.food());
        mark_bits(
            &mut out[HAZARDS * CELLS..(HAZARDS + 1) * CELLS],
            state.hazards(),
        );
        out[MY_TAIL * CELLS + me.tail() as usize] = 1.0;

        let rules = game.rules();
        let wrapped = rules.mode.wrapped();
        let constrictor = rules.mode.constrictor();
        let obstacles = obstacles(state);
        mark_moves(
            &mut out[MY_MOVES * CELLS..(MY_MOVES + 1) * CELLS],
            me.head(),
            wrapped,
            obstacles,
        );
        for enemy in 0..N {
            if enemy != player && state.snake(enemy).is_alive() {
                mark_moves(
                    &mut out[ENEMY_MOVES * CELLS..(ENEMY_MOVES + 1) * CELLS],
                    state.snake(enemy).head(),
                    wrapped,
                    obstacles,
                );
            }
        }

        let my_distance = bfs(state, &[player], wrapped, obstacles);
        let enemies: Vec<_> = (0..N)
            .filter(|&enemy| enemy != player && state.snake(enemy).is_alive())
            .collect();
        let enemy_distance = bfs(state, &enemies, wrapped, obstacles);
        let mut my_voronoi = 0i32;
        let mut enemy_voronoi = 0i32;
        for position in 0..CELLS {
            if my_distance[position] != u16::MAX {
                out[MY_REACH * CELLS + position] = 1.0;
            }
            if enemy_distance[position] != u16::MAX {
                out[ENEMY_REACH * CELLS + position] = 1.0;
            }
            match my_distance[position].cmp(&enemy_distance[position]) {
                std::cmp::Ordering::Less => {
                    out[MY_VORONOI * CELLS + position] = 1.0;
                    my_voronoi += 1;
                }
                std::cmp::Ordering::Greater if enemy_distance[position] != u16::MAX => {
                    out[ENEMY_VORONOI * CELLS + position] = 1.0;
                    enemy_voronoi += 1;
                }
                _ => {}
            }
        }

        fill(&mut out, MY_LENGTH, me.len() as f32 / CELLS as f32);
        fill(&mut out, ENEMY_LENGTH, largest_enemy as f32 / CELLS as f32);
        fill(
            &mut out,
            MY_HEALTH,
            f32::from(me.health()) / f32::from(MAX_HEALTH),
        );
        fill(
            &mut out,
            ENEMY_HEALTH,
            f32::from(lowest_enemy_health) / f32::from(MAX_HEALTH),
        );
        fill(
            &mut out,
            VORONOI_DIFF,
            (my_voronoi - enemy_voronoi) as f32 / CELLS as f32,
        );
        fill(
            &mut out,
            TURN,
            f32::from(state.turn_number().min(500)) / 500.0,
        );
        fill(
            &mut out,
            LIVE_PLAYERS,
            state.alive_count() as f32 / N as f32,
        );
        fill(&mut out, WRAPPED, f32::from(wrapped));
        fill(&mut out, CONSTRICTOR, f32::from(constrictor));
        fill(&mut out, ONES, 1.0);
        out
    }

    fn action_index(
        &self,
        _game: &Battlesnake<N>,
        _state: &BoardState<N>,
        _player: usize,
        action: Direction,
    ) -> usize {
        action as usize
    }
}

fn fill(out: &mut [f32], plane: usize, value: f32) {
    out[plane * CELLS..(plane + 1) * CELLS].fill(value);
}

fn mark_bits(out: &mut [f32], mut board: u128) {
    while board != 0 {
        let position = board.trailing_zeros() as usize;
        out[position] = 1.0;
        board &= board - 1;
    }
}

fn obstacles<const N: usize>(state: &BoardState<N>) -> u128 {
    let mut board = 0;
    for snake in state.snakes() {
        if !snake.is_alive() {
            continue;
        }
        for position in snake.cells().take(snake.len().saturating_sub(1)) {
            board |= bit(position);
        }
    }
    board
}

fn mark_moves(out: &mut [f32], head: u8, wrapped: bool, obstacles: u128) {
    for direction in Direction::ALL {
        if let Some(position) = next_cell(head, direction, wrapped)
            && obstacles & bit(position) == 0
        {
            out[position as usize] = 1.0;
        }
    }
}

fn bfs<const N: usize>(
    state: &BoardState<N>,
    players: &[usize],
    wrapped: bool,
    obstacles: u128,
) -> [u16; CELLS] {
    let mut distance = [u16::MAX; CELLS];
    let mut queue = VecDeque::new();
    for &player in players {
        let head = state.snake(player).head();
        distance[head as usize] = 0;
        queue.push_back(head);
    }
    while let Some(position) = queue.pop_front() {
        let next_distance = distance[position as usize] + 1;
        for direction in Direction::ALL {
            let Some(next) = next_cell(position, direction, wrapped) else {
                continue;
            };
            if distance[next as usize] == u16::MAX && obstacles & bit(next) == 0 {
                distance[next as usize] = next_distance;
                queue.push_back(next);
            }
        }
    }
    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::battlesnake::Rules;
    use game_core::{SimultaneousGame, SimultaneousPolicyValueEncoder};

    #[test]
    fn perspective_swaps_heads_and_keeps_public_planes() {
        let game = Battlesnake::<2>::new(Rules::default());
        let state = game.initial_state();
        let encoder = BattlesnakeEncoder;
        let first = encoder.encode_state(&game, &state, 0);
        let second = encoder.encode_state(&game, &state, 1);
        assert_eq!(first.len(), PLANES * CELLS);
        let first_head = state.snake(0).head() as usize;
        let second_head = state.snake(1).head() as usize;
        assert_eq!(first[MY_HEAD * CELLS + first_head], 1.0);
        assert_eq!(first[ENEMY_HEADS * CELLS + second_head], 1.0);
        assert_eq!(second[MY_HEAD * CELLS + second_head], 1.0);
        assert_eq!(second[ENEMY_HEADS * CELLS + first_head], 1.0);
        assert_eq!(
            &first[FOOD * CELLS..(FOOD + 1) * CELLS],
            &second[FOOD * CELLS..(FOOD + 1) * CELLS]
        );
    }

    #[test]
    fn policy_is_the_four_absolute_directions() {
        let game = Battlesnake::<2>::new(Rules::default());
        let state = game.initial_state();
        let encoder = BattlesnakeEncoder;
        for (index, direction) in Direction::ALL.into_iter().enumerate() {
            assert_eq!(encoder.action_index(&game, &state, 0, direction), index);
        }
    }
}
