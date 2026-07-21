//! Canonical simultaneous Battlesnake rules on the standard 11×11 board.
//!
//! The transition order follows `BattlesnakeOfficial/rules`: every living
//! snake moves, loses health, takes hazard damage (unless it landed on food),
//! eats and grows, and is then eliminated for health, walls, body collisions,
//! or head-to-head collisions. Food placement is a separate chance transition
//! after the complete joint move has resolved.

use game_core::{Rng, SimultaneousGame, SimultaneousTurn};

pub mod search;
mod ui;

pub const SIDE: usize = 11;
pub const CELLS: usize = SIDE * SIDE;
pub const MAX_HEALTH: i16 = 100;
const BODY_CAPACITY: usize = 128;
const OUT_OF_BOUNDS: u8 = 127;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::Up,
        Direction::Right,
        Direction::Down,
        Direction::Left,
    ];

    pub const fn opposite(self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChanceAction {
    NoFood,
    PlaceFood(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Standard,
    Royale,
    Constrictor,
    Wrapped,
    WrappedConstrictor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialFood {
    /// Official Battlesnake opening: one nearby food per snake plus the center.
    Official,
    /// A calmer arcade opening with exactly one food at the center.
    One,
}

impl Mode {
    pub(crate) const fn wrapped(self) -> bool {
        matches!(self, Mode::Wrapped | Mode::WrappedConstrictor)
    }

    pub(crate) const fn constrictor(self) -> bool {
        matches!(self, Mode::Constrictor | Mode::WrappedConstrictor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub mode: Mode,
    pub initial_food: InitialFood,
    pub food_spawn_chance: u8,
    pub minimum_food: u8,
    pub hazard_damage: i16,
    pub shrink_every_n_turns: u16,
    pub seed: u64,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            mode: Mode::Standard,
            initial_food: InitialFood::Official,
            food_spawn_chance: 15,
            minimum_food: 1,
            hazard_damage: 14,
            shrink_every_n_turns: 25,
            seed: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elimination {
    Alive,
    OutOfHealth,
    Hazard,
    OutOfBounds,
    SelfCollision,
    BodyCollision,
    HeadToHead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Body {
    cells: [u8; BODY_CAPACITY],
    head: u8,
    len: u8,
}

impl Body {
    fn stacked(cell: u8) -> Self {
        let mut cells = [0; BODY_CAPACITY];
        cells[..3].fill(cell);
        Self {
            cells,
            head: 0,
            len: 3,
        }
    }

    fn from_cells(cells: &[u8]) -> Self {
        assert!(!cells.is_empty() && cells.len() <= BODY_CAPACITY);
        let mut out = [0; BODY_CAPACITY];
        out[..cells.len()].copy_from_slice(cells);
        Self {
            cells: out,
            head: 0,
            len: cells.len() as u8,
        }
    }

    #[inline]
    fn at(self, index: usize) -> u8 {
        debug_assert!(index < self.len as usize);
        self.cells[(self.head as usize + index) % BODY_CAPACITY]
    }

    #[inline]
    fn head(self) -> u8 {
        self.at(0)
    }

    #[inline]
    fn tail(self) -> u8 {
        self.at(self.len as usize - 1)
    }

    fn iter(self) -> impl Iterator<Item = u8> {
        (0..self.len as usize).map(move |index| self.at(index))
    }

    #[inline]
    fn advance(&mut self, new_head: u8) {
        self.head = if self.head == 0 {
            (BODY_CAPACITY - 1) as u8
        } else {
            self.head - 1
        };
        self.cells[self.head as usize] = new_head;
    }

    #[inline]
    fn grow(&mut self) {
        if self.len as usize >= BODY_CAPACITY {
            return;
        }
        // The official engine first drops the old tail, then appends a copy of
        // the new tail. The pre-move tail therefore vacates on an eating turn,
        // while the doubled tail stays put on the following turn.
        let tail = self.tail();
        let end = (self.head as usize + self.len as usize) % BODY_CAPACITY;
        self.cells[end] = tail;
        self.len += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BattleSnake {
    body: Body,
    health: i16,
    heading: Direction,
    elimination: Elimination,
}

impl BattleSnake {
    pub fn from_cells(cells: &[u8], health: i16, heading: Direction) -> Self {
        Self {
            body: Body::from_cells(cells),
            health,
            heading,
            elimination: Elimination::Alive,
        }
    }

    pub fn head(self) -> u8 {
        self.body.head()
    }

    pub fn tail(self) -> u8 {
        self.body.tail()
    }

    pub fn len(self) -> usize {
        self.body.len as usize
    }

    /// A competitive snake always retains its body after elimination.
    pub const fn is_empty(self) -> bool {
        false
    }

    pub fn is_alive(self) -> bool {
        self.elimination == Elimination::Alive
    }

    pub fn health(self) -> i16 {
        self.health
    }

    pub fn heading(self) -> Direction {
        self.heading
    }

    pub fn elimination(self) -> Elimination {
        self.elimination
    }

    pub fn cells(self) -> impl Iterator<Item = u8> {
        self.body.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Moves,
    RequiredFood(u8),
    FoodRoll,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardState<const N: usize> {
    snakes: [BattleSnake; N],
    food: u128,
    hazards: u128,
    turn: u16,
    phase: Phase,
}

impl<const N: usize> BoardState<N> {
    pub fn from_parts(snakes: [BattleSnake; N], food: u128, hazards: u128, turn: u16) -> Self {
        Self {
            snakes,
            food,
            hazards,
            turn,
            phase: Phase::Moves,
        }
    }

    pub fn snake(&self, player: usize) -> &BattleSnake {
        &self.snakes[player]
    }

    pub fn snakes(&self) -> &[BattleSnake; N] {
        &self.snakes
    }

    pub fn food(&self) -> u128 {
        self.food
    }

    pub fn hazards(&self) -> u128 {
        self.hazards
    }

    pub fn turn_number(&self) -> u16 {
        self.turn
    }

    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|snake| snake.is_alive()).count()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Battlesnake<const N: usize> {
    rules: Rules,
}

impl<const N: usize> Battlesnake<N> {
    pub fn new(rules: Rules) -> Self {
        assert!(
            (2..=4).contains(&N),
            "the competitive engine supports 2-4 players"
        );
        assert!(rules.food_spawn_chance <= 100);
        assert!(rules.shrink_every_n_turns > 0);
        Self { rules }
    }

    pub fn standard() -> Self {
        Self::new(Rules::default())
    }

    pub fn rules(self) -> Rules {
        self.rules
    }

    /// Actions that cannot kill this snake from walls, its own post-move
    /// body, starvation, or hazard damage. Enemy bodies and heads are
    /// deliberately excluded: their simultaneous moves can change whether a
    /// collision occurs, so masking them would hide legitimate tactics.
    pub fn nonfatal_action_mask(self, state: &BoardState<N>, player: usize) -> [bool; 4] {
        assert!(player < N);
        let snake = state.snakes[player];
        if !snake.is_alive() {
            return [true, false, false, false];
        }
        let own_body = snake
            .cells()
            .take(snake.len().saturating_sub(1))
            .fold(0u128, |occupied, cell| occupied | bit(cell));
        Direction::ALL.map(|action| {
            let Some(destination) = next_cell(snake.head(), action, self.rules.mode.wrapped())
            else {
                return false;
            };
            if own_body & bit(destination) != 0 {
                return false;
            }
            if state.food & bit(destination) != 0 {
                return true;
            }
            let hazard = if state.hazards & bit(destination) != 0 {
                self.rules.hazard_damage
            } else {
                0
            };
            snake.health - 1 - hazard > 0
        })
    }

    pub fn initial_state_with_rng(self, rng: &mut Rng) -> BoardState<N> {
        let min = 1u8;
        let mid = (SIDE / 2) as u8;
        let max = (SIDE - 2) as u8;
        let mut corners = [
            cell(min, min),
            cell(min, max),
            cell(max, min),
            cell(max, max),
        ];
        let mut cardinals = [
            cell(min, mid),
            cell(mid, min),
            cell(mid, max),
            cell(max, mid),
        ];
        shuffle(&mut corners, rng);
        shuffle(&mut cardinals, rng);
        let starts = if rng.below(2) == 0 {
            corners
        } else {
            cardinals
        };
        let snakes = std::array::from_fn(|player| BattleSnake {
            body: Body::stacked(starts[player]),
            health: MAX_HEALTH,
            heading: Direction::Up,
            elimination: Elimination::Alive,
        });
        let mut state = BoardState {
            snakes,
            food: 0,
            hazards: 0,
            turn: 0,
            phase: Phase::Moves,
        };
        match self.rules.initial_food {
            InitialFood::Official => self.place_initial_food(&mut state, rng),
            InitialFood::One => {
                state.food = bit(cell((SIDE / 2) as u8, (SIDE / 2) as u8));
            }
        }
        if self.rules.mode.constrictor() {
            state.food = 0;
        }
        state
    }

    fn place_initial_food(self, state: &mut BoardState<N>, rng: &mut Rng) {
        let center = cell((SIDE / 2) as u8, (SIDE / 2) as u8);
        for snake in state.snakes {
            let (x, y) = xy(snake.head());
            let mut candidates = [None; 4];
            let mut count = 0;
            for (dx, dy) in [(-1i16, -1i16), (-1, 1), (1, -1), (1, 1)] {
                let (nx, ny) = (i16::from(x) + dx, i16::from(y) + dy);
                if !(0..SIDE as i16).contains(&nx) || !(0..SIDE as i16).contains(&ny) {
                    continue;
                }
                let candidate = cell(nx as u8, ny as u8);
                if candidate == center || state.food & bit(candidate) != 0 {
                    continue;
                }
                let corner =
                    (nx == 0 || nx == SIDE as i16 - 1) && (ny == 0 || ny == SIDE as i16 - 1);
                if corner {
                    continue;
                }
                let away = (nx < i16::from(x) && x < SIDE as u8 / 2)
                    || (nx > i16::from(x) && x > SIDE as u8 / 2)
                    || (ny < i16::from(y) && y < SIDE as u8 / 2)
                    || (ny > i16::from(y) && y > SIDE as u8 / 2);
                if away {
                    candidates[count] = Some(candidate);
                    count += 1;
                }
            }
            if count > 0 {
                state.food |= bit(candidates[rng.below(count)].expect("candidate"));
            }
        }
        if state.occupied_alive() & bit(center) == 0 {
            state.food |= bit(center);
        }
    }

    fn resolve(self, state: &mut BoardState<N>, actions: &[Direction]) {
        assert_eq!(state.phase, Phase::Moves);
        assert_eq!(actions.len(), N);

        // Movement. A reverse is not rejected: like the official API it is a
        // valid direction and normally becomes a self collision with the neck.
        for (player, action) in actions.iter().copied().enumerate() {
            let snake = &mut state.snakes[player];
            if !snake.is_alive() {
                continue;
            }
            let next =
                next_cell(snake.head(), action, self.rules.mode.wrapped()).unwrap_or(OUT_OF_BOUNDS);
            snake.body.advance(next);
            snake.heading = action;
        }

        // Starvation and hazard damage happen before feeding. The official
        // hazard stage exempts a cell containing food, which then restores
        // health in the following feed stage.
        let old_food = state.food;
        for snake in &mut state.snakes {
            if !snake.is_alive() {
                continue;
            }
            snake.health -= 1;
            let head = snake.head();
            if head < CELLS as u8 && old_food & bit(head) == 0 && state.hazards & bit(head) != 0 {
                snake.health = (snake.health - self.rules.hazard_damage).clamp(0, MAX_HEALTH);
            }
        }

        let mut eaten = 0u128;
        for snake in &mut state.snakes {
            if !snake.is_alive() || snake.head() >= CELLS as u8 {
                continue;
            }
            let head_bit = bit(snake.head());
            if old_food & head_bit != 0 {
                snake.body.grow();
                snake.health = MAX_HEALTH;
                eaten |= head_bit;
            }
        }
        state.food &= !eaten;

        // Health/wall elimination is applied before collision checks, so an
        // already eliminated snake's body cannot eliminate another snake.
        for snake in &mut state.snakes {
            if !snake.is_alive() {
                continue;
            }
            if snake.health <= 0 {
                snake.elimination =
                    if snake.head() < CELLS as u8 && state.hazards & bit(snake.head()) != 0 {
                        Elimination::Hazard
                    } else {
                        Elimination::OutOfHealth
                    };
            } else if snake.head() >= CELLS as u8 {
                snake.elimination = Elimination::OutOfBounds;
            }
        }

        let before_collisions = state.snakes;
        let mut collision_deaths = [Elimination::Alive; N];
        for player in 0..N {
            let snake = before_collisions[player];
            if !snake.is_alive() {
                continue;
            }
            let head = snake.head();
            if snake.body.iter().skip(1).any(|body| body == head) {
                collision_deaths[player] = Elimination::SelfCollision;
                continue;
            }
            let mut body_hit = false;
            for (other_player, other) in before_collisions.iter().copied().enumerate() {
                if other_player == player || !other.is_alive() {
                    continue;
                }
                if other.body.iter().skip(1).any(|body| body == head) {
                    body_hit = true;
                    break;
                }
            }
            if body_hit {
                collision_deaths[player] = Elimination::BodyCollision;
                continue;
            }
            for (other_player, other) in before_collisions.iter().copied().enumerate() {
                if other_player == player || !other.is_alive() {
                    continue;
                }
                if other.head() == head && snake.len() <= other.len() {
                    collision_deaths[player] = Elimination::HeadToHead;
                    break;
                }
            }
        }
        for (snake, cause) in state.snakes.iter_mut().zip(collision_deaths) {
            if snake.is_alive() && cause != Elimination::Alive {
                snake.elimination = cause;
            }
        }

        if self.rules.mode.constrictor() {
            state.food = 0;
            for snake in &mut state.snakes {
                snake.health = MAX_HEALTH;
                if snake.len() >= 2 && snake.body.tail() != snake.body.at(snake.len() - 2) {
                    snake.body.grow();
                }
            }
        }

        state.turn = state.turn.saturating_add(1);
        if self.rules.mode == Mode::Royale {
            state.hazards =
                royale_hazards(self.rules.seed, state.turn, self.rules.shrink_every_n_turns);
        }
        self.prepare_food(state);
    }

    fn prepare_food(self, state: &mut BoardState<N>) {
        if state.alive_count() <= 1 || self.rules.mode.constrictor() {
            state.phase = Phase::Moves;
            return;
        }
        let current = state.food.count_ones() as u8;
        if current < self.rules.minimum_food {
            let missing = self.rules.minimum_food - current;
            state.phase = if state.available_food_cells() == 0 {
                Phase::Moves
            } else {
                Phase::RequiredFood(missing)
            };
        } else if self.rules.food_spawn_chance > 0 && state.available_food_cells() != 0 {
            state.phase = Phase::FoodRoll;
        } else {
            state.phase = Phase::Moves;
        }
    }
}

impl<const N: usize> BoardState<N> {
    fn occupied_alive(&self) -> u128 {
        let mut occupied = 0;
        for snake in self.snakes {
            if !snake.is_alive() {
                continue;
            }
            for body in snake.cells() {
                if body < CELLS as u8 {
                    occupied |= bit(body);
                }
            }
        }
        occupied
    }

    /// Official standard-map food placement excludes food, live bodies, and
    /// every orthogonal square a living head could enter on its next move.
    pub fn available_food_cells(&self) -> u128 {
        let mut blocked = self.food | self.occupied_alive();
        for snake in self.snakes {
            if !snake.is_alive() || snake.head() >= CELLS as u8 {
                continue;
            }
            for direction in Direction::ALL {
                if let Some(next) = next_cell(snake.head(), direction, false) {
                    blocked |= bit(next);
                }
            }
        }
        board_mask() & !blocked
    }
}

impl<const N: usize> SimultaneousGame for Battlesnake<N> {
    type State = BoardState<N>;
    type Action = Direction;
    type ChanceAction = ChanceAction;

    fn num_players(&self) -> usize {
        N
    }

    fn initial_state(&self) -> BoardState<N> {
        self.initial_state_with_rng(&mut Rng::new(self.rules.seed))
    }

    fn initial_state_with_rng(&self, rng: &mut Rng) -> BoardState<N> {
        Battlesnake::initial_state_with_rng(*self, rng)
    }

    fn turn(&self, state: &BoardState<N>) -> SimultaneousTurn {
        match state.phase {
            Phase::Moves => SimultaneousTurn::Players,
            Phase::RequiredFood(_) | Phase::FoodRoll => SimultaneousTurn::Chance,
        }
    }

    fn is_terminal(&self, state: &BoardState<N>) -> bool {
        state.alive_count() <= 1
    }

    fn is_active(&self, state: &BoardState<N>, player: usize) -> bool {
        state.snakes[player].is_alive()
    }

    fn returns(&self, state: &BoardState<N>, player: usize) -> f64 {
        if !self.is_terminal(state) {
            return 0.0;
        }
        match state.alive_count() {
            0 => 0.0,
            1 if state.snakes[player].is_alive() => 1.0,
            1 => -1.0,
            _ => unreachable!(),
        }
    }

    fn legal_actions(&self, _state: &BoardState<N>, _player: usize) -> Vec<Direction> {
        Direction::ALL.to_vec()
    }

    fn num_actions(&self, _state: &BoardState<N>, _player: usize) -> usize {
        4
    }

    fn action_at(&self, _state: &BoardState<N>, _player: usize, index: usize) -> Direction {
        Direction::ALL[index]
    }

    fn apply_joint(&self, state: &mut BoardState<N>, actions: &[Direction]) {
        self.resolve(state, actions);
    }

    fn chance_outcomes(&self, state: &BoardState<N>) -> Vec<(ChanceAction, f64)> {
        let available = state.available_food_cells();
        let count = available.count_ones() as usize;
        match state.phase {
            Phase::Moves => Vec::new(),
            Phase::RequiredFood(_) => {
                if count == 0 {
                    return vec![(ChanceAction::NoFood, 1.0)];
                }
                let probability = 1.0 / count as f64;
                bits(available)
                    .map(|cell| (ChanceAction::PlaceFood(cell), probability))
                    .collect()
            }
            Phase::FoodRoll => {
                if count == 0 || self.rules.food_spawn_chance == 0 {
                    return vec![(ChanceAction::NoFood, 1.0)];
                }
                // Source parity with maps/standard.go. The live standard map
                // rolls `(100-rand.Intn(100)) < chance`, so the configured
                // integer C has exactly C-1 successful values (including the
                // surprising 14% effective rate for the canonical default 15).
                let spawn = f64::from(self.rules.food_spawn_chance.saturating_sub(1)) / 100.0;
                if spawn == 0.0 {
                    return vec![(ChanceAction::NoFood, 1.0)];
                }
                let mut outcomes = Vec::with_capacity(count + 1);
                if spawn < 1.0 {
                    outcomes.push((ChanceAction::NoFood, 1.0 - spawn));
                }
                outcomes.extend(
                    bits(available)
                        .map(|cell| (ChanceAction::PlaceFood(cell), spawn / count as f64)),
                );
                outcomes
            }
        }
    }

    fn sample_chance_action(&self, state: &BoardState<N>, rng: &mut Rng) -> ChanceAction {
        let available = state.available_food_cells();
        let count = available.count_ones() as usize;
        let place_uniform = |rng: &mut Rng| {
            ChanceAction::PlaceFood(
                bits(available)
                    .nth(rng.below(count))
                    .expect("non-empty available food cells"),
            )
        };
        match state.phase {
            Phase::Moves => panic!("chance action requested during a player turn"),
            Phase::RequiredFood(_) if count == 0 => ChanceAction::NoFood,
            Phase::RequiredFood(_) => place_uniform(rng),
            Phase::FoodRoll if count == 0 || self.rules.food_spawn_chance == 0 => {
                ChanceAction::NoFood
            }
            Phase::FoodRoll => {
                // Exact maps/standard.go predicate; configured 15 means 14
                // successful values out of rand.Intn(100)'s 0..99 range.
                let roll = rng.below(100);
                if 100 - roll < usize::from(self.rules.food_spawn_chance) {
                    place_uniform(rng)
                } else {
                    ChanceAction::NoFood
                }
            }
        }
    }

    fn apply_chance(&self, state: &mut BoardState<N>, action: ChanceAction) {
        match (state.phase, action) {
            (Phase::FoodRoll, ChanceAction::NoFood) => state.phase = Phase::Moves,
            (Phase::FoodRoll, ChanceAction::PlaceFood(cell)) => {
                assert!(state.available_food_cells() & bit(cell) != 0);
                state.food |= bit(cell);
                state.phase = Phase::Moves;
            }
            (Phase::RequiredFood(remaining), ChanceAction::PlaceFood(cell)) => {
                assert!(state.available_food_cells() & bit(cell) != 0);
                state.food |= bit(cell);
                state.phase = if remaining > 1 && state.available_food_cells() != 0 {
                    Phase::RequiredFood(remaining - 1)
                } else {
                    Phase::Moves
                };
            }
            (Phase::RequiredFood(_), ChanceAction::NoFood) if state.available_food_cells() == 0 => {
                state.phase = Phase::Moves;
            }
            _ => panic!(
                "invalid Battlesnake chance action {action:?} for {:?}",
                state.phase
            ),
        }
    }

    fn state_key(&self, state: &BoardState<N>) -> Option<u64> {
        use game_core::hash::combine;
        let mut key = combine(state.turn as u64, state.food as u64);
        key = combine(
            key,
            match state.phase {
                Phase::Moves => 0,
                Phase::RequiredFood(remaining) => 1 + u64::from(remaining),
                Phase::FoodRoll => 257,
            },
        );
        key = combine(key, (state.food >> 64) as u64);
        key = combine(key, state.hazards as u64);
        key = combine(key, (state.hazards >> 64) as u64);
        for snake in state.snakes {
            key = combine(key, snake.health as u64);
            key = combine(key, snake.elimination as u64);
            key = combine(key, snake.len() as u64);
            for body in snake.cells() {
                key = combine(key, u64::from(body));
            }
        }
        Some(key)
    }

    fn action_id(&self, action: &Direction) -> u64 {
        *action as u64
    }
}

#[inline]
pub const fn cell(x: u8, y: u8) -> u8 {
    y * SIDE as u8 + x
}

#[inline]
pub const fn xy(cell: u8) -> (u8, u8) {
    (cell % SIDE as u8, cell / SIDE as u8)
}

#[inline]
pub const fn bit(cell: u8) -> u128 {
    1u128 << cell
}

#[inline]
pub const fn board_mask() -> u128 {
    (1u128 << CELLS) - 1
}

fn bits(mut board: u128) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if board == 0 {
            return None;
        }
        let cell = board.trailing_zeros() as u8;
        board &= board - 1;
        Some(cell)
    })
}

pub(crate) fn next_cell(from: u8, direction: Direction, wrapped: bool) -> Option<u8> {
    if from >= CELLS as u8 {
        return None;
    }
    let (x, y) = xy(from);
    let (mut nx, mut ny) = match direction {
        Direction::Up => (i16::from(x), i16::from(y) + 1),
        Direction::Right => (i16::from(x) + 1, i16::from(y)),
        Direction::Down => (i16::from(x), i16::from(y) - 1),
        Direction::Left => (i16::from(x) - 1, i16::from(y)),
    };
    if wrapped {
        nx = nx.rem_euclid(SIDE as i16);
        ny = ny.rem_euclid(SIDE as i16);
    }
    ((0..SIDE as i16).contains(&nx) && (0..SIDE as i16).contains(&ny))
        .then(|| cell(nx as u8, ny as u8))
}

fn shuffle<T>(values: &mut [T], rng: &mut Rng) {
    for i in (1..values.len()).rev() {
        values.swap(i, rng.below(i + 1));
    }
}

fn royale_hazards(seed: u64, turn: u16, every: u16) -> u128 {
    let shrinks = turn / every;
    if shrinks == 0 {
        return 0;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        (0i16, SIDE as i16 - 1, 0i16, SIDE as i16 - 1);
    for index in 0..shrinks {
        match game_core::hash::splitmix64(seed.wrapping_add(u64::from(index))) & 3 {
            0 => min_x += 1,
            1 => max_x -= 1,
            2 => min_y += 1,
            _ => max_y -= 1,
        }
    }
    let mut hazards = 0;
    for y in 0..SIDE as i16 {
        for x in 0..SIDE as i16 {
            if x < min_x || x > max_x || y < min_y || y > max_y {
                hazards |= bit(cell(x as u8, y as u8));
            }
        }
    }
    hazards
}
