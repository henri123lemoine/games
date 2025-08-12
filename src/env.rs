use serde::{Deserialize, Serialize};
use thiserror::Error;

const NUM_CARDS: usize = 11;
const TARGET: u8 = 21;

/// Action available to the current player: draw a card or stand.
///
/// Example
/// let a = Action::Draw;
/// assert!(matches!(a, Action::Draw));
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Draw,
    Stand,
}

/// Partial observation available to an agent controlling one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub self_total: u8,
    pub opp_face_up: u8,
    pub self_face_up: u8,
    pub self_face_down: u8,
    pub self_stood: bool,
    pub opp_stood: bool,
    pub deck_count: u8,
    pub round: u8,
    pub self_hearts: u8,
    pub opp_hearts: u8,
}

/// Result of a completed round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundOutcome {
    pub winner: Option<usize>,
    pub damage: u8,
}

#[derive(Debug, Error)]
pub enum EnvError {
    #[error("no active round; call start_new_round")]
    NoActiveRound,
    #[error("game is over")]
    GameOver,
    #[error("invalid action: {0}")]
    InvalidAction(&'static str),
}

#[derive(Debug, Clone)]
struct PlayerRound {
    face_up: u8,
    face_down: u8,
    total: u8,
    stood: bool,
    bust: bool,
    draws: u8,
}

impl PlayerRound {
    fn new() -> Self {
        Self {
            face_up: 0,
            face_down: 0,
            total: 0,
            stood: false,
            bust: false,
            draws: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct RoundState {
    deck_mask: u16, // 11-bit mask: bit i => card (i+1) available
    deck_order: Option<[u8; NUM_CARDS]>,
    draw_index: u8, // index into deck_order when present
    players: [PlayerRound; 2],
}

impl RoundState {
    fn new_with_mask_and_order(mask: u16, order: Option<[u8; NUM_CARDS]>) -> Self {
        Self {
            deck_mask: mask,
            deck_order: order,
            draw_index: 0,
            players: [PlayerRound::new(), PlayerRound::new()],
        }
    }

    fn deck_count(&self) -> u8 {
        self.deck_mask.count_ones() as u8
    }
}

/// Twenty-One game environment for two players.
///
/// Use [`Env::start_new_round`] to deal cards, then repeatedly call [`Env::step`]
/// with [`Action::Draw`] or [`Action::Stand`]. When a round ends, the returned
/// [`StepResult`] contains the outcome and whether the game is over.
///
/// Example
/// let mut env = Env::new(42);
/// env.start_new_round().unwrap();
/// loop {
///     let p = env.current_player();
///     let obs = env.observation(p);
///     let act = if obs.self_total < 17 { Action::Draw } else { Action::Stand };
///     let res = env.step(act).unwrap();
///     if res.round_over { break; }
/// }
#[derive(Debug, Clone)]
pub struct Env {
    hearts: [u8; 2],
    round: u8,
    current_player: u8,
    round_state: Option<RoundState>,
    preset_decks: Vec<[u8; NUM_CARDS]>,
    preset_round_index: usize,
    rng: XorShift64,
    game_over: bool,
}

impl Env {
    /// Create a new environment with a random seed.
    pub fn new(seed: u64) -> Self {
        Self {
            hearts: [6, 6],
            round: 1,
            current_player: 0,
            round_state: None,
            preset_decks: Vec::new(),
            preset_round_index: 0,
            rng: XorShift64::seed(seed),
            game_over: false,
        }
    }

    /// Create a new environment using predetermined deck orders per round.
    /// Create a new environment using predetermined deck orders per round.
    /// Intended for deterministic tests.
    pub fn new_with_preset_decks(preset_decks: Vec<[u8; NUM_CARDS]>) -> Self {
        Self::new_with_preset_decks_and_hearts(preset_decks, 6)
    }

    /// Create a new environment using predetermined deck orders and starting hearts.
    /// Create a new environment using predetermined deck orders and starting hearts.
    /// Intended for deterministic tests.
    pub fn new_with_preset_decks_and_hearts(
        preset_decks: Vec<[u8; NUM_CARDS]>,
        starting_hearts: u8,
    ) -> Self {
        Self {
            hearts: [starting_hearts, starting_hearts],
            round: 1,
            current_player: 0,
            round_state: None,
            preset_decks,
            preset_round_index: 0,
            rng: XorShift64::seed(0xDEADBEEFCAFEBABE),
            game_over: false,
        }
    }

    /// Current round number (starts at 1).
    pub fn round(&self) -> u8 {
        self.round
    }

    /// Hearts remaining for a player (0 or 1).
    pub fn hearts(&self, player: usize) -> u8 {
        self.hearts[player]
    }

    /// Index of the current player (0 or 1).
    pub fn current_player(&self) -> usize {
        self.current_player as usize
    }

    /// Start a new round: deals one face-up and one face-down card to each player.
    pub fn start_new_round(&mut self) -> Result<(), EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        if self.round_state.is_some() {
            return Err(EnvError::InvalidAction("round already in progress"));
        }
        let (mask, order_opt) = if let Some(order) = self.preset_decks.get(self.preset_round_index)
        {
            let mut mask: u16 = 0;
            for &c in order.iter() {
                mask |= 1u16 << (c as u16 - 1);
            }
            (mask, Some(*order))
        } else {
            // mask for cards 1..=11
            (0x7FF, None)
        };
        self.preset_round_index = self.preset_round_index.saturating_add(1);
        let mut rs = RoundState::new_with_mask_and_order(mask, order_opt);
        // Deal: P0 face-up, P1 face-up, P0 face-down, P1 face-down
        let p0_up = self.draw_card(&mut rs)?;
        let p1_up = self.draw_card(&mut rs)?;
        let p0_dn = self.draw_card(&mut rs)?;
        let p1_dn = self.draw_card(&mut rs)?;
        rs.players[0].face_up = p0_up;
        rs.players[1].face_up = p1_up;
        rs.players[0].face_down = p0_dn;
        rs.players[1].face_down = p1_dn;
        rs.players[0].total = p0_up.saturating_add(p0_dn);
        rs.players[1].total = p1_up.saturating_add(p1_dn);
        self.round_state = Some(rs);
        self.current_player = 0;
        Ok(())
    }

    /// Get current observation for a player.
    pub fn observation(&self, player: usize) -> Observation {
        let rs = self.round_state.as_ref();
        let (self_total, self_up, self_dn, self_stood, opp_up, opp_stood, deck_count) =
            if let Some(rs) = rs {
                let me = &rs.players[player];
                let opp = &rs.players[1 - player];
                (
                    me.total,
                    me.face_up,
                    me.face_down,
                    me.stood,
                    opp.face_up,
                    opp.stood,
                    rs.deck_count(),
                )
            } else {
                (0, 0, 0, false, 0, false, 0)
            };
        Observation {
            self_total,
            opp_face_up: opp_up,
            self_face_up: self_up,
            self_face_down: self_dn,
            self_stood,
            opp_stood,
            deck_count,
            round: self.round,
            self_hearts: self.hearts[player],
            opp_hearts: self.hearts[1 - player],
        }
    }

    /// Take an action for the current player.
    /// Take an action for the current player.
    /// Returns whether the round or game ended and the round outcome.
    pub fn step(&mut self, action: Action) -> Result<StepResult, EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        let mut rs = self.round_state.take().ok_or(EnvError::NoActiveRound)?;
        let p = self.current_player as usize;
        match action {
            Action::Draw => {
                if rs.players[p].stood {
                    return Err(EnvError::InvalidAction("cannot draw after standing"));
                }
                if rs.deck_count() == 0 {
                    return Err(EnvError::InvalidAction("no cards remaining"));
                }
                let c = self.draw_card(&mut rs)?;
                {
                    let me = &mut rs.players[p];
                    me.total = me.total.saturating_add(c);
                    me.draws = me.draws.saturating_add(1);
                    if me.total > TARGET {
                        me.bust = true;
                        me.stood = true;
                    }
                }
            }
            Action::Stand => {
                rs.players[p].stood = true;
            }
        }

        // Determine if round ends
        let both_stood = rs.players[0].stood && rs.players[1].stood;
        let deck_empty = rs.deck_count() == 0;
        let mut round_over = false;
        let mut outcome = None;
        if both_stood || deck_empty {
            round_over = true;
            let (winner, damage) = self.evaluate_winner(&rs);
            outcome = Some(RoundOutcome { winner, damage });
            if let Some(w) = winner {
                let loser = 1 - w;
                let dmg = damage;
                self.hearts[loser] = self.hearts[loser].saturating_sub(dmg);
                if self.hearts[loser] == 0 {
                    self.game_over = true;
                }
            }
            // Prepare for next round
            self.round = self.round.saturating_add(1);
            self.current_player = 0;
            self.round_state = None;
            return Ok(StepResult {
                round_over,
                game_over: self.game_over,
                outcome,
            });
        }

        // Continue the round; switch player
        self.current_player ^= 1;
        self.round_state = Some(rs);
        Ok(StepResult {
            round_over,
            game_over: false,
            outcome,
        })
    }

    fn evaluate_winner(&self, rs: &RoundState) -> (Option<usize>, u8) {
        let a = &rs.players[0];
        let b = &rs.players[1];
        let a_over = a.total > TARGET;
        let b_over = b.total > TARGET;
        let damage = self.round; // damage equals round number
        let winner = if a_over ^ b_over {
            if a_over {
                Some(1)
            } else {
                Some(0)
            }
        } else {
            let a_dist = if a.total > TARGET {
                a.total - TARGET
            } else {
                TARGET - a.total
            };
            let b_dist = if b.total > TARGET {
                b.total - TARGET
            } else {
                TARGET - b.total
            };
            match a_dist.cmp(&b_dist) {
                core::cmp::Ordering::Less => Some(0),
                core::cmp::Ordering::Greater => Some(1),
                core::cmp::Ordering::Equal => None,
            }
        };
        (winner, damage)
    }

    fn draw_card(&mut self, rs: &mut RoundState) -> Result<u8, EnvError> {
        if rs.deck_count() == 0 {
            return Err(EnvError::InvalidAction("no cards remaining"));
        }
        if let Some(order) = rs.deck_order {
            // Sequential draw from preset deck (deterministic for tests)
            let idx = rs.draw_index as usize;
            if idx >= NUM_CARDS {
                return Err(EnvError::InvalidAction("no cards remaining"));
            }
            rs.draw_index = (idx as u8) + 1;
            let card = order[idx];
            let bit = 1u16 << (card as u16 - 1);
            rs.deck_mask &= !bit;
            Ok(card)
        } else {
            // Random draw from remaining mask using XORSHIFT
            let remaining = rs.deck_mask.count_ones() as u64;
            let k = (self.rng.next_u64() % remaining) as u32;
            let mut seen = 0u32;
            for i in 0..NUM_CARDS {
                if (rs.deck_mask & (1u16 << i)) != 0 {
                    if seen == k {
                        rs.deck_mask &= !(1u16 << i);
                        return Ok((i as u8) + 1);
                    }
                    seen += 1;
                }
            }
            Err(EnvError::InvalidAction("failed to draw card"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub round_over: bool,
    pub game_over: bool,
    pub outcome: Option<RoundOutcome>,
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn seed(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}
