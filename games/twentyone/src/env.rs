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
    last_action_stand: bool,
    bust: bool,
    draws: u8,
    up_cards: [u8; 8],
    up_len: u8,
}

impl PlayerRound {
    fn new() -> Self {
        Self {
            face_up: 0,
            face_down: 0,
            total: 0,
            last_action_stand: false,
            bust: false,
            draws: 0,
            up_cards: [0; 8],
            up_len: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct RoundState {
    deck_mask: u16, // 11-bit mask: bit i => card (i+1) available
    deck_order: Option<[u8; NUM_CARDS]>,
    draw_index: u8, // index into deck_order when present
    players: [PlayerRound; 2],
    prev_stand: bool,
}

impl RoundState {
    fn new_with_mask_and_order(mask: u16, order: Option<[u8; NUM_CARDS]>) -> Self {
        Self {
            deck_mask: mask,
            deck_order: order,
            draw_index: 0,
            players: [PlayerRound::new(), PlayerRound::new()],
            prev_stand: false,
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
    last_reveal_down: Option<[u8; 2]>,
    last_public_up: Option<([u8; 8], u8, [u8; 8], u8)>,
    preset_decks: Vec<[u8; NUM_CARDS]>,
    preset_round_index: usize,
    rng: XorShift64,
    game_over: bool,
}

/// Heart counts above this alias in the packed state/infoset keys, which
/// allocate 3 bits per player's hearts.
pub const MAX_HEARTS: u8 = 7;

fn assert_hearts(starting_hearts: u8) {
    assert!(
        (1..=MAX_HEARTS).contains(&starting_hearts),
        "starting hearts must be in 1..={MAX_HEARTS}: the packed state keys \
         allocate 3 bits per heart count"
    );
}

impl Env {
    /// Create a new environment with a random seed.
    pub fn new(seed: u64) -> Self {
        Self {
            hearts: [6, 6],
            round: 1,
            current_player: 0,
            round_state: None,
            last_reveal_down: None,
            last_public_up: None,
            preset_decks: Vec::new(),
            preset_round_index: 0,
            rng: XorShift64::seed(seed),
            game_over: false,
        }
    }

    /// Create a new environment with a chosen seed and starting heart count.
    pub fn with_hearts(seed: u64, starting_hearts: u8) -> Self {
        assert_hearts(starting_hearts);
        Self {
            hearts: [starting_hearts, starting_hearts],
            ..Self::new(seed)
        }
    }

    /// Create a new environment using predetermined deck orders per round.
    /// Intended for deterministic tests.
    pub fn new_with_preset_decks(preset_decks: Vec<[u8; NUM_CARDS]>) -> Self {
        Self::new_with_preset_decks_and_hearts(preset_decks, 6)
    }

    /// Create a new environment using predetermined deck orders and starting hearts.
    /// Intended for deterministic tests.
    pub fn new_with_preset_decks_and_hearts(
        preset_decks: Vec<[u8; NUM_CARDS]>,
        starting_hearts: u8,
    ) -> Self {
        assert_hearts(starting_hearts);
        Self {
            hearts: [starting_hearts, starting_hearts],
            round: 1,
            current_player: 0,
            round_state: None,
            last_reveal_down: None,
            last_public_up: None,
            preset_decks,
            preset_round_index: 0,
            rng: XorShift64::seed(0xDEADBEEFCAFEBABE),
            game_over: false,
        }
    }

    /// Construct an environment paused between rounds at a given hearts/round
    /// state (no active round, RNG seeded deterministically). Used by the solver
    /// to evaluate round subgames independently.
    pub fn from_state(hearts: [u8; 2], round: u8) -> Self {
        assert!(
            hearts[0] <= MAX_HEARTS && hearts[1] <= MAX_HEARTS,
            "heart counts above {MAX_HEARTS} alias in the packed state keys"
        );
        Self {
            hearts,
            round,
            current_player: 0,
            round_state: None,
            last_reveal_down: None,
            last_public_up: None,
            preset_decks: Vec::new(),
            preset_round_index: 0,
            rng: XorShift64::seed(0x1234_5678_9ABC_DEF0),
            game_over: hearts[0] == 0 || hearts[1] == 0,
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
        rs.players[0].up_cards[0] = p0_up;
        rs.players[0].up_len = 1;
        rs.players[1].up_cards[0] = p1_up;
        rs.players[1].up_len = 1;
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
                    me.last_action_stand,
                    opp.face_up,
                    opp.last_action_stand,
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

    /// Take an action for the current player, drawing from the internal RNG.
    ///
    /// This is the self-play / interactive entry point. For solver search where
    /// chance must be controlled, use [`Env::draw_specific`] and [`Env::stand`].
    pub fn step(&mut self, action: Action) -> Result<StepResult, EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        let mut rs = self.round_state.take().ok_or(EnvError::NoActiveRound)?;
        let p = self.current_player as usize;
        match action {
            Action::Draw => {
                if rs.deck_count() == 0 {
                    return Err(EnvError::InvalidAction("no cards remaining"));
                }
                let c = self.draw_card(&mut rs)?;
                Self::apply_draw_to(&mut rs, p, c);
                Ok(self.resolve_step(rs, false))
            }
            Action::Stand => {
                let will_end = Self::apply_stand_to(&mut rs, p);
                Ok(self.resolve_step(rs, will_end))
            }
        }
    }

    /// Record a drawn card `c` for player `p` within the round state.
    fn apply_draw_to(rs: &mut RoundState, p: usize, c: u8) {
        let me = &mut rs.players[p];
        me.total = me.total.saturating_add(c);
        me.draws = me.draws.saturating_add(1);
        me.last_action_stand = false;
        if (me.up_len as usize) < me.up_cards.len() {
            me.up_cards[me.up_len as usize] = c;
            me.up_len = me.up_len.saturating_add(1);
        }
        if me.total > TARGET {
            me.bust = true;
        }
        rs.prev_stand = false;
    }

    /// Record a stand for player `p`. Returns whether this ends the round
    /// (two consecutive stands).
    fn apply_stand_to(rs: &mut RoundState, p: usize) -> bool {
        rs.players[p].last_action_stand = true;
        let will_end = rs.prev_stand;
        rs.prev_stand = true;
        will_end
    }

    /// Resolve a round after an action: either end the round (and possibly the
    /// game) or hand the turn to the other player.
    fn resolve_step(&mut self, rs: RoundState, will_end_by_consec: bool) -> StepResult {
        let deck_empty = rs.deck_count() == 0;
        if will_end_by_consec || deck_empty {
            let (winner, damage) = self.evaluate_winner(&rs);
            self.last_reveal_down = Some([rs.players[0].face_down, rs.players[1].face_down]);
            self.last_public_up = Some((
                rs.players[0].up_cards,
                rs.players[0].up_len,
                rs.players[1].up_cards,
                rs.players[1].up_len,
            ));
            if let Some(w) = winner {
                let loser = 1 - w;
                self.hearts[loser] = self.hearts[loser].saturating_sub(damage);
                if self.hearts[loser] == 0 {
                    self.game_over = true;
                }
            }
            self.round = self.round.saturating_add(1);
            self.current_player = 0;
            self.round_state = None;
            return StepResult {
                round_over: true,
                game_over: self.game_over,
                outcome: Some(RoundOutcome { winner, damage }),
            };
        }
        self.current_player ^= 1;
        self.round_state = Some(rs);
        StepResult {
            round_over: false,
            game_over: false,
            outcome: None,
        }
    }

    /// Whether the game has ended (a player reached 0 hearts).
    pub fn is_game_over(&self) -> bool {
        self.game_over
    }

    /// Whether a round is currently in progress (cards dealt, awaiting actions).
    pub fn round_active(&self) -> bool {
        self.round_state.is_some()
    }

    /// Terminal utility for `player`: +1 if they are the surviving winner, else -1.
    pub fn utility(&self, player: usize) -> f64 {
        if self.hearts[player] > 0 { 1.0 } else { -1.0 }
    }

    /// The 11-bit deck mask for the active round (bit `i` set => card `i+1`
    /// remains). Zero if no round is active. Allocation-free hot path for solvers.
    pub fn deck_mask(&self) -> u16 {
        self.round_state
            .as_ref()
            .map(|rs| rs.deck_mask)
            .unwrap_or(0)
    }

    /// The 11-bit mask of cards `player` has not seen: the undrawn deck plus the
    /// opponent's hidden face-down card. From `player`'s information set the
    /// opponent's face-down is uniformly one of these. Zero if no round is active.
    pub fn unseen_mask(&self, player: usize) -> u16 {
        match &self.round_state {
            Some(rs) => {
                let opp = 1 - player;
                rs.deck_mask | (1u16 << (rs.players[opp].face_down as u16 - 1))
            }
            None => 0,
        }
    }

    /// A hypothetical clone in which `player`'s opponent holds face-down card `f`
    /// instead of its actual one, with the deck adjusted so the real card returns
    /// and `f` is removed. Used to enumerate determinizations of the one hidden
    /// card for inference-time search. `f` must lie in [`Env::unseen_mask`].
    pub fn with_opp_facedown(&self, player: usize, f: u8) -> Self {
        let mut env = self.clone();
        if let Some(rs) = env.round_state.as_mut() {
            let opp = 1 - player;
            let old = rs.players[opp].face_down;
            if old != f {
                rs.players[opp].total = rs.players[opp].total - old + f;
                rs.players[opp].face_down = f;
                rs.deck_mask |= 1u16 << (old as u16 - 1);
                rs.deck_mask &= !(1u16 << (f as u16 - 1));
            }
        }
        env
    }

    /// The cards still in the deck for the active round (true, god's-eye view).
    /// Empty if no round is active.
    pub fn remaining_deck(&self) -> Vec<u8> {
        match &self.round_state {
            Some(rs) => (0..NUM_CARDS as u16)
                .filter(|i| rs.deck_mask & (1 << i) != 0)
                .map(|i| (i + 1) as u8)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Begin a round by dealing four specific cards in dealing order
    /// (p0 face-up, p1 face-up, p0 face-down, p1 face-down), bypassing the RNG.
    ///
    /// This is the controllable-chance counterpart of [`Env::start_new_round`],
    /// used to enumerate the deal during exact best-response search. Always draws
    /// from the full 1..=11 deck (preset decks are ignored).
    pub fn deal_specific(&mut self, cards: [u8; 4]) -> Result<(), EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        if self.round_state.is_some() {
            return Err(EnvError::InvalidAction("round already in progress"));
        }
        let mut rs = RoundState::new_with_mask_and_order(0x7FF, None);
        for &c in &cards {
            let bit = Self::card_bit(c)?;
            if rs.deck_mask & bit == 0 {
                return Err(EnvError::InvalidAction("card not available to deal"));
            }
            rs.deck_mask &= !bit;
        }
        rs.players[0].face_up = cards[0];
        rs.players[1].face_up = cards[1];
        rs.players[0].face_down = cards[2];
        rs.players[1].face_down = cards[3];
        rs.players[0].total = cards[0].saturating_add(cards[2]);
        rs.players[1].total = cards[1].saturating_add(cards[3]);
        rs.players[0].up_cards[0] = cards[0];
        rs.players[0].up_len = 1;
        rs.players[1].up_cards[0] = cards[1];
        rs.players[1].up_len = 1;
        self.round_state = Some(rs);
        self.current_player = 0;
        Ok(())
    }

    /// Draw a specific card for the current player, bypassing the RNG.
    /// The controllable-chance counterpart of `step(Action::Draw)`.
    pub fn draw_specific(&mut self, card: u8) -> Result<StepResult, EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        let mut rs = self.round_state.take().ok_or(EnvError::NoActiveRound)?;
        let bit = Self::card_bit(card)?;
        if rs.deck_mask & bit == 0 {
            return Err(EnvError::InvalidAction("card not available to draw"));
        }
        rs.deck_mask &= !bit;
        let p = self.current_player as usize;
        Self::apply_draw_to(&mut rs, p, card);
        Ok(self.resolve_step(rs, false))
    }

    /// Stand for the current player (chance-free; the controllable counterpart
    /// of `step(Action::Stand)`).
    pub fn stand(&mut self) -> Result<StepResult, EnvError> {
        if self.game_over {
            return Err(EnvError::GameOver);
        }
        let mut rs = self.round_state.take().ok_or(EnvError::NoActiveRound)?;
        let p = self.current_player as usize;
        let will_end = Self::apply_stand_to(&mut rs, p);
        Ok(self.resolve_step(rs, will_end))
    }

    fn card_bit(card: u8) -> Result<u16, EnvError> {
        if card < 1 || card as usize > NUM_CARDS {
            return Err(EnvError::InvalidAction("card out of range"));
        }
        Ok(1u16 << (card as u16 - 1))
    }

    /// A packed, lossless sufficient-statistic key for `player`'s current
    /// information set, used to share regret/strategy across strategically
    /// equivalent histories.
    ///
    /// Two histories share a key iff `player` faces an identical decision: same
    /// round, hearts, own total, stood flags, and the same *set of cards they
    /// have not seen*. The opponent's visible total is implied (it equals
    /// `66 - own_total - sum(unseen)`), and draw order is irrelevant because the
    /// next drawn card is uniform over the unseen set. Returns 0 when no round
    /// is active.
    pub fn sufficient_key(&self, player: usize) -> u64 {
        let rs = match &self.round_state {
            Some(rs) => rs,
            None => return 0,
        };
        let opp = 1 - player;
        let me = &rs.players[player];
        let op = &rs.players[opp];
        let mut seen: u16 = 0;
        for i in 0..me.up_len as usize {
            seen |= 1 << (me.up_cards[i] as u16 - 1);
        }
        seen |= 1 << (me.face_down as u16 - 1);
        for i in 0..op.up_len as usize {
            seen |= 1 << (op.up_cards[i] as u16 - 1);
        }
        let unseen = (!seen) & 0x7FF;
        let round = (self.round as u64).min(63);
        (round & 0x3F)
            | ((self.hearts[player] as u64 & 0x7) << 6)
            | ((self.hearts[opp] as u64 & 0x7) << 9)
            | ((me.total as u64 & 0x7F) << 12)
            | ((unseen as u64) << 19)
            | ((me.last_action_stand as u64) << 30)
            | ((op.last_action_stand as u64) << 31)
    }

    /// A lossy abstraction of [`Env::sufficient_key`] for large variants: it keeps
    /// round, hearts, own total, and stood flags exact, but instead of the exact
    /// unseen-card *set* it stores the count of unseen cards in four value bands
    /// (1–3, 4–6, 7–9, 10–11).
    ///
    /// That histogram is what the draw decision actually needs — it determines
    /// the bust probability (how many unseen cards exceed `21 - total`) and the
    /// opponent's possible holdings — so play under this abstraction stays strong,
    /// while the per-subgame information-set count drops by roughly an order of
    /// magnitude, letting CFR converge far faster on the full game. (The headline
    /// exploitability metric still uses the true game, so it fairly measures how
    /// exploitable the abstract strategy is in real play.)
    pub fn abstract_key(&self, player: usize) -> u64 {
        let rs = match &self.round_state {
            Some(rs) => rs,
            None => return 0,
        };
        let opp = 1 - player;
        let me = &rs.players[player];
        let op = &rs.players[opp];
        let mut seen: u16 = 0;
        for i in 0..me.up_len as usize {
            seen |= 1 << (me.up_cards[i] as u16 - 1);
        }
        seen |= 1 << (me.face_down as u16 - 1);
        for i in 0..op.up_len as usize {
            seen |= 1 << (op.up_cards[i] as u16 - 1);
        }
        let unseen = (!seen) & 0x7FF;
        let band =
            |lo: u32, hi: u32| ((unseen >> (lo - 1)) & ((1 << (hi - lo + 1)) - 1)).count_ones();
        let b0 = band(1, 3) as u64;
        let b1 = band(4, 6) as u64;
        let b2 = band(7, 9) as u64;
        let b3 = band(10, 11) as u64;
        let round = (self.round as u64).min(63);
        (round & 0x3F)
            | ((self.hearts[player] as u64 & 0x7) << 6)
            | ((self.hearts[opp] as u64 & 0x7) << 9)
            | ((me.total as u64 & 0x7F) << 12)
            | ((b0 & 0x3) << 19)
            | ((b1 & 0x3) << 21)
            | ((b2 & 0x3) << 23)
            | ((b3 & 0x3) << 25)
            | ((me.last_action_stand as u64) << 27)
            | ((op.last_action_stand as u64) << 28)
    }

    /// A packed god's-eye key sufficient to determine the value of the current
    /// in-round position under any fixed strategy keyed on [`Env::sufficient_key`].
    ///
    /// The value depends only on: the remaining deck, both players' totals and
    /// stood flags, both face-down cards (which fix each player's unseen set as
    /// `deck ∪ {opponent_face_down}`), whose turn it is, the consecutive-stand
    /// flag, and the hearts/round context. Card *order* and face-up identities
    /// beyond these are irrelevant. Only meaningful while a round is active.
    pub fn search_key(&self) -> u64 {
        let rs = match &self.round_state {
            Some(rs) => rs,
            None => return 0,
        };
        let p0 = &rs.players[0];
        let p1 = &rs.players[1];
        (rs.deck_mask as u64 & 0x7FF)
            | ((self.current_player as u64 & 0x1) << 11)
            | ((rs.prev_stand as u64) << 12)
            | ((p0.total as u64 & 0x7F) << 13)
            | ((p1.total as u64 & 0x7F) << 20)
            | ((p0.last_action_stand as u64) << 27)
            | ((p1.last_action_stand as u64) << 28)
            | ((p0.face_down as u64 & 0xF) << 29)
            | ((p1.face_down as u64 & 0xF) << 33)
            | (((self.round as u64).min(63) & 0x3F) << 37)
            | ((self.hearts[0] as u64 & 0x7) << 43)
            | ((self.hearts[1] as u64 & 0x7) << 46)
    }

    fn evaluate_winner(&self, rs: &RoundState) -> (Option<usize>, u8) {
        let a = &rs.players[0];
        let b = &rs.players[1];
        let a_over = a.total > TARGET;
        let b_over = b.total > TARGET;
        let damage = self.round;
        let winner = if a_over ^ b_over {
            if a_over { Some(1) } else { Some(0) }
        } else {
            let a_dist = a.total.abs_diff(TARGET);
            let b_dist = b.total.abs_diff(TARGET);
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

    /// Public up cards for a player in the current round.
    /// Returns a fixed-size array and its used length, or None if no round.
    pub fn public_up_cards(&self, player: usize) -> Option<([u8; 8], u8)> {
        let rs = self.round_state.as_ref()?;
        let pr = &rs.players[player];
        Some((pr.up_cards, pr.up_len))
    }

    /// The last round's down cards revealed at round end, if any.
    pub fn last_reveal(&self) -> Option<[u8; 2]> {
        self.last_reveal_down
    }

    /// The last round's public up cards (for both players) captured at round end.
    pub fn last_public_up(&self) -> Option<([u8; 8], u8, [u8; 8], u8)> {
        self.last_public_up
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic policy used by both engines so trajectories must match.
    fn policy(obs: &Observation) -> Action {
        if obs.deck_count == 0 {
            return Action::Stand;
        }
        let h = (obs.self_total as u32 * 7
            + obs.opp_face_up as u32 * 5
            + obs.round as u32 * 3
            + obs.deck_count as u32)
            % 10;
        if obs.self_total < 17 || h < 3 {
            Action::Draw
        } else {
            Action::Stand
        }
    }

    fn random_perm(rng: &mut XorShift64) -> [u8; NUM_CARDS] {
        let mut order = [0u8; NUM_CARDS];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = (i + 1) as u8;
        }
        for i in (1..NUM_CARDS).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        order
    }

    /// Play a full game via the preset-deck `step` path.
    fn play_preset(orders: &[[u8; NUM_CARDS]]) -> Vec<(u8, u8, u8, Option<usize>)> {
        let mut env = Env::new_with_preset_decks(orders.to_vec());
        let mut trace = Vec::new();
        while !env.is_game_over() {
            env.start_new_round().unwrap();
            loop {
                let p = env.current_player();
                let obs = env.observation(p);
                let res = env.step(policy(&obs)).unwrap();
                if res.round_over {
                    trace.push((
                        env.hearts(0),
                        env.hearts(1),
                        env.round(),
                        res.outcome.and_then(|o| o.winner),
                    ));
                    break;
                }
            }
        }
        trace
    }

    /// Play the same game via the controllable-chance API, consuming each round's
    /// deck order in sequence (4 deals, then draws).
    fn play_controlled(orders: &[[u8; NUM_CARDS]]) -> Vec<(u8, u8, u8, Option<usize>)> {
        let mut env = Env::new(0);
        let mut trace = Vec::new();
        let mut round_idx = 0;
        while !env.is_game_over() {
            let order = orders[round_idx];
            round_idx += 1;
            env.deal_specific([order[0], order[1], order[2], order[3]])
                .unwrap();
            let mut next = 4usize;
            loop {
                let p = env.current_player();
                let obs = env.observation(p);
                let res = match policy(&obs) {
                    Action::Draw => {
                        let c = order[next];
                        next += 1;
                        env.draw_specific(c).unwrap()
                    }
                    Action::Stand => env.stand().unwrap(),
                };
                if res.round_over {
                    trace.push((
                        env.hearts(0),
                        env.hearts(1),
                        env.round(),
                        res.outcome.and_then(|o| o.winner),
                    ));
                    break;
                }
            }
        }
        trace
    }

    #[test]
    fn controlled_matches_preset_engine() {
        let mut rng = XorShift64::seed(0xC0FFEE);
        for _ in 0..5000 {
            let orders: Vec<[u8; NUM_CARDS]> = (0..16).map(|_| random_perm(&mut rng)).collect();
            assert_eq!(play_preset(&orders), play_controlled(&orders));
        }
    }

    #[test]
    fn sufficient_key_ignores_seen_card_order() {
        // Two deals with the same per-player card *sets* but different deal/draw
        // identities that yield the same own-total and unseen-set must collapse
        // to the same key for the player to move.
        let mut a = Env::new(0);
        a.deal_specific([5, 9, 6, 2]).unwrap(); // p0 sees {5,6}, total 11; opp up {9}
        let mut b = Env::new(0);
        b.deal_specific([6, 9, 5, 2]).unwrap(); // p0 sees {6,5}, total 11; opp up {9}
        assert_eq!(a.sufficient_key(0), b.sufficient_key(0));

        // A different own total must give a different key.
        let mut c = Env::new(0);
        c.deal_specific([5, 9, 7, 2]).unwrap(); // p0 total 12
        assert_ne!(a.sufficient_key(0), c.sufficient_key(0));
    }
}
