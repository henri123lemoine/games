//! Config-agnostic featurization and action vocabulary for the Liar's Dice
//! policy/value net (a [`solvers::azero::Mlp`]).
//!
//! One fixed-length input vector spans every supported configuration (players
//! 2..=8, dice/player up to [`MAX_DICE_PER`], faces 2..=[`MF`]) by rotating the
//! seats so the acting player is always reference index 0 and masking the unused
//! seats/faces to zero — so a single net generalizes over player count, dice
//! count, faces, and arbitrary mid-game dice vectors. The action head is a fixed
//! vocabulary (the four bidding moves plus the full `Open(q, f)` grid) evaluated
//! only on the legal subset via the MLP's masked softmax.

use game_core::{Agent, Game, Rng};
use solvers::azero::{InferCache, Mlp};

use crate::{Action, HIST_K, LdState, LiarsDice, MAX_FACES, MAX_PLAYERS};

/// Seats the featurization reserves room for (= [`MAX_PLAYERS`]).
pub const MP: usize = MAX_PLAYERS;
/// Faces the featurization reserves room for (= [`MAX_FACES`]).
pub const MF: usize = MAX_FACES;
/// Largest per-seat starting dice count the inputs are scaled against.
pub const MAX_DICE_PER: usize = 8;
/// Largest total dice the opening-bid vocabulary covers (`MP * MAX_DICE_PER`).
pub const MAXTOTAL: usize = MP * MAX_DICE_PER;

/// Size of the policy head: the four bidding actions plus an `Open(q, f)` for
/// every `q in 1..=MAXTOTAL`, `f in 1..=MF`.
pub const fn policy_len() -> usize {
    4 + MAXTOTAL * MF
}

/// Length of the [`encode`] feature vector — constant across all configs.
pub const fn feature_len() -> usize {
    MF        // A: self hand histogram
    + 1       // B: self dice
    + MP      // C: per-seat dice
    + MP      // D: per-seat alive
    + 3       // E: totals
    + 13      // F: current bid + derived belief features
    + 3       // G: position relative to the bid owner
    + (MP + 1)// H: per-seat endorsed face + endorsers of the bid face
    + 2       // I: config (faces, players)
    + (MP + 1) // J: per-seat raises this round (rotated) + total round bid-depth
}

/// Per raw-history token features used by [`history_encode`].
pub const HISTORY_TOKEN_WIDTH: usize = 10;
/// Extra features appended by the history-attention architecture: the full
/// recent-token window plus one deterministic attention-pooled summary token.
pub const HISTORY_EXTRA: usize = HIST_K * HISTORY_TOKEN_WIDTH + HISTORY_TOKEN_WIDTH;

/// Input width for the C13 history-attention policy/value variant. It keeps the
/// base information-state features and appends a compact tokenized view of the
/// recent public bid path plus an attention-pooled summary. The downstream MLP
/// is trained with the same PG/R-NaD objective as the standard net.
pub const fn history_feature_len() -> usize {
    feature_len() + HISTORY_EXTRA
}

/// Policy-head index for an action. `Open(q, f)` occupies the grid above the
/// four fixed bidding moves. Panics on a chance `Roll`.
pub fn action_index(a: Action) -> usize {
    match a {
        Action::RaiseQuantity => 0,
        Action::RaiseFace => 1,
        Action::CallLiar => 2,
        Action::CallExact => 3,
        Action::Open(q, f) => 4 + (q as usize - 1) * MF + (f as usize - 1),
        Action::Roll(_) => panic!("Roll is a chance action, not a policy action"),
    }
}

/// Policy-head indices of the legal actions at `s`, in `legal_actions` order.
pub fn support(game: &LiarsDice, s: &LdState) -> Vec<usize> {
    game.legal_actions(s)
        .into_iter()
        .map(action_index)
        .collect()
}

/// The legal actions and their policy indices as parallel vectors.
pub fn legal_actions_and_support(game: &LiarsDice, s: &LdState) -> (Vec<Action>, Vec<usize>) {
    let acts = game.legal_actions(s);
    let sup = acts.iter().map(|&a| action_index(a)).collect();
    (acts, sup)
}

/// Encode the information set observable to `player` as a fixed-length vector,
/// rotated so `player` is reference index 0 and the others follow in turn order.
pub fn encode(game: &LiarsDice, s: &LdState, player: usize) -> Vec<f32> {
    let p = game.players as usize;
    let faces = game.faces;
    let seat = |k: usize| (player + k) % p; // rotated seat at offset k

    let dice = &s.dice_left;
    let total: u32 = (0..p).map(|i| u32::from(dice[i])).sum();
    let my_dice = u32::from(dice[player]);
    let unknown = total - my_dice;
    let num_alive = (0..p).filter(|&i| dice[i] > 0).count();
    let (qty, face) = (s.qty, s.face);
    let my_face_count = if qty > 0 {
        u32::from(s.hands[player][face as usize - 1])
    } else {
        0
    };

    let mut x = Vec::with_capacity(feature_len());

    // A: self hand histogram (faces beyond `faces` are zero).
    for &c in &s.hands[player] {
        x.push(f32::from(c) / MAX_DICE_PER as f32);
    }
    // B: self dice.
    x.push(my_dice as f32 / MAX_DICE_PER as f32);
    // C: per-seat dice (index 0 = me).
    for k in 0..MP {
        x.push(if k < p {
            dice[seat(k)] as f32 / MAX_DICE_PER as f32
        } else {
            0.0
        });
    }
    // D: per-seat alive.
    for k in 0..MP {
        x.push(if k < p && dice[seat(k)] > 0 { 1.0 } else { 0.0 });
    }
    // E: totals.
    x.push(total as f32 / MAXTOTAL as f32);
    x.push(num_alive as f32 / MP as f32);
    x.push(unknown as f32 / MAXTOTAL as f32);
    // F: current bid + belief-style derived features.
    x.push(qty as f32 / MAXTOTAL as f32);
    x.push(if total > 0 {
        qty as f32 / total as f32
    } else {
        0.0
    });
    for f in 0..MF {
        x.push(if qty > 0 && f == face as usize - 1 {
            1.0
        } else {
            0.0
        });
    }
    x.push(my_face_count as f32 / MAX_DICE_PER as f32);
    let need = qty.saturating_sub(my_face_count as u8) as u32;
    x.push(if qty > 0 && unknown > 0 {
        (need as f32 / unknown as f32).min(1.0)
    } else {
        0.0
    });
    x.push(if qty > 0 {
        need as f32 / MAXTOTAL as f32
    } else {
        0.0
    });
    x.push(if s.first_round { 1.0 } else { 0.0 });
    x.push(if qty == 0 { 1.0 } else { 0.0 });
    // G: position relative to the bid owner.
    let last = s.last_bidder as usize;
    x.push(((player + p - last) % p) as f32 / p as f32);
    let live_between = if player == last {
        0
    } else {
        let mut cnt = 0;
        let mut q = (player + 1) % p;
        while q != last {
            if dice[q] > 0 {
                cnt += 1;
            }
            q = (q + 1) % p;
        }
        cnt
    };
    x.push(live_between as f32 / p as f32);
    x.push(if player == last { 1.0 } else { 0.0 });
    // H: per-seat endorsed face + how many live seats endorse the bid face.
    for k in 0..MP {
        x.push(if k < p {
            s.endorsed[seat(k)] as f32 / MF as f32
        } else {
            0.0
        });
    }
    let endorsers = if qty > 0 {
        (0..p)
            .filter(|&i| dice[i] > 0 && s.endorsed[i] == face)
            .count()
    } else {
        0
    };
    x.push(endorsers as f32 / p as f32);
    // I: config.
    x.push(f32::from(faces) / MF as f32);
    x.push(p as f32 / MP as f32);
    // J: structured round history. Per-seat number of bids made this round
    // (rotated so index 0 = me), conveying who has raised how often — the
    // signaling path the monotonic current bid alone omits — plus a scalar
    // round bid-depth = total raises this round.
    let mut depth = 0u32;
    for k in 0..MP {
        let r = if k < p {
            s.raises_this_round[seat(k)]
        } else {
            0
        };
        x.push(f32::from(r) / MAX_DICE_PER as f32);
        depth += u32::from(r);
    }
    x.push(depth as f32 / MAXTOTAL as f32);

    debug_assert_eq!(x.len(), feature_len());
    x
}

/// C13 architecture input: base features plus a compact bid-history attention
/// encoding. The raw recent action window is exposed oldest-to-newest; each
/// token is featurized independently, then a recency/current-bid-aware attention
/// summary is appended. The attention weights are deterministic, keeping the
/// artifact dependency-free while giving the learned head a direct architecture
/// variant focused on public bid path, not only aggregate per-seat counts.
pub fn history_encode(game: &LiarsDice, s: &LdState, player: usize) -> Vec<f32> {
    let mut x = encode(game, s, player);
    let mut pooled = [0.0f32; HISTORY_TOKEN_WIDTH];
    let mut weight_sum = 0.0f32;
    for (idx, &code) in s.raw_history().iter().enumerate() {
        let tok = history_token_features(code, game.faces, idx);
        let w = history_attention_weight(&tok, game, s);
        for (acc, &v) in pooled.iter_mut().zip(&tok) {
            *acc += w * v;
        }
        weight_sum += w;
        x.extend_from_slice(&tok);
    }
    if weight_sum > 0.0 {
        for v in &mut pooled {
            *v /= weight_sum;
        }
    }
    x.extend_from_slice(&pooled);
    debug_assert_eq!(x.len(), history_feature_len());
    x
}

fn history_token_features(code: u16, faces: u8, idx: usize) -> [f32; HISTORY_TOKEN_WIDTH] {
    let present = (code != 0) as u8 as f32;
    let recency = (idx + 1) as f32 / HIST_K as f32;
    let is_rq = (code == 1) as u8 as f32;
    let is_rf = (code == 2) as u8 as f32;
    let is_liar = (code == 3) as u8 as f32;
    let is_exact = (code == 4) as u8 as f32;
    let is_open = (code >= 5) as u8 as f32;
    let (q, f) = if code >= 5 {
        let off = code - 5;
        let q = off / u16::from(faces) + 1;
        let f = off % u16::from(faces) + 1;
        (q as f32 / MAXTOTAL as f32, f as f32 / MF as f32)
    } else {
        (0.0, 0.0)
    };
    let code_norm = (f32::from(code) / (5.0 + MAXTOTAL as f32 * MF as f32)).min(1.0);
    [
        present, recency, is_rq, is_rf, is_liar, is_exact, is_open, q, f, code_norm,
    ]
}

fn history_attention_weight(
    tok: &[f32; HISTORY_TOKEN_WIDTH],
    game: &LiarsDice,
    s: &LdState,
) -> f32 {
    if tok[0] == 0.0 {
        return 0.0;
    }
    let (_, face) = s.current_bid();
    let face_match = if face > 0 && tok[8] > 0.0 {
        let tok_face = (tok[8] * MF as f32).round() as u8;
        (tok_face == face) as u8 as f32
    } else {
        0.0
    };
    let pressure = f32::from(s.current_bid().0) / f32::from(game.players * game.dice).max(1.0);
    tok[1] * (1.0 + 0.5 * face_match + 0.25 * pressure * (tok[2] + tok[3] + tok[6]))
}

/// Distribution over the legal actions at `(state, player)` per the net's policy
/// head — usable as a [`solvers::Policy`] for exploitability measurement.
pub fn net_policy(
    net: &Mlp,
    cache: &InferCache,
    game: &LiarsDice,
    state: &LdState,
    player: usize,
) -> Vec<f64> {
    let sup = support(game, state);
    let x = encode(game, state, player);
    let (probs, _) = net.policy_value_cached(cache, &x, &sup);
    probs.iter().map(|&p| f64::from(p)).collect()
}

/// Distribution over legal actions for a history-attention net.
pub fn history_net_policy(
    net: &Mlp,
    cache: &InferCache,
    game: &LiarsDice,
    state: &LdState,
    player: usize,
) -> Vec<f64> {
    let sup = support(game, state);
    let x = history_encode(game, state, player);
    let (probs, _) = net.policy_value_cached(cache, &x, &sup);
    probs.iter().map(|&p| f64::from(p)).collect()
}

/// A Liar's Dice agent that plays the policy head of an [`Mlp`] over the
/// featurized information set — a single forward pass per decision.
pub struct NetAgent {
    net: Mlp,
    cache: InferCache,
}

impl NetAgent {
    pub fn new(net: Mlp) -> Self {
        assert_eq!(net.input_len(), feature_len(), "net input width");
        assert_eq!(net.policy_len(), policy_len(), "net policy width");
        let cache = net.infer_cache();
        Self { net, cache }
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::from_bytes(data)?))
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::load(path)?))
    }

    pub fn net(&self) -> &Mlp {
        &self.net
    }
}

impl Agent<LiarsDice> for NetAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let (acts, sup) = legal_actions_and_support(game, state);
        if acts.is_empty() {
            return 0;
        }
        let x = encode(game, state, player);
        let (probs, _) = self.net.policy_value_cached(&self.cache, &x, &sup);
        let weights: Vec<f64> = probs.iter().map(|&p| f64::from(p)).collect();
        if weights.iter().sum::<f64>() <= 0.0 {
            return 0;
        }
        rng.pick(&weights)
    }
}

/// Policy/value agent for the C13 bid-history architecture variant. It uses the
/// same policy vocabulary and checkpoint container as [`NetAgent`], but the MLP
/// input width is [`history_feature_len`] and inference runs over
/// [`history_encode`].
pub struct HistoryNetAgent {
    net: Mlp,
    cache: InferCache,
}

impl HistoryNetAgent {
    pub fn new(net: Mlp) -> Self {
        assert_eq!(
            net.input_len(),
            history_feature_len(),
            "history net input width"
        );
        assert_eq!(net.policy_len(), policy_len(), "history net policy width");
        let cache = net.infer_cache();
        Self { net, cache }
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::from_bytes(data)?))
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self::new(Mlp::load(path)?))
    }

    pub fn net(&self) -> &Mlp {
        &self.net
    }
}

impl Agent<LiarsDice> for HistoryNetAgent {
    fn act(&self, game: &LiarsDice, state: &LdState, player: usize, rng: &mut Rng) -> usize {
        let (acts, sup) = legal_actions_and_support(game, state);
        if acts.is_empty() {
            return 0;
        }
        let x = history_encode(game, state, player);
        let (probs, _) = self.net.policy_value_cached(&self.cache, &x, &sup);
        let weights: Vec<f64> = probs.iter().map(|&p| f64::from(p)).collect();
        if weights.iter().sum::<f64>() <= 0.0 {
            return 0;
        }
        rng.pick(&weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_core::Turn;

    fn rolled_state(game: &LiarsDice, hands: &[[u8; MAX_FACES]]) -> LdState {
        let mut s = game.initial_state();
        for &h in hands {
            game.apply(&mut s, Action::Roll(h));
        }
        s
    }

    #[test]
    fn feature_length_is_config_invariant() {
        for &(p, d, f) in &[(2u8, 2u8, 2u8), (3, 4, 6), (6, 8, 6), (5, 5, 6)] {
            let game = LiarsDice::new(p, d, f);
            let s = game.initial_state();
            assert_eq!(encode(&game, &s, 0).len(), feature_len());
        }
        // Mid-game state with an eliminated seat and a ragged dice vector.
        let game = LiarsDice::new(5, 8, 6);
        let mut s = game.initial_state();
        s.dice_left = [2, 0, 6, 8, 3, 0, 0, 0];
        s.qty = 4;
        s.face = 5;
        s.last_bidder = 2;
        s.turn = 3;
        assert_eq!(encode(&game, &s, 3).len(), feature_len());
    }

    #[test]
    fn history_feature_length_is_config_invariant() {
        for &(p, d, f) in &[(2u8, 2u8, 2u8), (3, 4, 6), (6, 8, 6), (5, 5, 6)] {
            let game = LiarsDice::new(p, d, f);
            let s = game.initial_state();
            assert_eq!(history_encode(&game, &s, 0).len(), history_feature_len());
        }
    }

    #[test]
    fn per_seat_block_starts_at_the_acting_player() {
        let game = LiarsDice::new(3, 4, 6);
        let mut s = game.initial_state();
        s.dice_left = [4, 3, 2, 0, 0, 0, 0, 0];
        let off = MF + 1; // start of block C
        for player in 0..3 {
            let x = encode(&game, &s, player);
            assert_eq!(
                x[off],
                s.dice_left[player] as f32 / MAX_DICE_PER as f32,
                "C[0] must be the acting player's own dice"
            );
        }
        // Two players see different self/rotated encodings.
        assert_ne!(encode(&game, &s, 0), encode(&game, &s, 1));
    }

    #[test]
    fn action_index_round_trips_and_support_matches_legal() {
        let game = LiarsDice::new(2, 2, 6);
        // Opening (free-open) state.
        let mut open = rolled_state(&game, &[[2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]);
        open.qty = 0;
        open.face = 0;
        let (acts, sup) = legal_actions_and_support(&game, &open);
        assert_eq!(
            sup,
            acts.iter().map(|&a| action_index(a)).collect::<Vec<_>>()
        );
        for &i in &sup {
            assert!(i < policy_len());
        }
        let mut sorted = sup.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), sup.len(), "indices distinct");

        // Non-opening state: support is a subset of the four bidding moves.
        let bidding = rolled_state(&game, &[[2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]);
        let sup2 = support(&game, &bidding);
        assert!(sup2.iter().all(|&i| i < 4));
    }

    #[test]
    fn masking_zeroes_unused_faces_seats_and_marks_opening() {
        let game = LiarsDice::new(2, 3, 3); // only 3 faces
        let mut s = game.initial_state();
        s.dice_left = [3, 3, 0, 0, 0, 0, 0, 0];
        s.qty = 0;
        s.face = 0;
        let x = encode(&game, &s, 0);
        // A: faces 4..6 of the hand histogram are zero.
        assert!(x[3..MF].iter().all(|&v| v == 0.0));
        // D: alive flags for absent seats (2..8) are zero.
        let alive_off = MF + 1 + MP;
        assert!(x[alive_off + 2..alive_off + MP].iter().all(|&v| v == 0.0));
        // F: opening node -> bid-face one-hot all zero, is_opening = 1.
        let bid_onehot_off = MF + 1 + MP + MP + 3 + 2;
        assert!(
            x[bid_onehot_off..bid_onehot_off + MF]
                .iter()
                .all(|&v| v == 0.0)
        );
        // is_opening is the last entry of block F.
        let is_opening_off = bid_onehot_off + MF + 3; // after one-hot + my_count,need_frac,need_raw,first_round
        assert_eq!(x[is_opening_off], 1.0);
        // J: at a fresh opening no live seat has bid, so every per-seat raise
        // count and the round bid-depth scalar are zero.
        let j_off = MF + 1 + MP + MP + 3 + 13 + 3 + (MP + 1) + 2;
        assert!(x[j_off..j_off + MP + 1].iter().all(|&v| v == 0.0));
        assert_eq!(j_off + MP + 1, feature_len(), "J is the final block");
    }

    #[test]
    fn block_j_rotates_per_seat_raise_counts() {
        // 3 players: seat 0 has bid twice this round, seat 2 once, seat 1 none.
        let game = LiarsDice::new(3, 4, 6);
        let mut s = rolled_state(
            &game,
            &[[4, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0]],
        );
        s.qty = 3;
        s.face = 4;
        s.raises_this_round = [2, 0, 1, 0, 0, 0, 0, 0];
        let p = game.players as usize;
        let j_off = MF + 1 + MP + MP + 3 + 13 + 3 + (MP + 1) + 2;
        // Viewed from each seat, J[0] is that seat's own raise count (rotated).
        for player in 0..p {
            let x = encode(&game, &s, player);
            assert_eq!(
                x[j_off],
                s.raises_this_round[player] as f32 / MAX_DICE_PER as f32,
                "J[0] is the acting seat's own raise count"
            );
            // Total round bid-depth (3 raises) is seat-invariant.
            assert!(
                (x[j_off + MP] - 3.0 / MAXTOTAL as f32).abs() < 1e-6,
                "round bid-depth scalar"
            );
        }
        // Seat 1's view: their two later neighbours (seats 2, 0) carry 1, 2.
        let x1 = encode(&game, &s, 1);
        assert_eq!(
            x1[j_off + 1],
            1.0 / MAX_DICE_PER as f32,
            "next seat = seat 2"
        );
        assert_eq!(x1[j_off + 2], 2.0 / MAX_DICE_PER as f32, "then seat 0");
    }

    #[test]
    fn derived_bid_features_are_correct() {
        let game = LiarsDice::new(2, 3, 6);
        // I hold two 5s; the live bid is 3x5. unknown = opp's 3 dice.
        let mut s = rolled_state(&game, &[[0, 0, 0, 0, 2, 1], [1, 1, 1, 0, 0, 0]]);
        s.qty = 3;
        s.face = 5;
        s.last_bidder = 1;
        s.turn = 0;
        let x = encode(&game, &s, 0);
        let my_count_off = MF + 1 + MP + MP + 3 + 2 + MF; // block F: after qty,qty/total,one-hot
        assert_eq!(
            x[my_count_off],
            2.0 / MAX_DICE_PER as f32,
            "my_count of bid face"
        );
        // need = 3 - 2 = 1; unknown = 3 -> need_frac = 1/3.
        assert!((x[my_count_off + 1] - 1.0 / 3.0).abs() < 1e-6, "need_frac");
    }

    #[test]
    fn net_agent_plays_only_legal_actions() {
        let net = Mlp::new(feature_len(), 64, policy_len(), 0xA11CE);
        let agent = NetAgent::new(net);
        for &(p, d, f) in &[(2u8, 2u8, 6u8), (3, 3, 6), (4, 5, 4)] {
            let game = LiarsDice::new(p, d, f);
            let mut rng = Rng::new(0x1234 + u64::from(p));
            for _ in 0..20 {
                let mut s = game.initial_state();
                let mut steps = 0;
                while !game.is_terminal(&s) {
                    steps += 1;
                    assert!(steps < 100_000);
                    match game.turn(&s) {
                        Turn::Chance => {
                            let o = game.chance_outcomes(&s);
                            let a = o[rng.below(o.len())].0;
                            game.apply(&mut s, a);
                        }
                        Turn::Player(pl) => {
                            let acts = game.legal_actions(&s);
                            let i = agent.act(&game, &s, pl, &mut rng);
                            assert!(i < acts.len(), "net agent must pick a legal action");
                            game.apply(&mut s, acts[i]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn history_net_agent_plays_only_legal_actions() {
        let net = Mlp::new(history_feature_len(), 16, policy_len(), 0xC13);
        let agent = HistoryNetAgent::new(net);
        let game = LiarsDice::new(3, 2, 4);
        let mut rng = Rng::new(0x5157);
        let mut s = game.initial_state();
        while matches!(game.turn(&s), Turn::Chance) {
            let a = game.sample_chance_action(&s, &mut rng);
            game.apply(&mut s, a);
        }
        let acts = game.legal_actions(&s);
        let i = agent.act(&game, &s, s.turn(), &mut rng);
        assert!(i < acts.len());
    }
}
