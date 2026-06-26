//! Deploy adapter: ONE round of the real (non-standard, multi-round, N-player)
//! Liar's Dice expressed as a [`RebelGame`] the vector-CFR solver can drive.
//!
//! The reference [`StandardLiarsDice`](crate::rebel::standard::StandardLiarsDice)
//! is a single-round 2-player game; the real [`LiarsDice`](crate::LiarsDice) has
//! relative-increment bids, [`CallExact`](crate::Action::CallExact), dice loss,
//! elimination, and re-rolls. This adapter models the *current round* as a
//! depth-limited subgame whose leaves are either the game ending (a die loss down
//! to one survivor → exact `±1` returns) or the round ending without the game
//! ending (a call resolved → the dice re-roll → the rest of the game collapses to
//! a scalar per seat, supplied by a [`ContinuationValue`]). The round dynamics
//! mirror [`LiarsDice`] exactly so the subgame is byte-identical to one real
//! round; the continuation closes it.
//!
//! ## Face indexing
//!
//! The real game numbers faces `1..=faces`; this adapter (like
//! [`Bid::Raise`] and [`hands`]) numbers them `0..faces`. A 0-based face `f`
//! here is the real game's face `f + 1`, and a seat's count of face `f` is the
//! 0-based array slot `hand[f]` — exactly the index
//! [`hands::face_count_marginal`] consumes, so every count computed here is in
//! the same convention as the belief vectors.
//!
//! ## Round-end encoding in [`PublicState`]
//!
//! A call ends the round; the resulting public state carries the challenged bid
//! in `bid` and its owner in `last_bidder`, exactly as the standard game. To
//! distinguish a liar call from an exact call without widening [`PublicState`],
//! the out-of-range `turn` sentinel does double duty: `turn == players` marks a
//! [`Bid::Call`] (liar) leaf and `turn == players + 1` marks a [`Bid::CallExact`]
//! leaf (both satisfy the `turn >= players` terminal test). The caller seat is
//! not stored: bids are placed in turn order, so the seat that called is always
//! the live seat immediately after the bid owner, recovered as
//! `next_alive(last_bidder)`.

use crate::rebel::game::{Bid, RebelGame};
use crate::rebel::hands::{self, hand_count};
use crate::rebel::pbs::{Belief, MAX_SEATS, PublicState};
use crate::subgame::ContinuationValue;

/// One round of the real Liar's Dice as a [`RebelGame`], with round-end calls
/// closed by a [`ContinuationValue`] over the post-round dice vector.
pub struct LiarsDiceAdapter<'a, C: ContinuationValue> {
    players: usize,
    faces: u8,
    dice_left: [u8; MAX_SEATS],
    opener: usize,
    first_round: bool,
    open_qty_cap: Option<u8>,
    cont: &'a C,
}

/// The strategically-relevant opening-quantity cap for a round of `total` dice
/// over `faces` faces: roughly twice the expected per-face count (`total/faces`)
/// plus a safety buffer, clamped to `total`. Openings above this — claiming far
/// more of a face than the dice could plausibly show — are dominated junk that a
/// Nash opener never plays, so capping the wide round-open action set there is
/// lossless (validated) while shrinking the depth-limited solve over it. At
/// 5p5d6f (`total=25`, `faces=6`) the expected per-face count is ~4.2 and this
/// gives `2*5 + buffer`, covering every opening equilibrium ever uses.
pub fn principled_open_cap(total: u8, faces: u8) -> u8 {
    const BUFFER: u8 = 3;
    let expected = total.div_ceil(faces);
    (2 * expected + BUFFER).min(total).max(1)
}

impl<'a, C: ContinuationValue> LiarsDiceAdapter<'a, C> {
    /// The round that opens with `dice_left` on the table (index = seat, `0` =
    /// eliminated), `opener` to act, over `faces` faces. `first_round` selects the
    /// opening convention exactly as [`LiarsDice`](crate::LiarsDice): the game's
    /// very first round forces a `1×1` open, every later round opens freely.
    pub fn new(
        players: usize,
        faces: u8,
        dice_left: [u8; MAX_SEATS],
        opener: usize,
        first_round: bool,
        cont: &'a C,
    ) -> Self {
        assert!((2..=MAX_SEATS).contains(&players));
        assert!(faces >= 2 && faces as usize <= hands::MAX_FACES);
        assert!(dice_left[opener] > 0, "the opener must be a live seat");
        Self {
            players,
            faces,
            dice_left,
            opener,
            first_round,
            open_qty_cap: None,
            cont,
        }
    }

    /// Restrict the round-open action set to openings of quantity `<= cap`
    /// (`None` = no cap = the full `1..=total` range). Faithful ReBeL action
    /// abstraction: only the wide free-open node is narrowed; mid-round relative
    /// raises, the forced first-round `1×1`, and every other mechanic are
    /// unchanged. See [`principled_open_cap`] for the deploy default.
    pub fn with_open_cap(mut self, cap: Option<u8>) -> Self {
        self.open_qty_cap = cap;
        self
    }

    /// [`with_open_cap`](Self::with_open_cap) at the [`principled_open_cap`] for
    /// this round's total dice and faces — the deploy data-gen / agent default.
    pub fn with_principled_open_cap(self) -> Self {
        let cap = principled_open_cap(self.total_dice(&self.dice_left), self.faces);
        self.with_open_cap(Some(cap))
    }

    fn total_dice(&self, dice: &[u8; MAX_SEATS]) -> u8 {
        dice[..self.players].iter().sum()
    }

    fn num_alive(&self, dice: &[u8; MAX_SEATS]) -> usize {
        dice[..self.players].iter().filter(|&&d| d > 0).count()
    }

    fn next_alive(&self, dice: &[u8; MAX_SEATS], from: usize) -> usize {
        let mut p = (from + 1) % self.players;
        while dice[p] == 0 {
            p = (p + 1) % self.players;
        }
        p
    }

    fn prev_alive(&self, dice: &[u8; MAX_SEATS], from: usize) -> usize {
        let mut p = (from + self.players - 1) % self.players;
        while dice[p] == 0 {
            p = (p + self.players - 1) % self.players;
        }
        p
    }

    /// The game return to `traverser` of a state with one seat left standing.
    fn game_over_return(&self, dice: &[u8; MAX_SEATS], traverser: usize) -> f64 {
        match (0..self.players).find(|&s| dice[s] > 0) {
            Some(w) if w == traverser => 1.0,
            _ => -1.0 / (self.players as f64 - 1.0),
        }
    }

    /// Value to `traverser` after `loser` drops a die: the exact game return if
    /// that ends the game, else the continuation over the post-loss dice opened by
    /// the loser (recovered to the next live seat if the loss eliminated them) —
    /// mirroring [`LiarsDice::resolve_after_call`](crate::LiarsDice), whose
    /// `next_opener` argument is exactly the die loser for every call.
    fn value_after_loss(&self, dice: &[u8; MAX_SEATS], loser: usize, traverser: usize) -> f64 {
        let mut next = *dice;
        next[loser] -= 1;
        if self.num_alive(&next) <= 1 {
            return self.game_over_return(&next, traverser);
        }
        let opener = if next[loser] > 0 {
            loser
        } else {
            self.next_alive(&next, loser)
        };
        self.cont
            .value(self.faces, &next[..self.players], opener, traverser)
    }

    /// Continuation value to `traverser` with the dice unchanged (a correct exact
    /// call loses no die), opened by `opener`.
    fn value_no_loss(&self, dice: &[u8; MAX_SEATS], opener: usize, traverser: usize) -> f64 {
        self.cont
            .value(self.faces, &dice[..self.players], opener, traverser)
    }
}

/// Discrete convolution `out[k] = Σ_{i+j=k} a[i]·b[j]`.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

impl<C: ContinuationValue> RebelGame for LiarsDiceAdapter<'_, C> {
    fn players(&self) -> usize {
        self.players
    }

    fn faces(&self) -> u8 {
        self.faces
    }

    fn dice_left(&self) -> [u8; 8] {
        self.dice_left
    }

    fn root(&self) -> PublicState {
        // The real game's first round forces a `1×1` open OWNED BY THE PHANTOM seat
        // (the seat before the opener); the opener then RESPONDS to that standing
        // bid — exactly `LiarsDice::initial_state` (seat 0 faces seat `players-1`'s
        // forced `1×1`). So a first-round root already carries the bid; only a
        // continuing round opens free. `(1, 0)` is the 0-based form of the real
        // `1×1` (face 1 → 0-based face 0).
        let (bid, last_bidder) = if self.first_round {
            (
                Some((1, 0)),
                (self.opener + self.players - 1) % self.players,
            )
        } else {
            (None, self.prev_alive(&self.dice_left, self.opener))
        };
        PublicState {
            players: self.players as u8,
            faces: self.faces,
            dice_left: self.dice_left,
            bid,
            turn: self.opener,
            last_bidder,
            first_round: self.first_round,
        }
    }

    fn acting(&self, p: &PublicState) -> usize {
        p.turn
    }

    fn is_terminal(&self, p: &PublicState) -> bool {
        p.turn >= self.players || self.num_alive(&p.dice_left) <= 1
    }

    fn legal_actions(&self, p: &PublicState) -> Vec<Bid> {
        if self.is_terminal(p) {
            return Vec::new();
        }
        let total = self.total_dice(&p.dice_left);
        match p.bid {
            // A free open: every `(qty, face)` up to the opening cap (no cap =
            // every quantity `1..=total`). The first round never reaches here — its
            // root already carries the phantom's forced `1×1`, so the opener
            // responds to a standing bid (the `Some` arm).
            None => {
                let qmax = self.open_qty_cap.map_or(total, |c| c.min(total));
                let mut acts = Vec::with_capacity(qmax as usize * self.faces as usize);
                for q in 1..=qmax {
                    for f in 0..self.faces {
                        acts.push(Bid::Raise { qty: q, face: f });
                    }
                }
                acts
            }
            Some((q, f)) => {
                let mut acts = Vec::with_capacity(4);
                if q < total {
                    acts.push(Bid::Raise {
                        qty: q + 1,
                        face: f,
                    });
                }
                if f + 1 < self.faces {
                    acts.push(Bid::Raise {
                        qty: q,
                        face: f + 1,
                    });
                } else if q < total {
                    acts.push(Bid::Raise {
                        qty: q + 1,
                        face: 0,
                    });
                }
                acts.push(Bid::Call);
                acts.push(Bid::CallExact);
                acts
            }
        }
    }

    fn apply(&self, p: &PublicState, a: Bid) -> PublicState {
        let mut next = p.clone();
        match a {
            Bid::Raise { qty, face } => {
                next.bid = Some((qty, face));
                next.last_bidder = p.turn;
                next.turn = self.next_alive(&p.dice_left, p.turn);
            }
            Bid::Call => next.turn = self.players,
            Bid::CallExact => next.turn = self.players + 1,
        }
        next
    }

    fn terminal_cfv(&self, p: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        let trav_dice = p.dice_left[traverser];
        let trav_hands = hand_count(trav_dice, self.faces);

        if p.turn < self.players {
            let v = self.game_over_return(&p.dice_left, traverser);
            return vec![v; trav_hands];
        }

        let is_exact = p.turn == self.players + 1;
        let (qty, face) = p.bid.expect("a round-end call carries the challenged bid");
        let qty = qty as usize;
        let face = face as usize;
        let bidder = p.last_bidder;
        let caller = self.next_alive(&p.dice_left, bidder);

        let mut others = vec![1.0f64];
        for seat in 0..self.players {
            if seat == traverser || p.dice_left[seat] == 0 {
                continue;
            }
            let marginal = hands::tables(p.dice_left[seat], self.faces)
                .face_marginal(&belief.per_seat[seat], face);
            others = convolve(&others, &marginal);
        }
        let others_max = others.len() - 1;

        let own_counts = &hands::tables(trav_dice, self.faces).hands;

        if is_exact {
            let value_exact = self.value_no_loss(&p.dice_left, caller, traverser);
            let value_miss = self.value_after_loss(&p.dice_left, caller, traverser);
            own_counts
                .iter()
                .map(|hand| {
                    let own = hand[face] as usize;
                    let need = qty as isize - own as isize;
                    let p_exact = if (0..=others_max as isize).contains(&need) {
                        others[need as usize]
                    } else {
                        0.0
                    };
                    p_exact * value_exact + (1.0 - p_exact) * value_miss
                })
                .collect()
        } else {
            let value_caller_loses = self.value_after_loss(&p.dice_left, caller, traverser);
            let value_bidder_loses = self.value_after_loss(&p.dice_left, bidder, traverser);
            let mut suffix = vec![0.0f64; others.len() + 1];
            for t in (0..others.len()).rev() {
                suffix[t] = suffix[t + 1] + others[t];
            }
            own_counts
                .iter()
                .map(|hand| {
                    let own = hand[face] as usize;
                    let threshold = qty as isize - own as isize;
                    let p_true = if threshold <= 0 {
                        1.0
                    } else if threshold as usize > others_max {
                        0.0
                    } else {
                        suffix[threshold as usize]
                    };
                    p_true * value_caller_loses + (1.0 - p_true) * value_bidder_loses
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::cfr::{CfrParams, CfrVariant, Solver};
    use crate::rebel::exploit::exploitability;
    use crate::rebel::leaf::TerminalLeaf;
    use crate::subgame::DiceShareValue;
    use crate::{FitConfig, fit_two_player};

    fn dice_vec(counts: &[u8]) -> [u8; MAX_SEATS] {
        let mut d = [0u8; MAX_SEATS];
        d[..counts.len()].copy_from_slice(counts);
        d
    }

    /// A round-end call leaf: the challenged `(qty, face)` (0-based face) owned by
    /// `bidder`, ended by a liar or exact call.
    fn call_leaf<C: ContinuationValue>(
        ad: &LiarsDiceAdapter<C>,
        qty: u8,
        face: u8,
        bidder: usize,
        exact: bool,
    ) -> PublicState {
        let mut p = ad.root();
        p.bid = Some((qty, face));
        p.last_bidder = bidder;
        p.turn = if exact { ad.players + 1 } else { ad.players };
        p
    }

    /// Brute-force per-hand terminal value for `traverser`: enumerate the full
    /// joint of every other live seat's hand (belief-weighted), resolve the call
    /// to the exact dice outcome, and average the traverser's value.
    fn brute_cfv<C: ContinuationValue>(
        ad: &LiarsDiceAdapter<C>,
        p: &PublicState,
        traverser: usize,
        belief: &Belief,
    ) -> Vec<f64> {
        let n = ad.players;
        let faces = ad.faces;
        let (qty, face) = p.bid.unwrap();
        let qty = qty as usize;
        let face = face as usize;
        let bidder = p.last_bidder;
        let caller = ad.next_alive(&p.dice_left, bidder);
        let is_exact = p.turn == n + 1;

        let others: Vec<usize> = (0..n)
            .filter(|&s| s != traverser && p.dice_left[s] > 0)
            .collect();
        let other_hands: Vec<Vec<_>> = others
            .iter()
            .map(|&s| hands::enumerate(p.dice_left[s], faces))
            .collect();

        hands::enumerate(p.dice_left[traverser], faces)
            .iter()
            .map(|th| {
                let own = th[face] as usize;
                let mut acc = 0.0;
                let mut idx = vec![0usize; others.len()];
                loop {
                    let mut weight = 1.0;
                    let mut others_count = 0usize;
                    for (k, &seat) in others.iter().enumerate() {
                        weight *= belief.per_seat[seat][idx[k]];
                        others_count += other_hands[k][idx[k]][face] as usize;
                    }
                    let total = own + others_count;
                    let value = if is_exact {
                        if total == qty {
                            ad.value_no_loss(&p.dice_left, caller, traverser)
                        } else {
                            ad.value_after_loss(&p.dice_left, caller, traverser)
                        }
                    } else {
                        let loser = if total < qty { bidder } else { caller };
                        ad.value_after_loss(&p.dice_left, loser, traverser)
                    };
                    acc += weight * value;

                    let mut k = others.len();
                    loop {
                        if k == 0 {
                            return acc;
                        }
                        k -= 1;
                        idx[k] += 1;
                        if idx[k] < other_hands[k].len() {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
            })
            .collect()
    }

    fn skewed(n: usize) -> Vec<f64> {
        let raw: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let z: f64 = raw.iter().sum();
        raw.iter().map(|x| x / z).collect()
    }

    fn belief_for(
        ad_players: usize,
        dice: &[u8; MAX_SEATS],
        faces: u8,
        skew_seat: usize,
    ) -> Belief {
        let per_seat = (0..ad_players)
            .map(|s| {
                let n = hand_count(dice[s], faces);
                if s == skew_seat && n > 1 {
                    skewed(n)
                } else {
                    hands::prior(dice[s], faces)
                }
            })
            .collect();
        Belief { per_seat }
    }

    #[test]
    fn terminal_cfv_matches_brute_force_two_player() {
        let cv = DiceShareValue;
        for &(faces, counts) in &[(3u8, [2u8, 2u8]), (4, [2, 1]), (3, [1, 2])] {
            let dice = dice_vec(&counts);
            let ad = LiarsDiceAdapter::new(2, faces, dice, 0, false, &cv);
            let total = counts[0] + counts[1];
            for skew in 0..2 {
                let belief = belief_for(2, &dice, faces, skew);
                for qty in 1..=total {
                    for face in 0..faces {
                        for bidder in 0..2 {
                            for exact in [false, true] {
                                let leaf = call_leaf(&ad, qty, face, bidder, exact);
                                for trav in 0..2 {
                                    let got = ad.terminal_cfv(&leaf, trav, &belief);
                                    let want = brute_cfv(&ad, &leaf, trav, &belief);
                                    assert_eq!(got.len(), want.len());
                                    for (g, w) in got.iter().zip(&want) {
                                        assert!(
                                            (g - w).abs() < 1e-9,
                                            "2p {faces}f qty={qty} face={face} bidder={bidder} \
                                             exact={exact} trav={trav}: {g} vs {w}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn terminal_cfv_matches_brute_force_three_player() {
        let cv = DiceShareValue;
        let faces = 3u8;
        for &counts in &[[2u8, 2u8, 2u8], [2, 1, 1], [1, 2, 1]] {
            let dice = dice_vec(&counts);
            let ad = LiarsDiceAdapter::new(3, faces, dice, 0, false, &cv);
            let total: u8 = counts.iter().sum();
            for skew in 0..3 {
                let belief = belief_for(3, &dice, faces, skew);
                for qty in 1..=total {
                    for face in 0..faces {
                        for (bidder, &bidder_dice) in dice.iter().enumerate().take(3) {
                            if bidder_dice == 0 {
                                continue;
                            }
                            for exact in [false, true] {
                                let leaf = call_leaf(&ad, qty, face, bidder, exact);
                                for trav in 0..3 {
                                    let got = ad.terminal_cfv(&leaf, trav, &belief);
                                    let want = brute_cfv(&ad, &leaf, trav, &belief);
                                    assert_eq!(got.len(), want.len());
                                    for (g, w) in got.iter().zip(&want) {
                                        assert!(
                                            (g - w).abs() < 1e-9,
                                            "3p {faces}f qty={qty} face={face} bidder={bidder} \
                                             exact={exact} trav={trav}: {g} vs {w}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn legal_actions_mirror_the_real_relative_raises() {
        let cv = DiceShareValue;
        let dice = dice_vec(&[2, 2]);
        let ad = LiarsDiceAdapter::new(2, 3, dice, 0, false, &cv);

        let root = ad.root();
        let opening = ad.legal_actions(&root);
        assert_eq!(opening.len(), 4 * 3, "every (qty 1..=total, face) opens");
        assert!(opening.iter().all(|a| matches!(a, Bid::Raise { .. })));

        // Mid-bid at 2x(face 1): RaiseQuantity -> 3x1, RaiseFace -> 2x2, then both calls.
        let mut p = root;
        p.bid = Some((2, 1));
        p.last_bidder = 0;
        p.turn = 1;
        assert_eq!(
            ad.legal_actions(&p),
            vec![
                Bid::Raise { qty: 3, face: 1 },
                Bid::Raise { qty: 2, face: 2 },
                Bid::Call,
                Bid::CallExact,
            ]
        );

        // Top face wraps quantity: at 2x(max face 2), RaiseFace -> 3x0.
        let mut wrap = ad.root();
        wrap.bid = Some((2, 2));
        wrap.last_bidder = 0;
        wrap.turn = 1;
        assert_eq!(
            ad.legal_actions(&wrap),
            vec![
                Bid::Raise { qty: 3, face: 2 },
                Bid::Raise { qty: 3, face: 0 },
                Bid::Call,
                Bid::CallExact,
            ]
        );

        // Maxed bid (qty == total, top face): only the two calls remain.
        let mut maxed = ad.root();
        maxed.bid = Some((4, 2));
        maxed.last_bidder = 0;
        maxed.turn = 1;
        assert_eq!(ad.legal_actions(&maxed), vec![Bid::Call, Bid::CallExact]);
    }

    #[test]
    fn open_cap_prunes_only_high_quantity_openings() {
        let cv = DiceShareValue;
        let dice = dice_vec(&[3, 3]);
        let ad = LiarsDiceAdapter::new(2, 4, dice, 0, false, &cv).with_open_cap(Some(2));
        let root = ad.root();
        let opening = ad.legal_actions(&root);
        // total = 6, faces = 4: full open is 6×4 = 24; cap 2 keeps q∈{1,2} → 2×4.
        assert_eq!(opening.len(), 2 * 4);
        assert!(
            opening
                .iter()
                .all(|a| matches!(a, Bid::Raise { qty, .. } if *qty <= 2))
        );
        // q=1 still opens for every face — the cap never removes the lowest open.
        assert!(opening.contains(&Bid::Raise { qty: 1, face: 0 }));

        // Mid-round responses off a standing bid are untouched by the cap.
        let mut p = root;
        p.bid = Some((2, 1));
        p.last_bidder = 0;
        p.turn = 1;
        let capped = LiarsDiceAdapter::new(2, 4, dice, 0, false, &cv).with_open_cap(Some(2));
        let uncapped = LiarsDiceAdapter::new(2, 4, dice, 0, false, &cv);
        assert_eq!(capped.legal_actions(&p), uncapped.legal_actions(&p));

        // No cap reproduces the full opening; the principled cap clamps to total.
        assert_eq!(uncapped.legal_actions(&uncapped.root()).len(), 6 * 4);
        assert_eq!(principled_open_cap(6, 4), 6);
        assert_eq!(principled_open_cap(25, 6), 13);
    }

    #[test]
    fn first_round_root_is_the_phantom_one_by_one_with_the_opener_responding() {
        // The real entry round: the phantom (seat `players-1`) owns the forced
        // `1×1` and the opener (seat 0) responds to it, mirroring
        // `LiarsDice::initial_state`.
        let cv = DiceShareValue;
        let dice = dice_vec(&[2, 2]);
        let ad = LiarsDiceAdapter::new(2, 3, dice, 0, true, &cv);
        let root = ad.root();
        assert_eq!(root.bid, Some((1, 0)), "0-based 1×1 stands at the root");
        assert_eq!(root.turn, 0, "the opener acts");
        assert_eq!(root.last_bidder, 1, "the phantom (prev seat) owns the bid");

        // The opener responds: relative raises off the 1×1, plus both calls. With
        // total 4 and 3 faces, RaiseQuantity → 2×1 (0-based (2,0)) and RaiseFace →
        // 1×2 (0-based (1,1)).
        assert_eq!(
            ad.legal_actions(&root),
            vec![
                Bid::Raise { qty: 2, face: 0 },
                Bid::Raise { qty: 1, face: 1 },
                Bid::Call,
                Bid::CallExact,
            ]
        );

        // The opener's first real raise becomes the standing bid it owns.
        let after = ad.apply(&root, Bid::Raise { qty: 2, face: 0 });
        assert_eq!(after.bid, Some((2, 0)));
        assert_eq!(after.last_bidder, 0);
        assert_eq!(ad.acting(&after), 1);
    }

    fn adapter_root_value<C: ContinuationValue>(
        ad: &LiarsDiceAdapter<C>,
        seat: usize,
        iters: usize,
    ) -> (f64, Vec<Vec<Vec<f64>>>) {
        let initial = Belief::uniform_prior(&ad.root());
        let terminal = TerminalLeaf::new(ad);
        let params = CfrParams {
            num_iters: iters,
            max_depth: u32::MAX,
            variant: CfrVariant::LinearCfr,
            alternating: true,
            cfr_avg: false,
        };
        let mut solver = Solver::new(ad, params, &terminal, initial.clone());
        solver.multistep();
        let prior = &initial.per_seat[seat];
        let value: f64 = solver
            .root_values_mean(seat)
            .iter()
            .zip(prior)
            .map(|(v, p)| v * p)
            .sum();
        (value, solver.average_strategy().to_vec())
    }

    /// EXACT-ORACLE GATE: the adapter solved against the converged exact lattice
    /// reproduces the proven per-round Nash value (GATE A) and is itself solved to
    /// equilibrium (GATE B). Heavy (fits the lattice at full precision); run with
    /// `cargo test -p liars-dice -- --ignored adapter`.
    #[test]
    #[ignore = "fits the exact lattice; run explicitly"]
    fn exact_oracle_gate_reproduces_lattice() {
        for &(dice, faces) in &[(1u8, 3u8), (1, 4), (2, 3)] {
            let fit = fit_two_player(dice, faces, FitConfig::default());
            for a in 1..=dice {
                for b in 1..=dice {
                    for opener in 0..2usize {
                        let exact = fit
                            .lattice
                            .get_two_player(&[a, b], opener)
                            .expect("lattice covers every (a,b,opener)");
                        let ad = LiarsDiceAdapter::new(
                            2,
                            faces,
                            dice_vec(&[a, b]),
                            opener,
                            false,
                            &fit.lattice,
                        );
                        let (v0, avg) = adapter_root_value(&ad, 0, 2048);
                        let gate_a = (v0 - exact).abs();
                        let gate_b = exploitability(&ad, &avg);
                        println!(
                            "{dice}d{faces}f [{a},{b}] opener={opener}: \
                             adapter_v0={v0:.5} exact={exact:.5} |Δ|={gate_a:.5} expl={gate_b:.5}"
                        );
                        assert!(
                            gate_a < 0.01,
                            "GATE A {dice}d{faces}f [{a},{b}] opener={opener}: \
                             adapter {v0} vs lattice {exact}, |Δ|={gate_a}"
                        );
                        assert!(
                            gate_b < 0.02,
                            "GATE B {dice}d{faces}f [{a},{b}] opener={opener}: \
                             exploitability {gate_b}"
                        );
                    }
                }
            }
        }
    }
}
