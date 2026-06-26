//! Reference standard Liar's Dice (2 players) for the paper Table-2 gate.
//!
//! Rules: each player rolls `num_dice` dice over `num_faces` faces. The top face
//! (index `num_faces - 1`) is WILD and counts toward every bid. Players alternate
//! bidding strictly higher bids until someone calls liar, which resolves the
//! single round. Bids are ordered by `id = (qty - 1) * faces + face`; a higher
//! id is a strictly higher bid.
//!
//! This is one round only — there is no die dropping or re-roll — so
//! [`PublicState::first_round`] is left `true` throughout and unused. A call is
//! encoded by setting `turn` to the out-of-range sentinel `players`, leaving the
//! challenged bid in `bid` and its owner in `last_bidder` (in two players the
//! challenger is `1 - last_bidder`).

use crate::rebel::game::{Bid, RebelGame};
use crate::rebel::hands::{self, Hand};
use crate::rebel::pbs::{Belief, MAX_SEATS, PublicState};

pub struct StandardLiarsDice {
    pub num_dice: u8,
    pub num_faces: u8,
}

impl StandardLiarsDice {
    pub fn new(num_dice: u8, num_faces: u8) -> Self {
        assert!(num_dice >= 1 && num_dice as usize <= hands::MAX_DICE);
        assert!(num_faces >= 2 && num_faces as usize <= hands::MAX_FACES);
        Self {
            num_dice,
            num_faces,
        }
    }

    fn wild_face(&self) -> u8 {
        self.num_faces - 1
    }

    /// Dice in `hand` that count toward a bid on `face`: the face itself plus the
    /// wild face (unless `face` is the wild face, which is not double-counted).
    fn num_matches(&self, hand: &Hand, face: u8) -> u8 {
        let wild = self.wild_face();
        if face == wild {
            hand[face as usize]
        } else {
            hand[face as usize] + hand[wild as usize]
        }
    }

    fn bid_id(&self, qty: u8, face: u8) -> u32 {
        (u32::from(qty) - 1) * u32::from(self.num_faces) + u32::from(face)
    }

    fn max_qty(&self) -> u8 {
        2 * self.num_dice
    }
}

impl RebelGame for StandardLiarsDice {
    fn players(&self) -> usize {
        2
    }

    fn faces(&self) -> u8 {
        self.num_faces
    }

    fn dice_left(&self) -> [u8; 8] {
        let mut d = [0u8; MAX_SEATS];
        d[0] = self.num_dice;
        d[1] = self.num_dice;
        d
    }

    fn root(&self) -> PublicState {
        PublicState {
            players: 2,
            faces: self.num_faces,
            dice_left: self.dice_left(),
            bid: None,
            turn: 0,
            last_bidder: 1,
            first_round: true,
        }
    }

    fn acting(&self, p: &PublicState) -> usize {
        p.turn
    }

    fn is_terminal(&self, p: &PublicState) -> bool {
        p.turn >= p.players as usize
    }

    fn legal_actions(&self, p: &PublicState) -> Vec<Bid> {
        if self.is_terminal(p) {
            return Vec::new();
        }
        let cur = p.bid.map(|(q, f)| self.bid_id(q, f));
        let mut acts = Vec::new();
        for q in 1..=self.max_qty() {
            for f in 0..self.num_faces {
                if cur.is_none_or(|c| self.bid_id(q, f) > c) {
                    acts.push(Bid::Raise { qty: q, face: f });
                }
            }
        }
        if p.bid.is_some() {
            acts.push(Bid::Call);
        }
        acts
    }

    fn apply(&self, p: &PublicState, a: Bid) -> PublicState {
        let mut next = p.clone();
        match a {
            Bid::Raise { qty, face } => {
                next.bid = Some((qty, face));
                next.last_bidder = p.turn;
                next.turn = (p.turn + 1) % p.players as usize;
            }
            Bid::Call => {
                next.turn = p.players as usize;
            }
            Bid::CallExact => unreachable!("standard Liar's Dice has no exact call"),
        }
        next
    }

    fn terminal_cfv(&self, p: &PublicState, traverser: usize, belief: &Belief) -> Vec<f64> {
        let (qty, face) = p.bid.expect("a terminal state holds the challenged bid");
        let bidder = p.last_bidder;
        let opp = 1 - traverser;
        let opp_dice = p.dice_left[opp];

        let mut opp_dist = vec![0.0f64; opp_dice as usize + 1];
        for (hand, &b) in hands::enumerate(opp_dice, p.faces)
            .iter()
            .zip(&belief.per_seat[opp])
        {
            opp_dist[self.num_matches(hand, face) as usize] += b;
        }
        let mut suffix = vec![0.0f64; opp_dice as usize + 2];
        for t in (0..=opp_dice as usize).rev() {
            suffix[t] = suffix[t + 1] + opp_dist[t];
        }

        let my_dice = p.dice_left[traverser];
        hands::enumerate(my_dice, p.faces)
            .iter()
            .map(|hand| {
                let threshold = i64::from(qty) - i64::from(self.num_matches(hand, face));
                let p_true = if threshold <= 0 {
                    1.0
                } else if threshold > i64::from(opp_dice) {
                    0.0
                } else {
                    suffix[threshold as usize]
                };
                if traverser == bidder {
                    2.0 * p_true - 1.0
                } else {
                    1.0 - 2.0 * p_true
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rebel::hands;

    fn terminal_state(g: &StandardLiarsDice, qty: u8, face: u8, bidder: usize) -> PublicState {
        let mut p = g.root();
        p.bid = Some((qty, face));
        p.last_bidder = bidder;
        p.turn = p.players as usize;
        p
    }

    fn brute_terminal_cfv(
        g: &StandardLiarsDice,
        qty: u8,
        face: u8,
        bidder: usize,
        traverser: usize,
        opp_belief: &[f64],
    ) -> Vec<f64> {
        let my_hands = hands::enumerate(g.num_dice, g.num_faces);
        let opp_hands = hands::enumerate(g.num_dice, g.num_faces);
        my_hands
            .iter()
            .map(|mh| {
                let mut v = 0.0;
                for (oh, &b) in opp_hands.iter().zip(opp_belief) {
                    let total =
                        u32::from(g.num_matches(mh, face)) + u32::from(g.num_matches(oh, face));
                    let bid_true = total >= u32::from(qty);
                    let win = if traverser == bidder {
                        bid_true
                    } else {
                        !bid_true
                    };
                    v += b * if win { 1.0 } else { -1.0 };
                }
                v
            })
            .collect()
    }

    #[test]
    fn terminal_cfv_matches_brute_force() {
        for &(d, f) in &[(1u8, 4u8), (1, 5), (2, 3)] {
            let g = StandardLiarsDice::new(d, f);
            let n = hands::hand_count(d, f);
            let uniform = hands::prior(d, f);
            let mut skewed = vec![0.0; n];
            for (i, s) in skewed.iter_mut().enumerate() {
                *s = (i + 1) as f64;
            }
            let z: f64 = skewed.iter().sum();
            for s in skewed.iter_mut() {
                *s /= z;
            }
            for belief in [&uniform, &skewed] {
                for qty in 1..=g.max_qty() {
                    for face in 0..f {
                        for bidder in 0..2 {
                            for traverser in 0..2 {
                                let mut bel = Belief {
                                    per_seat: vec![hands::prior(d, f), hands::prior(d, f)],
                                };
                                bel.per_seat[1 - traverser] = belief.clone();
                                let p = terminal_state(&g, qty, face, bidder);
                                let got = g.terminal_cfv(&p, traverser, &bel);
                                let want =
                                    brute_terminal_cfv(&g, qty, face, bidder, traverser, belief);
                                assert_eq!(got.len(), want.len());
                                for (gi, wi) in got.iter().zip(&want) {
                                    assert!(
                                        (gi - wi).abs() < 1e-12,
                                        "{d}x{f} qty={qty} face={face} bidder={bidder} trav={traverser}: {gi} vs {wi}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn legal_actions_are_in_strictly_increasing_bid_order() {
        let g = StandardLiarsDice::new(2, 3);
        let mut p = g.root();
        // Opening: every bid, no call.
        let opening = g.legal_actions(&p);
        assert!(opening.iter().all(|a| matches!(a, Bid::Raise { .. })));
        let ids: Vec<u32> = opening
            .iter()
            .map(|a| match a {
                Bid::Raise { qty, face } => g.bid_id(*qty, *face),
                Bid::Call | Bid::CallExact => unreachable!(),
            })
            .collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(opening.len(), (g.max_qty() as usize) * g.num_faces as usize);

        // After a bid: strictly higher bids then a single call.
        p = g.apply(&p, Bid::Raise { qty: 1, face: 1 });
        let acts = g.legal_actions(&p);
        let cur = g.bid_id(1, 1);
        let raises: Vec<u32> = acts
            .iter()
            .filter_map(|a| match a {
                Bid::Raise { qty, face } => Some(g.bid_id(*qty, *face)),
                Bid::Call | Bid::CallExact => None,
            })
            .collect();
        assert!(raises.iter().all(|&id| id > cur));
        assert!(raises.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(acts.last(), Some(&Bid::Call));
    }

    #[test]
    fn apply_and_is_terminal_track_the_round() {
        let g = StandardLiarsDice::new(1, 6);
        let mut p = g.root();
        assert!(!g.is_terminal(&p));
        assert_eq!(g.acting(&p), 0);

        p = g.apply(&p, Bid::Raise { qty: 1, face: 0 });
        assert!(!g.is_terminal(&p));
        assert_eq!(g.acting(&p), 1);
        assert_eq!(p.last_bidder, 0);

        p = g.apply(&p, Bid::Raise { qty: 2, face: 0 });
        assert_eq!(g.acting(&p), 0);
        assert_eq!(p.last_bidder, 1);

        p = g.apply(&p, Bid::Call);
        assert!(g.is_terminal(&p));
        assert_eq!(p.bid, Some((2, 0)));
        assert_eq!(p.last_bidder, 1);
        assert!(g.legal_actions(&p).is_empty());
    }
}
