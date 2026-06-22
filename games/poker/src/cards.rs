//! Cards, and a fast best-five-of-seven hand evaluator.
//!
//! A card is a `u8` in `0..52`: `rank = card / 4` (0 = deuce … 12 = ace),
//! `suit = card % 4`. Hand strength is a single `u32` that orders any two made
//! hands correctly — the category in the high nibble, then up to five rank
//! kickers packed four bits each, most significant first. Comparing two
//! `HandRank` values is exactly comparing the hands.

pub type Card = u8;

pub const NUM_CARDS: usize = 52;
pub const NUM_RANKS: usize = 13;
pub const RANK_CHARS: [char; NUM_RANKS] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
/// Suit order: clubs, diamonds, hearts, spades (matches the unicode below).
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

#[inline]
pub fn rank_of(card: Card) -> u8 {
    card / 4
}

#[inline]
pub fn suit_of(card: Card) -> u8 {
    card % 4
}

#[inline]
pub fn make_card(rank: u8, suit: u8) -> Card {
    rank * 4 + suit
}

/// `"As"`, `"Td"`, `"2c"` — the two-character form used in tests and logs.
pub fn card_str(card: Card) -> String {
    format!(
        "{}{}",
        RANK_CHARS[rank_of(card) as usize],
        SUIT_CHARS[suit_of(card) as usize]
    )
}

/// Parse `"As"`, `"td"`, `"2C"` (rank then suit, case-insensitive). Returns
/// `None` for anything that isn't a valid card.
pub fn parse_card(s: &str) -> Option<Card> {
    let s = s.trim();
    let mut chars = s.chars();
    let r = chars.next()?.to_ascii_uppercase();
    let su = chars.next()?.to_ascii_lowercase();
    if chars.next().is_some() {
        return None;
    }
    let rank = RANK_CHARS.iter().position(|&c| c == r)? as u8;
    let suit = SUIT_CHARS.iter().position(|&c| c == su)? as u8;
    Some(make_card(rank, suit))
}

/// Hand categories, ordered weakest to strongest; the value is the high field
/// of a [`HandRank`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Quads = 7,
    StraightFlush = 8,
}

/// A comparable strength: higher is better, and equal values are exact ties
/// (so split pots compare with `==`). The category sits above five 4-bit
/// kicker slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandRank(pub u32);

impl HandRank {
    fn build(cat: Category, kickers: [u8; 5]) -> HandRank {
        let mut v = (cat as u32) << 20;
        for (i, &k) in kickers.iter().enumerate() {
            v |= (k as u32) << (16 - 4 * i);
        }
        HandRank(v)
    }

    pub fn category(self) -> Category {
        match self.0 >> 20 {
            0 => Category::HighCard,
            1 => Category::Pair,
            2 => Category::TwoPair,
            3 => Category::Trips,
            4 => Category::Straight,
            5 => Category::Flush,
            6 => Category::FullHouse,
            7 => Category::Quads,
            _ => Category::StraightFlush,
        }
    }
}

/// The top-card rank index of the best straight in a 13-bit rank mask, or
/// `None`. The wheel (A-2-3-4-5) reports 3, the five.
fn straight_high(rank_mask: u16) -> Option<u8> {
    for top in (4..=12).rev() {
        let window = 0b11111u16 << (top - 4);
        if rank_mask & window == window {
            return Some(top as u8);
        }
    }
    let wheel = (1 << 12) | 0b1111;
    (rank_mask & wheel == wheel).then_some(3)
}

/// Evaluate the best five-card hand from any 5–7 cards. Panics in debug on
/// fewer than five.
pub fn evaluate(cards: &[Card]) -> HandRank {
    debug_assert!(cards.len() >= 5, "need at least five cards to evaluate");

    // Per-rank counts and per-suit rank masks in one pass.
    let mut rank_count = [0u8; NUM_RANKS];
    let mut suit_mask = [0u16; 4];
    let mut all_mask: u16 = 0;
    for &c in cards {
        let r = rank_of(c);
        let s = suit_of(c);
        rank_count[r as usize] += 1;
        suit_mask[s as usize] |= 1 << r;
        all_mask |= 1 << r;
    }

    // Flush (and straight flush) take priority over everything below a flush.
    let mut flush_suit = None;
    for (s, &m) in suit_mask.iter().enumerate() {
        if m.count_ones() >= 5 {
            flush_suit = Some(s);
            break;
        }
    }
    if let Some(s) = flush_suit {
        let fmask = suit_mask[s];
        if let Some(top) = straight_high(fmask) {
            return HandRank::build(Category::StraightFlush, [top, 0, 0, 0, 0]);
        }
        // Plain flush: the top five ranks of the flush suit.
        let top5 = top_n_ranks(fmask, 5);
        return HandRank::build(Category::Flush, top5);
    }

    // Group ranks by multiplicity, each group held high-rank-first.
    let mut quads = Vec::new();
    let mut trips = Vec::new();
    let mut pairs = Vec::new();
    for r in (0..NUM_RANKS).rev() {
        match rank_count[r] {
            4 => quads.push(r as u8),
            3 => trips.push(r as u8),
            2 => pairs.push(r as u8),
            _ => {}
        }
    }

    if let Some(&q) = quads.first() {
        let kick = best_kickers(&rank_count, &[q], 1)[0];
        return HandRank::build(Category::Quads, [q, kick, 0, 0, 0]);
    }

    // Full house: trips + a pair (or a second set of trips used as the pair).
    if let Some(&t) = trips.first() {
        let pair = trips.get(1).copied().or_else(|| pairs.first().copied());
        if let Some(p) = pair {
            return HandRank::build(Category::FullHouse, [t, p, 0, 0, 0]);
        }
    }

    if let Some(top) = straight_high(all_mask) {
        return HandRank::build(Category::Straight, [top, 0, 0, 0, 0]);
    }

    if let Some(&t) = trips.first() {
        let k = best_kickers(&rank_count, &[t], 2);
        return HandRank::build(Category::Trips, [t, k[0], k[1], 0, 0]);
    }

    if pairs.len() >= 2 {
        let (hi, lo) = (pairs[0], pairs[1]);
        let kick = best_kickers(&rank_count, &[hi, lo], 1)[0];
        return HandRank::build(Category::TwoPair, [hi, lo, kick, 0, 0]);
    }

    if let Some(&p) = pairs.first() {
        let k = best_kickers(&rank_count, &[p], 3);
        return HandRank::build(Category::Pair, [p, k[0], k[1], k[2], 0]);
    }

    HandRank::build(Category::HighCard, top_n_ranks(all_mask, 5))
}

/// The `n` highest set ranks of a 13-bit mask, high first, zero-padded to five.
fn top_n_ranks(mask: u16, n: usize) -> [u8; 5] {
    let mut out = [0u8; 5];
    let mut i = 0;
    for r in (0..NUM_RANKS as u8).rev() {
        if i == n {
            break;
        }
        if mask & (1 << r) != 0 {
            out[i] = r;
            i += 1;
        }
    }
    out
}

/// The `n` highest ranks not in `exclude`, high first, zero-padded to five.
/// Used to fill the kicker slots after the made-hand ranks are claimed.
fn best_kickers(rank_count: &[u8; NUM_RANKS], exclude: &[u8], n: usize) -> [u8; 5] {
    let mut out = [0u8; 5];
    let mut i = 0;
    for r in (0..NUM_RANKS as u8).rev() {
        if i == n {
            break;
        }
        if rank_count[r as usize] > 0 && !exclude.contains(&r) {
            out[i] = r;
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(cards: &str) -> HandRank {
        let cs: Vec<Card> = cards
            .split_whitespace()
            .map(|c| parse_card(c).unwrap())
            .collect();
        evaluate(&cs)
    }

    #[test]
    fn card_roundtrips_through_string() {
        for c in 0..NUM_CARDS as u8 {
            assert_eq!(parse_card(&card_str(c)), Some(c));
        }
        assert_eq!(parse_card("As"), Some(make_card(12, 3)));
        assert_eq!(parse_card("2c"), Some(0));
        assert_eq!(parse_card("zz"), None);
        assert_eq!(parse_card("A"), None);
    }

    #[test]
    fn categories_are_ordered() {
        let sf = hand("Ts Js Qs Ks As 2c 3d");
        let quads = hand("9c 9d 9h 9s Kc 2d 3h");
        let boat = hand("9c 9d 9h Kc Kd 2s 3h");
        let flush = hand("2s 5s 8s Js Ks 3d 4c");
        let straight = hand("5c 6d 7h 8s 9c 2d Kd");
        let trips = hand("9c 9d 9h Kc Qd 2s 3h");
        let two_pair = hand("9c 9d Kc Kd 2s 3h 4c");
        let pair = hand("9c 9d Kc Qd 2s 3h 4c");
        let high = hand("2c 4d 6h 8s Tc Qd Ks");
        let ranks = [
            high, pair, two_pair, trips, straight, flush, boat, quads, sf,
        ];
        for w in ranks.windows(2) {
            assert!(w[0] < w[1], "{:?} should be < {:?}", w[0], w[1]);
        }
        assert_eq!(sf.category(), Category::StraightFlush);
        assert_eq!(quads.category(), Category::Quads);
        assert_eq!(boat.category(), Category::FullHouse);
        assert_eq!(flush.category(), Category::Flush);
        assert_eq!(straight.category(), Category::Straight);
        assert_eq!(trips.category(), Category::Trips);
        assert_eq!(two_pair.category(), Category::TwoPair);
        assert_eq!(pair.category(), Category::Pair);
        assert_eq!(high.category(), Category::HighCard);
    }

    #[test]
    fn wheel_is_the_lowest_straight() {
        let wheel = hand("Ac 2d 3h 4s 5c 9d Kd");
        assert_eq!(wheel.category(), Category::Straight);
        let six_high = hand("2c 3d 4h 5s 6c 9d Kd");
        assert!(wheel < six_high, "the wheel is the weakest straight");
        // Ace-high straight beats everything below it.
        let broadway = hand("Tc Jd Qh Ks Ac 2d 3h");
        assert!(six_high < broadway);
    }

    #[test]
    fn wheel_straight_flush_beats_lower() {
        let steel = hand("As 2s 3s 4s 5s 9d Kd"); // 5-high straight flush
        assert_eq!(steel.category(), Category::StraightFlush);
        let six_sf = hand("2s 3s 4s 5s 6s 9d Kd");
        assert!(steel < six_sf);
    }

    #[test]
    fn kickers_break_ties() {
        // Same pair of kings, different kickers.
        let a = hand("Kc Kd Ah 7s 4c 2d 3h"); // kings, ace kicker
        let b = hand("Kc Kd Qh 7s 4c 2d 3h"); // kings, queen kicker
        assert!(a > b);
        // Identical best five from different hole cards: exact tie.
        let x = hand("Kc Kd 9h 7s 4c");
        let y = hand("Ks Kh 9d 7c 4d");
        assert_eq!(x, y, "identical hands must compare equal for split pots");
    }

    #[test]
    fn two_pair_uses_top_two_and_a_kicker() {
        // Three pairs available; only the top two plus best kicker count.
        let h = hand("Ac Ad Kc Kd Qc Qd 2s");
        assert_eq!(h.category(), Category::TwoPair);
        let lower = hand("Ac Ad Kc Kd Js 3s 2s");
        assert!(
            h > lower,
            "aces+kings with a Q kicker beats with a J kicker"
        );
    }

    #[test]
    fn full_house_from_two_trips_picks_higher_as_trips() {
        let h = hand("Ac Ad Ah Kc Kd Ks 2c");
        assert_eq!(h.category(), Category::FullHouse);
        // Aces full of kings beats kings full of aces.
        let other = hand("Ac Ad Ah Kc Kd 2s 3s");
        assert!(h >= other);
        let kings_full = hand("Kc Kd Ks Ac Ad 2s 3s");
        assert!(h > kings_full, "AAA KK should beat KKK AA");
    }

    #[test]
    fn flush_beats_straight_even_with_higher_top() {
        let flush = hand("2s 5s 8s 9s Ks 3d 4c");
        let straight = hand("9c Td Jh Qs Kc 2d 4d");
        assert!(flush > straight);
    }

    /// Reference evaluator: the max over all five-card subsets of seven cards,
    /// each scored by an independent simple scorer. Used to cross-check the
    /// categorical [`evaluate`] on random hands.
    mod reference {
        use super::super::*;

        /// Score exactly five cards. Category-based but written straightforwardly
        /// (no shared code with `evaluate`), so agreement is real evidence.
        fn score5(cards: &[Card; 5]) -> u32 {
            let mut rc = [0u8; NUM_RANKS];
            let mut suits = [0u8; 4];
            for &c in cards {
                rc[rank_of(c) as usize] += 1;
                suits[suit_of(c) as usize] += 1;
            }
            let flush = suits.contains(&5);
            let mask: u16 = cards.iter().fold(0, |m, &c| m | (1 << rank_of(c)));
            let straight = straight_top(mask);
            // Sorted (count, rank) descending: groups first, then high cards.
            let mut groups: Vec<(u8, u8)> = (0..NUM_RANKS as u8)
                .filter(|&r| rc[r as usize] > 0)
                .map(|r| (rc[r as usize], r))
                .collect();
            groups.sort_by(|a, b| b.cmp(a));
            let counts: Vec<u8> = groups.iter().map(|g| g.0).collect();
            let cat = if straight.is_some() && flush {
                Category::StraightFlush
            } else if counts == [4, 1] {
                Category::Quads
            } else if counts == [3, 2] {
                Category::FullHouse
            } else if flush {
                Category::Flush
            } else if straight.is_some() {
                Category::Straight
            } else if counts == [3, 1, 1] {
                Category::Trips
            } else if counts == [2, 2, 1] {
                Category::TwoPair
            } else if counts == [2, 1, 1, 1] {
                Category::Pair
            } else {
                Category::HighCard
            };
            let mut v = (cat as u32) << 20;
            let kickers: Vec<u8> = match cat {
                Category::Straight | Category::StraightFlush => vec![straight.unwrap()],
                _ => groups.iter().map(|g| g.1).collect(),
            };
            for (i, &k) in kickers.iter().take(5).enumerate() {
                v |= (k as u32) << (16 - 4 * i);
            }
            v
        }

        fn straight_top(mask: u16) -> Option<u8> {
            for top in (4..=12u8).rev() {
                let w = 0b11111u16 << (top - 4);
                if mask & w == w {
                    return Some(top);
                }
            }
            let wheel = (1 << 12) | 0b1111;
            (mask & wheel == wheel).then_some(3)
        }

        pub fn best7(cards: &[Card; 7]) -> u32 {
            let mut best = 0;
            for i in 0..7 {
                for j in (i + 1)..7 {
                    let five: Vec<Card> = (0..7)
                        .filter(|&k| k != i && k != j)
                        .map(|k| cards[k])
                        .collect();
                    let arr: [Card; 5] = five.try_into().unwrap();
                    best = best.max(score5(&arr));
                }
            }
            best
        }
    }

    #[test]
    fn categorical_evaluator_matches_brute_force() {
        use game_core::Rng;
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..50_000 {
            // Deal seven distinct cards.
            let mut deck: Vec<Card> = (0..NUM_CARDS as u8).collect();
            for i in (1..deck.len()).rev() {
                deck.swap(i, rng.below(i + 1));
            }
            let seven: [Card; 7] = deck[..7].try_into().unwrap();
            let fast = evaluate(&seven);
            let slow = reference::best7(&seven);
            // The two pack kickers identically, so the full u32 must match.
            assert_eq!(
                fast.0,
                slow,
                "mismatch on {:?}: fast {:#x} vs brute {:#x}",
                seven.iter().map(|&c| card_str(c)).collect::<Vec<_>>(),
                fast.0,
                slow
            );
        }
    }
}
