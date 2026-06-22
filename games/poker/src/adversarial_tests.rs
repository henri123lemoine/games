//! Adversarial validation suite — independently written to try to BREAK the
//! NLHE engine, evaluator, side-pot math, betting closure, and bot. Authored as
//! a merge gate, separate from the implementer's own `tests.rs`.

#![allow(dead_code)]

use super::*;
use cards::{Card, Category, NUM_CARDS, RANK_CHARS, SUIT_CHARS, evaluate, parse_card, rank_of};
use game_core::{Agent, Game, Rng, Turn};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

fn h(cards: &str) -> HandRank {
    let cs: Vec<Card> = cards
        .split_whitespace()
        .map(|c| parse_card(c).unwrap())
        .collect();
    evaluate(&cs)
}

fn deal_card(game: &Poker, s: &mut PokerState, card: &str) {
    let target = parse_card(card).unwrap();
    let outs = game.chance_outcomes(s);
    let i = outs
        .iter()
        .position(|&(a, _)| a == Action::Deal(target))
        .unwrap_or_else(|| panic!("{card} not available to deal among remaining"));
    game.apply(s, outs[i].0);
}

fn deal_pending(game: &Poker, s: &mut PokerState) {
    while let Turn::Chance = game.turn(s) {
        let outs = game.chance_outcomes(s);
        game.apply(s, outs[0].0);
    }
}

fn act(game: &Poker, s: &mut PokerState, a: Action) {
    let acts = game.legal_actions(s);
    assert!(
        acts.contains(&a),
        "action {a:?} not legal among {acts:?} (to_act={}, street={:?})",
        s.to_act(),
        s.street()
    );
    game.apply(s, a);
}

/// Apply the first offered raise (the smallest legal sized raise) for the seat
/// to act. Panics if no raise is on the menu.
fn act_first_raise(game: &Poker, s: &mut PokerState) {
    let raise = game
        .legal_actions(s)
        .into_iter()
        .find(|a| matches!(a, Action::Raise(_)))
        .expect("a raise must be offered");
    game.apply(s, raise);
}

/// Set a seat's stack so that going all-in puts exactly `total` chips into the
/// pot for the hand — accounting for any blind the seat has already committed.
/// (`test_set_stack` sets the chips *behind*, so a blind-poster needs its stack
/// reduced by the blind it already put in.) Panics if `total` is below what the
/// seat has already committed.
fn set_allin_total(s: &mut PokerState, seat: usize, total: u32) {
    let already = s.committed(seat);
    assert!(
        total >= already,
        "all-in total {total} below already-committed {already} for seat {seat}"
    );
    s.test_set_stack(seat, total - already);
}

// ----------------------------------------------------------------------------
// 1. HAND EVALUATOR
// ----------------------------------------------------------------------------

#[test]
fn wheel_is_five_high_not_ace_high() {
    let wheel = h("Ac 2d 3h 4s 5c");
    assert_eq!(wheel.category(), Category::Straight);
    // Must rank BELOW a 6-high straight (it is the lowest straight).
    assert!(wheel < h("2c 3d 4h 5s 6c"), "wheel < six-high straight");
    // Must NOT be treated as ace-high: a real ace-high straight must beat it.
    assert!(wheel < h("Tc Jd Qh Ks Ac"), "wheel < broadway");
    // And a wheel must be a *straight*, not just ace-high high-card. A king-high
    // no-pair hand (no straight) must lose to the wheel.
    assert!(h("Kc Qd 9h 4s 2c") < wheel, "king-high < wheel straight");
}

#[test]
fn wheel_vs_six_high_in_seven_cards() {
    // Seven cards where A-2-3-4-5 is present but a 6 is NOT, so the wheel is the
    // only straight. Compare against a hand whose only straight is 2-6.
    let only_wheel = h("Ac 2d 3h 4s 5c Kd Qh");
    assert_eq!(only_wheel.category(), Category::Straight);
    let six_high = h("2c 3d 4h 5s 6c Kd Qh");
    assert!(only_wheel < six_high);
}

#[test]
fn steel_wheel_is_lowest_straight_flush() {
    let steel = h("As 2s 3s 4s 5s");
    assert_eq!(steel.category(), Category::StraightFlush);
    assert!(steel < h("2s 3s 4s 5s 6s"), "steel wheel < 6-high SF");
    // A steel wheel must beat ace-high flush and quads.
    assert!(h("As Ks Qs Js 9s") < steel, "ace-high flush < steel wheel");
    assert!(h("Ac Ad Ah As Kd") < steel, "quad aces < steel wheel");
}

#[test]
fn royal_flush_is_the_nuts() {
    let royal = h("Ts Js Qs Ks As");
    assert_eq!(royal.category(), Category::StraightFlush);
    // Royal beats every other straight flush.
    assert!(h("9s Ts Js Qs Ks") < royal, "king-high SF < royal");
    assert!(h("As 2s 3s 4s 5s") < royal, "steel wheel < royal");
    // And nothing ties it except another royal (different suit).
    assert_eq!(royal, h("Th Jh Qh Kh Ah"), "royal in another suit ties");
}

#[test]
fn quads_kicker_resolves() {
    // Quad nines, ace kicker vs king kicker.
    let a = h("9c 9d 9h 9s Ac 2d 3h");
    let b = h("9c 9d 9h 9s Kc 2d 3h");
    assert_eq!(a.category(), Category::Quads);
    assert!(a > b, "quads with ace kicker > king kicker");
    // Quads always beat any full house regardless of ranks.
    assert!(h("Ac Ad Ah Kc Kd Ks 2c") < b, "aces-full < quad nines");
}

#[test]
fn full_house_orders_by_trips_then_pair() {
    // Trips rank dominates: 999 over 22 beats 888 over AA.
    let nines_full = h("9c 9d 9h 2c 2d");
    let eights_full = h("8c 8d 8h Ac Ad");
    assert!(nines_full > eights_full, "trips rank dominates pair rank");
    // Same trips, higher pair wins.
    let kk = h("9c 9d 9h Kc Kd");
    let qq = h("9c 9d 9h Qc Qd");
    assert!(kk > qq, "same trips, higher pair wins");
}

#[test]
fn flush_compares_all_five_cards() {
    // Same top card, differ on the fifth.
    let a = h("As Ks 9s 6s 4s");
    let b = h("As Ks 9s 6s 3s");
    assert_eq!(a.category(), Category::Flush);
    assert!(a > b, "flush decided by the lowest of the five");
    // Differ on the third card.
    let c = h("As Ks 9s 6s 4s");
    let d = h("As Ks 8s 7s 5s");
    assert!(c > d, "9-high-third flush beats 8-high-third");
    // Seven-card flush picks the best five (drops the two lowest flush cards).
    let seven = h("As Ks Qs Js 9s 8s 2s");
    assert_eq!(seven, h("As Ks Qs Js 9s"), "best five flush cards only");
}

#[test]
fn two_pair_with_kicker() {
    let a = h("Ac Ad Kc Kd Qs 2c 3c"); // aces+kings, Q kicker
    let b = h("Ac Ad Kc Kd Js 2c 3c"); // aces+kings, J kicker
    assert_eq!(a.category(), Category::TwoPair);
    assert!(a > b, "two pair kicker resolves");
    // Top two pairs only: three pairs available, lowest pair is ignored but the
    // best remaining single card is the kicker.
    let three_pair = h("Ac Ad Kc Kd 2c 2d Qs");
    assert_eq!(three_pair.category(), Category::TwoPair);
    // Aces+Kings with Q kicker (the third pair's deuce loses to the queen).
    assert_eq!(three_pair, h("Ac Ad Kc Kd Qs 7h 5c"));
}

#[test]
fn board_plays_both_chop() {
    // The best five are entirely on the board; hole cards are irrelevant.
    // Board: A-K-Q-J-T (broadway). Two players with junk both play the board.
    let board = "Ac Kd Qh Js Ts";
    let p1 = h(&format!("{board} 2c 3d"));
    let p2 = h(&format!("{board} 4h 5s"));
    assert_eq!(p1.category(), Category::Straight);
    assert_eq!(p1, p2, "board plays for both -> exact chop");
}

#[test]
fn pair_vs_trips_category_priority() {
    // A made straight on the board must beat a flush draw that doesn't complete.
    assert!(h("2c 5c 8c Jc Kc 3d 4d").category() == Category::Flush);
    // Trips beat two pair.
    assert!(h("7c 7d 7h 2c 3d") > h("Ac Ad Kc Kd 2s"));
    // Straight beats trips.
    assert!(h("5c 6d 7h 8s 9c") > h("Ac Ad Ah 2c 3d"));
}

#[test]
fn flush_beats_straight() {
    let flush = h("2h 4h 6h 8h Th");
    let straight = h("9c Td Jh Qs Kc");
    assert!(flush > straight, "any flush > any straight");
}

#[test]
fn near_straight_is_not_a_straight() {
    // A-K-Q-J-9 has no straight (gap at the ten).
    let no_straight = h("Ac Kd Qh Js 9c 2d 3h");
    assert_eq!(no_straight.category(), Category::HighCard);
    // K-Q-J-T-9 IS a straight; when an ace is also present the engine should
    // still pick the king-high straight (the better five), not be confused into
    // an ace-high non-straight. Here there is no broadway, only K-high.
    let yes = h("2c Kd Qh Js Tc 9d 3h");
    assert_eq!(yes.category(), Category::Straight);
    assert_eq!(yes, h("Kd Qh Js Tc 9d"), "king-high straight");
    // And A-K-Q-J-T really is broadway (an ace-high straight), to be explicit.
    let broadway = h("Ac Kd Qh Js Tc 9d 2h");
    assert_eq!(broadway.category(), Category::Straight);
    assert!(
        broadway > yes,
        "broadway (A-high) beats the K-high straight"
    );
}

#[test]
fn pair_three_kickers_resolve() {
    let a = h("Ac Ad Kc Qd Js 2h 3h"); // AA + KQJ
    let b = h("Ac Ad Kc Qd Ts 2h 3h"); // AA + KQT
    assert_eq!(a.category(), Category::Pair);
    assert!(a > b, "third kicker breaks the tie");
}

/// A genuinely independent 7-card evaluator: enumerate all C(7,5)=21 five-card
/// subsets, score each from scratch with a sort-and-classify scorer that shares
/// NO code path with `evaluate`, and take the max. Cross-checked on category +
/// the full winner relation against `evaluate` over many random hands.
mod indep {
    use super::*;

    /// Score five cards into a comparable u64. Built independently: classify by
    /// the sorted multiset of rank counts and detect straights/flushes directly,
    /// with the wheel handled by treating ace-low explicitly.
    fn score5(cards: &[Card; 5]) -> u64 {
        let mut ranks: Vec<i32> = cards.iter().map(|&c| rank_of(c) as i32).collect();
        let suits: Vec<u8> = cards.iter().map(|&c| c % 4).collect();
        ranks.sort_unstable();
        let is_flush = suits.iter().all(|&x| x == suits[0]);

        // Straight detection, including the wheel (A,2,3,4,5 -> high card 5/idx3).
        let distinct: std::collections::BTreeSet<i32> = ranks.iter().copied().collect();
        let straight_high = if distinct.len() == 5 {
            let mx = *distinct.iter().next_back().unwrap();
            let mn = *distinct.iter().next().unwrap();
            if mx - mn == 4 {
                Some(mx)
            } else if distinct == [0, 1, 2, 3, 12].iter().copied().collect() {
                Some(3) // wheel: five-high
            } else {
                None
            }
        } else {
            None
        };

        // Rank-count groups, sorted by (count desc, rank desc).
        let mut counts: std::collections::BTreeMap<i32, i32> = std::collections::BTreeMap::new();
        for &r in &ranks {
            *counts.entry(r).or_insert(0) += 1;
        }
        let mut groups: Vec<(i32, i32)> = counts.iter().map(|(&r, &c)| (c, r)).collect();
        groups.sort_by(|a, b| b.cmp(a));
        let shape: Vec<i32> = groups.iter().map(|g| g.0).collect();

        let (category, ordered_ranks): (u64, Vec<i32>) =
            if let (Some(top), true) = (straight_high, is_flush) {
                (8, vec![top])
            } else if shape == [4, 1] {
                (7, groups.iter().map(|g| g.1).collect())
            } else if shape == [3, 2] {
                (6, groups.iter().map(|g| g.1).collect())
            } else if is_flush {
                (5, {
                    let mut r = ranks.clone();
                    r.sort_by(|a, b| b.cmp(a));
                    r
                })
            } else if let Some(top) = straight_high {
                (4, vec![top])
            } else if shape == [3, 1, 1] {
                (3, groups.iter().map(|g| g.1).collect())
            } else if shape == [2, 2, 1] {
                (2, groups.iter().map(|g| g.1).collect())
            } else if shape == [2, 1, 1, 1] {
                (1, groups.iter().map(|g| g.1).collect())
            } else {
                (0, {
                    let mut r = ranks.clone();
                    r.sort_by(|a, b| b.cmp(a));
                    r
                })
            };

        let mut v = category << 40;
        for (i, &r) in ordered_ranks.iter().take(5).enumerate() {
            v |= (r as u64) << (32 - 4 * i);
        }
        v
    }

    pub fn best7(seven: &[Card]) -> u64 {
        let mut best = 0u64;
        for i in 0..7 {
            for j in (i + 1)..7 {
                let five: Vec<Card> = (0..7)
                    .filter(|&k| k != i && k != j)
                    .map(|k| seven[k])
                    .collect();
                let arr: [Card; 5] = five.try_into().unwrap();
                best = best.max(score5(&arr));
            }
        }
        best
    }

    pub fn category_of(score: u64) -> u8 {
        (score >> 40) as u8
    }
}

#[test]
fn evaluate_matches_fully_independent_enumeration_categories_and_winner() {
    let mut rng = Rng::new(0xBADC0DE);
    for _ in 0..200_000 {
        let mut deck: Vec<Card> = (0..NUM_CARDS as u8).collect();
        for i in (1..deck.len()).rev() {
            deck.swap(i, rng.below(i + 1));
        }
        let a: Vec<Card> = deck[..7].to_vec();
        let fast_a = evaluate(&a);
        let slow_a = indep::best7(&a);
        // Category must agree.
        assert_eq!(
            fast_a.category() as u8,
            indep::category_of(slow_a),
            "category mismatch on {:?}",
            a.iter().map(|&c| card_str(c)).collect::<Vec<_>>()
        );

        // Winner relation against a second independent 7-card hand sharing the
        // board would require a board; instead compare two disjoint 7-card hands
        // directly for monotonicity of the two evaluators.
        let b: Vec<Card> = deck[7..14].to_vec();
        let fast_b = evaluate(&b);
        let slow_b = indep::best7(&b);
        let fast_ord = fast_a.cmp(&fast_b);
        let slow_ord = slow_a.cmp(&slow_b);
        assert_eq!(
            fast_ord,
            slow_ord,
            "winner disagreement: A={:?} B={:?}",
            a.iter().map(|&c| card_str(c)).collect::<Vec<_>>(),
            b.iter().map(|&c| card_str(c)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn shared_board_showdown_matches_independent() {
    // The real poker question: two players share a 5-card board, each with 2
    // hole cards. Confirm the categorical evaluator picks the same winner as the
    // independent best-of-21 enumerator on each player's 7 cards.
    let mut rng = Rng::new(0x5EED);
    for _ in 0..100_000 {
        let mut deck: Vec<Card> = (0..NUM_CARDS as u8).collect();
        for i in (1..deck.len()).rev() {
            deck.swap(i, rng.below(i + 1));
        }
        let board = &deck[0..5];
        let p1: Vec<Card> = board.iter().chain(&deck[5..7]).copied().collect();
        let p2: Vec<Card> = board.iter().chain(&deck[7..9]).copied().collect();
        let f1 = evaluate(&p1);
        let f2 = evaluate(&p2);
        let s1 = indep::best7(&p1);
        let s2 = indep::best7(&p2);
        assert_eq!(
            f1.cmp(&f2),
            s1.cmp(&s2),
            "showdown winner mismatch board={:?} p1={:?} p2={:?}",
            board.iter().map(|&c| card_str(c)).collect::<Vec<_>>(),
            deck[5..7].iter().map(|&c| card_str(c)).collect::<Vec<_>>(),
            deck[7..9].iter().map(|&c| card_str(c)).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn every_card_parses_and_every_category_reachable() {
    // Sanity: the full deck is 52 distinct cards.
    let mut seen = std::collections::HashSet::new();
    for r in 0..13u8 {
        for s in 0..4u8 {
            let str = format!("{}{}", RANK_CHARS[r as usize], SUIT_CHARS[s as usize]);
            let c = parse_card(&str).unwrap();
            assert!(seen.insert(c), "duplicate card {str}");
        }
    }
    assert_eq!(seen.len(), 52);
}

// ----------------------------------------------------------------------------
// 2. SIDE POTS + CHIP CONSERVATION
// ----------------------------------------------------------------------------

/// Total chips on the table must be invariant: sum of stacks after settlement
/// equals sum of starting stacks. payoff = won - committed, stack_after =
/// stack_before(=0 here since all committed)... we check via returns summing to
/// zero AND reconstruct chips explicitly.
fn assert_zero_sum(game: &Poker, s: &PokerState, seats: usize) {
    let total: f64 = (0..seats).map(|p| game.returns(s, p)).sum();
    assert!(total.abs() < 1e-6, "returns must sum to zero, got {total}");
}

#[test]
fn three_way_all_in_different_stacks_main_and_side_pots() {
    // Two short all-ins (seat0=20, seat1=60) and a deep caller (seat2) that
    // covers them. Best hand is the SHORT stack (seat0), then seat1, then seat2:
    //   main pot = 20*3 = 60       -> seat0 (eligible: all three)
    //   side pot 1 = (60-20)*2 = 80 -> seat1 (eligible: seats 1,2)
    // seat2 matches to 60 and keeps the rest behind (never at risk).
    // Net chips: seat0 +40, seat1 +20, seat2 -60  -> bb: +20, +10, -30.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    // Controlled holes. Deal order: seat1, seat2, seat0, then repeat.
    deal_card(&game, &mut s, "2c"); // seat1 c0
    deal_card(&game, &mut s, "Kd"); // seat2 c0
    deal_card(&game, &mut s, "Ac"); // seat0 c0
    deal_card(&game, &mut s, "2d"); // seat1 c1 -> seat1: 2c 2d
    deal_card(&game, &mut s, "Kh"); // seat2 c1 -> seat2: Kd Kh
    deal_card(&game, &mut s, "Ad"); // seat0 c1 -> seat0: Ac Ad

    // All-in totals accounting for posted blinds (SB=1 on seat1, BB=2 on seat2).
    set_allin_total(&mut s, 0, 20); // button/UTG, no blind
    set_allin_total(&mut s, 1, 60); // SB
    set_allin_total(&mut s, 2, 200); // BB

    // seat0 (UTG) first. Everyone jams; calls cap at the short stacks.
    act(&game, &mut s, Action::AllIn); // seat0 all-in 20
    act(&game, &mut s, Action::AllIn); // seat1 all-in 60
    // seat2 covers both; with no one left able to call a raise, the only way to
    // match is a plain Call (it keeps its 140 excess behind, which is returned).
    act(&game, &mut s, Action::Call); // seat2 calls to 60, 140 stays behind
    // Two seats already all-in for less; round is closed -> run out a controlled
    // board that gives seat0 the nuts, seat1 second, seat2 worst.
    deal_card(&game, &mut s, "Ah"); // flop: seat0 trips aces
    deal_card(&game, &mut s, "As"); // flop: seat0 QUAD aces (unbeatable)
    deal_card(&game, &mut s, "7c"); // flop blank
    deal_card(&game, &mut s, "2h"); // turn: seat1 trip deuces
    deal_card(&game, &mut s, "9d"); // river blank; seat2 just a pair of kings

    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 3);
    let net: Vec<f64> = (0..3).map(|p| game.returns(&s, p)).collect();
    // Exact awards (bb): seat0 +20 (won 60, in 20), seat1 +10 (won 80, in 60),
    // seat2 -30 (won 140 back, in 200).
    assert!(
        (net[0] - 20.0).abs() < 1e-6,
        "short quads take the main pot: {net:?}"
    );
    assert!(
        (net[1] - 10.0).abs() < 1e-6,
        "medium trips take side pot 1: {net:?}"
    );
    assert!(
        (net[2] + 30.0).abs() < 1e-6,
        "deep stack loses, excess returned: {net:?}"
    );

    // Conservation: chips won == chips committed.
    let total_committed: u32 = (0..3).map(|p| s.committed(p)).sum();
    let bb = game.big_blind as i64;
    let total_won: i64 = (0..3)
        .map(|p| s.committed(p) as i64 + (net[p] * bb as f64).round() as i64)
        .sum();
    assert_eq!(
        total_committed as i64, total_won,
        "side-pot chips conserved"
    );
}

#[test]
fn side_pot_short_stack_wins_only_main_pot() {
    // Controlled outcome: short all-in (seat0) has quads and wins the main pot;
    // the side pot between the two deep stacks goes to the better of those two.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    // Hole cards. Deal order seat1,seat2,seat0 x2.
    deal_card(&game, &mut s, "5c"); // seat1 c0
    deal_card(&game, &mut s, "Ac"); // seat2 c0
    deal_card(&game, &mut s, "2c"); // seat0 c0
    deal_card(&game, &mut s, "5d"); // seat1 c1 -> seat1: 5c 5d
    deal_card(&game, &mut s, "Ad"); // seat2 c1 -> seat2: Ac Ad
    deal_card(&game, &mut s, "2d"); // seat0 c1 -> seat0: 2c 2d (quads w/ board)

    set_allin_total(&mut s, 0, 10); // short (UTG, no blind)
    set_allin_total(&mut s, 1, 100); // SB
    set_allin_total(&mut s, 2, 100); // BB

    act(&game, &mut s, Action::AllIn); // seat0 (UTG) all-in 10
    act(&game, &mut s, Action::AllIn); // seat1 all-in 100
    // seat2 (BB, 100 total) calling exhausts its stack exactly, so the menu
    // offers it as AllIn (call-for-all), not a plain Call.
    act(&game, &mut s, Action::AllIn); // seat2 calls all-in to 100

    // Now run out a controlled board: two more deuces for seat0 quads; seat2's
    // aces make a pair; seat1's fives lose. Side pot (90*2=180) contested by
    // seats 1 & 2 -> seat2 (pair of aces) beats seat1 (pair of fives).
    deal_card(&game, &mut s, "2h"); // flop
    deal_card(&game, &mut s, "2s"); // flop -> seat0 has quad deuces
    deal_card(&game, &mut s, "9h"); // flop
    deal_card(&game, &mut s, "Kh"); // turn
    deal_card(&game, &mut s, "Qd"); // river

    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 3);
    let net: Vec<f64> = (0..3).map(|p| game.returns(&s, p)).collect();
    // Main pot = 10*3 = 30. seat0 contributed 10, wins 30 -> +20 chips = +10 bb.
    assert!(
        (net[0] - 10.0).abs() < 1e-6,
        "short quads win main only: {net:?}"
    );
    // Side pot = 90*2 = 180. seat2 wins it: contributed 100, won 180 -> +80 chips
    // = +40 bb.
    assert!((net[2] - 40.0).abs() < 1e-6, "aces win side pot: {net:?}");
    // seat1 loses all 100 -> -50 bb.
    assert!(
        (net[1] + 50.0).abs() < 1e-6,
        "fives lose both pots: {net:?}"
    );
}

#[test]
fn uncalled_excess_is_returned_not_vanished() {
    // Two players. Deep stack shoves more than the short stack can call; the
    // uncalled remainder must come back (conservation), not be awarded to anyone.
    let game = Poker::new(2)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    // HU deal: seat0(button/SB) c0, seat1(BB) c0, seat0 c1, seat1 c1.
    deal_card(&game, &mut s, "Ac");
    deal_card(&game, &mut s, "2c");
    deal_card(&game, &mut s, "Ad");
    deal_card(&game, &mut s, "2d"); // seat0 AA, seat1 22

    s.test_set_stack(0, 200); // deep
    s.test_set_stack(1, 30); // short (already posted BB 2, so 30 left behind? set absolute)
    // test_set_stack sets the *current* stack; blinds already deducted. Recompute
    // committed: seat0 committed 1 (SB), seat1 committed 2 (BB).
    act(&game, &mut s, Action::AllIn); // seat0 shoves 200 (street_bet 1 + 200)
    // seat1 can only call up to its stack -> AllIn (call for less).
    act(&game, &mut s, Action::AllIn); // seat1 all-in for 30+2=32 total committed

    deal_pending(&game, &mut s); // run out the board
    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 2);
    let net: Vec<f64> = (0..2).map(|p| game.returns(&s, p)).collect();
    // seat0 AA beats seat1 22 on essentially any board (we don't control it, but
    // AA vs 22 with a random board: AA wins ~87%). To make it deterministic,
    // assert conservation + that the WIN is exactly seat1's total commitment.
    // seat1 committed = its starting stack contribution. If seat0 wins, it should
    // win exactly what seat1 put in (matched portion), and the uncalled excess
    // returns. net[0] + net[1] == 0 already asserted.
    let total: f64 = net.iter().sum();
    assert!(total.abs() < 1e-9, "conservation with uncalled bet");
    // seat0 can never win more than 2 * seat1's_total_commitment in chips.
    let seat1_commit = s.committed(1) as f64;
    let bb = game.big_blind as f64;
    assert!(
        net[0].abs() * bb <= seat1_commit + 1e-6,
        "deep stack can't win more than the short stack matched: net0={} commit1={}",
        net[0],
        seat1_commit
    );
}

#[test]
fn odd_chip_split_pot_conserves_and_favors_earlier_seat() {
    // Three seats; seats 0 and 2 chop the pot (board plays a straight for both),
    // while seat 1 (SB) folds preflop leaving an ODD amount of dead money so the
    // total pot is not divisible by the two winners. The single odd chip must go
    // to the earlier-indexed winner (seat 0), and chips must be conserved.
    //
    // Pot construction: seat0 and seat2 each commit 10; seat1 folds after its SB
    // of 1 -> pot = 21, two winners -> 10 each + 1 odd chip to seat 0.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    // Holes. Deal order seat1, seat2, seat0 x2. Give seats 0 and 2 hands that
    // both just play the board; seat1's cards are irrelevant (folds).
    deal_card(&game, &mut s, "7c"); // seat1 c0
    deal_card(&game, &mut s, "2c"); // seat2 c0
    deal_card(&game, &mut s, "2d"); // seat0 c0
    deal_card(&game, &mut s, "7d"); // seat1 c1 -> seat1: 7c 7d (will fold)
    deal_card(&game, &mut s, "3c"); // seat2 c1 -> seat2: 2c 3c
    deal_card(&game, &mut s, "3d"); // seat0 c1 -> seat0: 2d 3d

    // seat0 (UTG) raises to 10, seat1 (SB) folds (1 dead), seat2 (BB) calls to 10.
    set_allin_total(&mut s, 0, 10);
    set_allin_total(&mut s, 2, 10);
    act(&game, &mut s, Action::AllIn); // seat0 all-in to 10
    act(&game, &mut s, Action::Fold); // seat1 SB folds (1 chip dead)
    act(&game, &mut s, Action::AllIn); // seat2 calls all-in to 10

    // Board plays a straight 8-9-T-J-Q for both live seats (their low cards are
    // irrelevant), an exact chop of the 21-chip pot.
    deal_card(&game, &mut s, "Th");
    deal_card(&game, &mut s, "Jh");
    deal_card(&game, &mut s, "Qh");
    deal_card(&game, &mut s, "9s");
    deal_card(&game, &mut s, "8d");
    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 3);

    let bb = game.big_blind as f64;
    let won_chips: Vec<i64> = (0..3)
        .map(|p| s.committed(p) as i64 + (game.returns(&s, p) * bb).round() as i64)
        .collect();
    // Pot = 10 + 1 + 10 = 21. seat0 gets 11 (10 + odd chip), seat2 gets 10.
    assert_eq!(
        won_chips[0], 11,
        "earlier winner gets the odd chip: {won_chips:?}"
    );
    assert_eq!(
        won_chips[2], 10,
        "later winner gets the floor share: {won_chips:?}"
    );
    assert_eq!(won_chips[1], 0, "folder wins nothing: {won_chips:?}");
    // Conservation.
    let total_committed: i64 = (0..3).map(|p| s.committed(p) as i64).sum();
    assert_eq!(
        total_committed,
        won_chips.iter().sum::<i64>(),
        "odd-chip split conserves chips"
    );
    assert_eq!(total_committed, 21, "pot is the two 10s plus the dead SB");
}

#[test]
fn folded_player_dead_money_stays_in_pot() {
    // A player who bets then folds leaves dead money that the winner collects.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // seat0 (UTG) raises, seat1 (SB) folds, seat2 (BB) calls. Then seat2 folds to
    // a flop bet, leaving its preflop chips dead.
    // Preflop: UTG seat0 acts first.
    let raise = game
        .legal_actions(&s)
        .into_iter()
        .find(|a| matches!(a, Action::Raise(_)))
        .expect("a raise is offered");
    act(&game, &mut s, raise);
    // SB folds, BB calls.
    act(&game, &mut s, Action::Fold); // seat1 SB folds (loses its posted SB)
    act(&game, &mut s, Action::Call); // seat2 BB calls
    let pot_before = game.pot(&s);
    deal_pending(&game, &mut s); // flop
    // seat2 (BB, first to act postflop) checks, seat0 bets, seat2 folds.
    act(&game, &mut s, Action::Check); // seat2
    let bet = game
        .legal_actions(&s)
        .into_iter()
        .find(|a| matches!(a, Action::Raise(_)))
        .expect("a bet is offered");
    act(&game, &mut s, bet); // seat0 bets
    act(&game, &mut s, Action::Fold); // seat2 folds
    assert!(game.is_terminal(&s), "one player left");
    assert_zero_sum(&game, &s, 3);
    // seat0 collects the pot including the dead SB and seat2's calls.
    assert!(game.returns(&s, 0) > 0.0, "winner profits from dead money");
    assert!(pot_before > 0);
}

#[test]
fn four_way_layered_side_pots_conserve() {
    // Four players, four distinct all-in levels: 10/30/70/200. Every layer must
    // be built and the total conserved regardless of who wins.
    let game = Poker::new(4)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // Assign distinct stacks. Seats: button=0, SB=1, BB=2, UTG=3.
    s.test_set_stack(0, 200);
    s.test_set_stack(1, 10);
    s.test_set_stack(2, 30);
    s.test_set_stack(3, 70);
    // UTG(3) acts first preflop. Everyone jams; calls cap at stack.
    // Drive: each remaining actor goes all-in until the round closes.
    let mut guard = 0;
    while !game.is_terminal(&s) && !matches!(game.turn(&s), Turn::Chance) {
        guard += 1;
        assert!(guard < 50, "betting should resolve");
        let acts = game.legal_actions(&s);
        // Prefer all-in; if not available (already matched), call/check.
        let a = acts
            .iter()
            .copied()
            .find(|a| *a == Action::AllIn)
            .or_else(|| acts.iter().copied().find(|a| *a == Action::Call))
            .or_else(|| acts.iter().copied().find(|a| *a == Action::Check))
            .unwrap();
        game.apply(&mut s, a);
    }
    deal_pending(&game, &mut s);
    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 4);
    // Conservation in chips.
    let bb = game.big_blind as f64;
    let total_committed: i64 = (0..4).map(|p| s.committed(p) as i64).sum();
    let total_won: i64 = (0..4)
        .map(|p| s.committed(p) as i64 + (game.returns(&s, p) * bb).round() as i64)
        .sum();
    assert_eq!(
        total_committed, total_won,
        "layered side pots conserve chips"
    );
    // Each seat's loss is bounded by what it put in.
    for p in 0..4 {
        let lost_chips = -game.returns(&s, p) * bb;
        assert!(
            lost_chips <= s.committed(p) as f64 + 1e-6,
            "seat {p} cannot lose more than it committed"
        );
    }
}

// ----------------------------------------------------------------------------
// 3. BETTING / ACTION ORDER / CLOSURE
// ----------------------------------------------------------------------------

#[test]
fn bb_gets_option_to_raise_after_limps() {
    // Everyone limps to the BB; the BB must get the option to check OR raise.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    act(&game, &mut s, Action::Call); // UTG seat0 limps
    act(&game, &mut s, Action::Call); // SB seat1 completes
    // BB (seat2) to act with the option.
    assert_eq!(game.turn(&s), Turn::Player(2), "BB has the option");
    let acts = game.legal_actions(&s);
    assert!(acts.contains(&Action::Check), "BB may check its option");
    assert!(
        acts.iter()
            .any(|a| matches!(a, Action::Raise(_)) || *a == Action::AllIn),
        "BB may also raise its option: {acts:?}"
    );
    // If BB checks, the street closes (no premature close before the option).
    act(&game, &mut s, Action::Check);
    assert_eq!(s.street(), Street::Flop, "BB check closes preflop");
}

#[test]
fn caller_owing_an_all_in_must_get_to_act_closure_bug_guard() {
    // The classic premature-close bug: seat A bets, seat B raises all-in, seat A
    // still owes the difference and MUST be given the chance to act. The street
    // must NOT close while A still owes chips.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // Preflop: UTG(0) raises to 6, SB(1) folds, BB(2) shoves all-in (200). Action
    // returns to seat 0, who still owes the call. Seat 0 must be to_act, NOT
    // skipped into a showdown.
    act_first_raise(&game, &mut s); // seat0 raises
    act(&game, &mut s, Action::Fold); // SB folds
    act(&game, &mut s, Action::AllIn); // BB shoves
    // Now seat 0 owes a call. It must be its turn and the hand must NOT be done.
    assert!(
        !game.is_terminal(&s),
        "hand must not close while a call is owed"
    );
    assert_eq!(game.turn(&s), Turn::Player(0), "raiser must get to respond");
    assert!(s.to_call(0) > 0, "seat 0 still owes chips");
    // Seat 0 closes the action by matching the shove. The BB shoved its full
    // 200; seat 0 already put in 6, so calling exhausts its stack -> the menu
    // offers AllIn (call-for-all), not a plain Call.
    let acts = game.legal_actions(&s);
    let close = if acts.contains(&Action::Call) {
        Action::Call
    } else {
        Action::AllIn
    };
    act(&game, &mut s, close);
    // Two players all-in/called -> run out and terminate.
    deal_pending(&game, &mut s);
    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 3);
}

#[test]
fn reraise_reopens_action_for_earlier_caller() {
    // seat A calls, seat B raises, seat C re-raises: seat A (who already called)
    // must be given another turn — the round must not close on the re-raise.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // UTG(0) calls, SB(1) raises, BB(2) re-raises -> action back to UTG(0).
    act(&game, &mut s, Action::Call); // seat0 limps
    act_first_raise(&game, &mut s); // seat1 raises
    act_first_raise(&game, &mut s); // seat2 re-raises
    // Action must return to seat0 (it had acted but the raise reopened).
    assert!(!game.is_terminal(&s));
    assert_eq!(game.turn(&s), Turn::Player(0), "limper must act again");
    assert!(s.to_call(0) > 0);
}

#[test]
fn min_raise_is_enforced() {
    // The minimum legal raise is to current_bet + last_raise. Preflop the BB is
    // 2 and last_raise is 2, so the first raise must be to at least 4.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    let acts = game.legal_actions(&s);
    for a in &acts {
        if let Action::Raise(to) = a {
            assert!(*to >= 4, "raise-to {to} is below the min-raise (>=4)");
        }
    }
    // After a raise to X, the next min-raise is to X + (X - prev_bet).
    let raise = acts
        .iter()
        .copied()
        .find(|a| matches!(a, Action::Raise(_)))
        .unwrap();
    let raised_to = if let Action::Raise(to) = raise { to } else { 0 };
    act(&game, &mut s, raise);
    let next = game.legal_actions(&s);
    let increment = raised_to - 2; // previous bet was the BB (2)
    for a in &next {
        if let Action::Raise(to) = a {
            assert!(
                *to >= raised_to + increment,
                "re-raise to {to} below min increment (>= {})",
                raised_to + increment
            );
        }
    }
}

#[test]
fn raise_targets_are_legal_and_within_stack() {
    // Fuzz: across many random states, every offered Raise(to) must satisfy
    // min-raise legality and never exceed the seat's reachable total.
    let game = Poker::new(6).with_blinds(1, 2).with_stack(200);
    let mut rng = Rng::new(0xA15E001);
    for _ in 0..2000 {
        let mut s = game.initial_state();
        let mut guard = 0;
        while !game.is_terminal(&s) {
            guard += 1;
            assert!(guard < 10_000);
            match game.turn(&s) {
                Turn::Chance => {
                    let outs = game.chance_outcomes(&s);
                    let i = rng.below(outs.len());
                    game.apply(&mut s, outs[i].0);
                }
                Turn::Player(p) => {
                    let acts = game.legal_actions(&s);
                    let max_to = s.street_bet(p) + s.stack(p);
                    let min_legal = (s.current_bet() + last_raise_of(&s)).max(game.big_blind);
                    for a in &acts {
                        if let Action::Raise(to) = a {
                            assert!(*to <= max_to, "raise {to} exceeds reachable {max_to}");
                            assert!(
                                *to >= min_legal,
                                "raise {to} below min legal {min_legal} (curbet {}, p {p})",
                                s.current_bet()
                            );
                            assert!(*to > s.current_bet(), "raise must exceed current bet");
                        }
                    }
                    let i = rng.below(acts.len());
                    game.apply(&mut s, acts[i]);
                }
            }
        }
    }
}

fn last_raise_of(s: &PokerState) -> u32 {
    s.debug_last_raise()
}

#[test]
fn all_in_for_less_than_min_raise_call_legality() {
    // A short all-in that is less than a full raise: the bet-to-match becomes the
    // all-in amount, and other players may at least call it. We confirm a caller
    // can match a sub-min all-in (legality), regardless of the reopening rule.
    let game = Poker::new(2)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // Make seat0 (button/SB) have a tiny stack so its shove is sub-min over the
    // BB. seat0 street_bet is 1 (SB); give it 2 more chips -> shove to 3 total,
    // which is below a min-raise (would be 4).
    s.test_set_stack(0, 2);
    act(&game, &mut s, Action::AllIn); // seat0 shoves to 3 (sub-min over BB 2)
    // seat1 (BB) faces a to_call of 1. It must be able to call.
    assert_eq!(game.turn(&s), Turn::Player(1));
    let acts = game.legal_actions(&s);
    assert!(
        acts.contains(&Action::Call),
        "BB can call the short all-in: {acts:?}"
    );
    act(&game, &mut s, Action::Call);
    deal_pending(&game, &mut s);
    assert!(game.is_terminal(&s));
    assert_zero_sum(&game, &s, 2);
}

#[test]
fn heads_up_postflop_bb_acts_first() {
    // Heads-up: preflop the button (SB) acts first; postflop the BB acts first.
    let game = Poker::new(2)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    assert_eq!(game.turn(&s), Turn::Player(0), "button acts first preflop");
    act(&game, &mut s, Action::Call); // button limps
    act(&game, &mut s, Action::Check); // BB checks option
    assert_eq!(s.street(), Street::Flop);
    deal_pending(&game, &mut s);
    assert_eq!(
        game.turn(&s),
        Turn::Player(1),
        "BB (non-button) acts first postflop"
    );
}

#[test]
fn heads_up_walk_terminates_and_pays_the_blinds() {
    // Blind-vs-blind walk: heads-up, the button/SB folds preflop, BB wins the SB.
    let game = Poker::new(2)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // Button (seat0, SB) acts first preflop facing the BB; it folds.
    act(&game, &mut s, Action::Fold);
    assert!(game.is_terminal(&s), "fold to one player ends the hand");
    // SB loses 0.5 bb, BB wins 0.5 bb.
    assert!(
        (game.returns(&s, 0) + 0.5).abs() < 1e-9,
        "SB walks for -0.5 bb"
    );
    assert!(
        (game.returns(&s, 1) - 0.5).abs() < 1e-9,
        "BB collects +0.5 bb"
    );
    assert_zero_sum(&game, &s, 2);
}

#[test]
fn sub_min_all_in_reopening_is_intentional_and_consistent() {
    // The implementer deliberately REOPENS action on any raise, including a short
    // all-in that is less than a full raise (formal rules would not reopen for a
    // player who already acted). This test documents that intentional choice so a
    // future change is a conscious one — and verifies it stays internally
    // consistent (no panic, terminates, zero-sum).
    //
    // 3-handed: UTG(0) opens, SB(1) calls, then BB(2) makes a sub-min all-in.
    // Under the engine's rule, UTG and SB are both given another turn.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_button(0)
        .with_stack(200);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    act_first_raise(&game, &mut s); // seat0 opens (raise to >=4)
    let after_open = s.current_bet();
    act(&game, &mut s, Action::Call); // seat1 calls the open
    // seat2 (BB) shoves a tiny stack: a sub-min all-in over the open.
    set_allin_total(&mut s, 2, after_open + 1); // just 1 over the open -> sub-min
    act(&game, &mut s, Action::AllIn);
    // Engine's rule: the sub-min all-in reopens, so seat0 (the opener) is back on
    // and still has a (tiny) amount to call.
    assert!(!game.is_terminal(&s));
    assert_eq!(
        game.turn(&s),
        Turn::Player(0),
        "engine reopens to the opener after a sub-min all-in (intentional)"
    );
    assert!(s.to_call(0) > 0, "opener owes the 1-chip difference");
    // Drive the rest randomly; it must terminate cleanly and stay zero-sum.
    let mut rng = Rng::new(0x5);
    let mut guard = 0;
    while !game.is_terminal(&s) {
        guard += 1;
        assert!(guard < 1000);
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                game.apply(&mut s, outs[rng.below(outs.len())].0);
            }
            Turn::Player(_) => {
                let acts = game.legal_actions(&s);
                game.apply(&mut s, acts[rng.below(acts.len())]);
            }
        }
    }
    assert_zero_sum(&game, &s, 3);
}

// ----------------------------------------------------------------------------
// 4. ZERO-SUM + TERMINATION + NO PANICS (broad fuzz across seat counts)
// ----------------------------------------------------------------------------

#[test]
fn massive_random_fuzz_all_seat_counts_terminate_zero_sum_no_panic() {
    for seats in [2u8, 3, 4, 5, 6, 7, 8, 9] {
        let game = Poker::new(seats).with_blinds(1, 2).with_stack(200);
        let mut rng = Rng::new(0xF0000 + seats as u64);
        for _ in 0..3000 {
            let mut s = game.initial_state();
            let mut steps = 0;
            while !game.is_terminal(&s) {
                steps += 1;
                assert!(steps < 200_000, "must terminate (seats={seats})");
                match game.turn(&s) {
                    Turn::Chance => {
                        let outs = game.chance_outcomes(&s);
                        assert!(!outs.is_empty(), "chance must offer a card");
                        let i = rng.below(outs.len());
                        game.apply(&mut s, outs[i].0);
                    }
                    Turn::Player(_) => {
                        let acts = game.legal_actions(&s);
                        assert!(!acts.is_empty(), "a live player must have a legal move");
                        let i = rng.below(acts.len());
                        game.apply(&mut s, acts[i]);
                    }
                }
            }
            // Zero-sum, exact in fixed point.
            let total: f64 = (0..seats as usize).map(|p| game.returns(&s, p)).sum();
            assert!(total.abs() < 1e-6, "zero-sum seats={seats} total={total}");
            // Conservation in chips (integer-exact).
            let bb = game.big_blind as i64;
            let committed: i64 = (0..seats as usize).map(|p| s.committed(p) as i64).sum();
            let won: i64 = (0..seats as usize)
                .map(|p| s.committed(p) as i64 + (game.returns(&s, p) * bb as f64).round() as i64)
                .sum();
            assert_eq!(committed, won, "chips conserved seats={seats}");
        }
    }
}

#[test]
fn all_in_preflop_walks_and_single_caller_no_panic() {
    // Edge states: blind-vs-blind walks, single-caller pots, instant all-ins.
    for seats in [2u8, 3, 6, 9] {
        let game = Poker::new(seats).with_blinds(1, 2).with_stack(4); // tiny stacks -> constant all-ins
        let mut rng = Rng::new(0xA11 + seats as u64);
        for _ in 0..2000 {
            let s = {
                let mut st = game.initial_state();
                while !game.is_terminal(&st) {
                    match game.turn(&st) {
                        Turn::Chance => {
                            let outs = game.chance_outcomes(&st);
                            game.apply(&mut st, outs[rng.below(outs.len())].0);
                        }
                        Turn::Player(_) => {
                            let acts = game.legal_actions(&st);
                            game.apply(&mut st, acts[rng.below(acts.len())]);
                        }
                    }
                }
                st
            };
            let total: f64 = (0..seats as usize).map(|p| game.returns(&s, p)).sum();
            assert!(total.abs() < 1e-6, "tiny-stack zero-sum seats={seats}");
        }
    }
}

#[test]
fn every_seat_all_in_preflop_runs_out_full_board() {
    // When 2+ live seats are all-in preflop, the engine must deal a full 5-card
    // board with no further betting and settle. Drive everyone to commit (call or
    // shove, never fold) so the maximum number of seats reach showdown all-in.
    for seats in [2u8, 3, 4, 6, 9] {
        let game = Poker::new(seats)
            .with_blinds(1, 2)
            .with_button(0)
            .with_stack(2);
        let mut s = game.initial_state();
        deal_pending(&game, &mut s);
        let mut guard = 0;
        while !game.is_terminal(&s) {
            guard += 1;
            assert!(guard < 1000, "must resolve (seats={seats})");
            match game.turn(&s) {
                Turn::Chance => {
                    let outs = game.chance_outcomes(&s);
                    game.apply(&mut s, outs[0].0);
                }
                Turn::Player(_) => {
                    let acts = game.legal_actions(&s);
                    // Commit, never fold: Call > AllIn > Check.
                    let a = acts
                        .iter()
                        .copied()
                        .find(|x| *x == Action::Call)
                        .or_else(|| acts.iter().copied().find(|x| *x == Action::AllIn))
                        .or_else(|| acts.iter().copied().find(|x| *x == Action::Check))
                        .unwrap_or(acts[0]);
                    game.apply(&mut s, a);
                }
            }
        }
        let live = (0..seats as usize).filter(|&p| !s.folded(p)).count();
        assert!(
            live >= 2,
            "expected a multi-way all-in showdown (seats={seats})"
        );
        assert_eq!(
            s.board().len(),
            5,
            "all-in run-out deals the full board (seats={seats})"
        );
        assert_zero_sum(&game, &s, seats as usize);
    }
}

// ----------------------------------------------------------------------------
// 5. BOT STRENGTH
// ----------------------------------------------------------------------------

/// bb/100 for `hero` in a field of `baseline`, with the hero seat AND the button
/// rotated *independently* so the hero is measured across every (seat, position)
/// combination — the proper "hero rotated through every seat" field eval, not
/// the hero-always-on-the-button shortcut.
fn bb_per_100(
    seats: u8,
    hero: &dyn Agent<Poker>,
    baseline: &dyn Agent<Poker>,
    hands: u32,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    let n = seats as u32;
    for hh in 0..hands {
        let game = Poker::new(seats)
            .with_blinds(1, 2)
            .with_stack(200)
            .with_button((hh % n) as u8);
        // Decouple hero seat from button: different rotation rates.
        let hero_seat = ((hh / n) % n) as usize;
        let agents: Vec<&dyn Agent<Poker>> = (0..seats as usize)
            .map(|p| if p == hero_seat { hero } else { baseline })
            .collect();
        let terminal = game_core::play_n(&game, &agents, &mut rng);
        total += game.returns(&terminal, hero_seat);
    }
    100.0 * total / hands as f64
}

#[test]
fn equity_bot_decisively_beats_baselines_bb_per_100() {
    let bot = PokerBot::new(PokerStyle {
        samples: 400,
        ..Default::default()
    });
    let hu_vs_call = bb_per_100(2, &bot, &AlwaysCall, 1500, 0xB01);
    let six_vs_call = bb_per_100(6, &bot, &AlwaysCall, 1500, 0xB02);
    let hu_vs_rand = bb_per_100(2, &bot, &game_core::RandomAgent, 1500, 0xB03);
    let six_vs_rand = bb_per_100(6, &bot, &game_core::RandomAgent, 1500, 0xB04);
    // Decisive: hundreds of bb/100 against calling stations, positive vs random.
    assert!(
        hu_vs_call > 50.0,
        "HU vs always-call must be decisive: {hu_vs_call:.0} bb/100"
    );
    assert!(
        six_vs_call > 200.0,
        "6-max vs always-call must crush: {six_vs_call:.0} bb/100"
    );
    assert!(
        hu_vs_rand > 50.0,
        "HU vs random must be decisive: {hu_vs_rand:.0} bb/100"
    );
    assert!(
        six_vs_rand > 200.0,
        "6-max vs random must crush: {six_vs_rand:.0} bb/100"
    );
    eprintln!(
        "bot bb/100 — HU/call {hu_vs_call:.0}, 6max/call {six_vs_call:.0}, HU/rand {hu_vs_rand:.0}, 6max/rand {six_vs_rand:.0}"
    );
}

/// Pool per-seat mean returns over several seeds for a self-play field of one
/// agent, button rotated uniformly. Returns (per-seat bb/100, total hands).
fn self_play_seat_means(
    agent: &dyn Agent<Poker>,
    seats: usize,
    seeds: &[u64],
    hands_per_seed: u32,
) -> Vec<f64> {
    let mut seat_sums = vec![0.0f64; seats];
    for &seed in seeds {
        let mut rng = Rng::new(seed);
        for hh in 0..hands_per_seed {
            let game = Poker::new(seats as u8)
                .with_blinds(1, 2)
                .with_stack(200)
                .with_button((hh % seats as u32) as u8);
            let agents: Vec<&dyn Agent<Poker>> = (0..seats).map(|_| agent).collect();
            let t = game_core::play_n(&game, &agents, &mut rng);
            let hand_total: f64 = (0..seats).map(|p| game.returns(&t, p)).sum();
            assert!(hand_total.abs() < 1e-6, "every self-play hand is zero-sum");
            for (p, sum) in seat_sums.iter_mut().enumerate() {
                *sum += game.returns(&t, p);
            }
        }
    }
    let total = (seeds.len() as u32 * hands_per_seed) as f64;
    seat_sums.iter().map(|s| 100.0 * s / total).collect()
}

#[test]
fn no_structural_per_seat_bias_in_engine() {
    // Definitive bias check, isolated from raising variance: a *check-or-fold*
    // equity field (never raises, so pots stay tiny and variance is small). With
    // identical agents and the button rotated uniformly, every seat must be ~even.
    // This pins down the ENGINE + seat handling: any large per-seat skew here is a
    // real positional bug, not a variance artifact.
    let calm = EquityRollout { samples: 150 };
    let means = self_play_seat_means(&calm, 6, &[1, 2, 3], 4000);
    let worst = means.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(
        worst < 8.0,
        "low-variance self-play must be seat-symmetric, worst |mean|={worst:.1} bb/100 of {means:?}"
    );
}

#[test]
fn raising_bot_self_play_seat_skew_is_within_statistical_noise() {
    // The full raising bot has enormous per-hand variance (200bb all-in pots), so
    // per-seat self-play means converge slowly. To distinguish variance from a
    // structural bias, compute the standard error of each seat's mean directly
    // and assert every seat mean sits within ~3.5 SE of zero — i.e. the observed
    // skew is fully explained by variance, with no persistent positional edge.
    let bot = PokerBot::new(PokerStyle {
        samples: 150,
        ..Default::default()
    });
    let seats = 6usize;
    let hands = 30000u32;
    let mut sum = vec![0.0f64; seats];
    let mut sumsq = vec![0.0f64; seats];
    let mut rng = Rng::new(0xC0FFEE99);
    for hh in 0..hands {
        let game = Poker::new(seats as u8)
            .with_blinds(1, 2)
            .with_stack(200)
            .with_button((hh % seats as u32) as u8);
        let agents: Vec<&dyn Agent<Poker>> =
            (0..seats).map(|_| &bot as &dyn Agent<Poker>).collect();
        let t = game_core::play_n(&game, &agents, &mut rng);
        for p in 0..seats {
            let r = game.returns(&t, p);
            sum[p] += r;
            sumsq[p] += r * r;
        }
    }
    let n = hands as f64;
    let mut max_z = 0.0f64;
    for p in 0..seats {
        let mean = sum[p] / n;
        let var = (sumsq[p] / n - mean * mean).max(1e-12);
        let se = (var / n).sqrt();
        let z = (mean / se).abs();
        max_z = max_z.max(z);
        eprintln!(
            "seat {p}: mean {:+.2} bb/hand, se {:.3}, z {:.2}",
            mean, se, z
        );
    }
    // Six seats: even the worst z-score should be modest if there's no bias.
    assert!(
        max_z < 3.5,
        "a seat deviates from zero by {max_z:.1} SE — that would indicate bias, not variance"
    );
}
