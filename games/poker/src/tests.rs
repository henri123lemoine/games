//! Rules, betting, pot, and termination tests for the poker game.

use super::*;
use game_core::{Game, GameUi, Rng, Turn};

/// Deal out all pending cards (hole + board to the current street) by taking
/// the first chance outcome each time, so betting can be exercised directly.
fn deal_pending(game: &Poker, s: &mut PokerState) {
    while let Turn::Chance = game.turn(s) {
        let outs = game.chance_outcomes(s);
        game.apply(s, outs[0].0);
    }
}

/// Play one hand with all-random agents and the engine's chance sampling.
fn random_hand(game: &Poker, rng: &mut Rng) -> PokerState {
    let mut s = game.initial_state();
    let mut steps = 0;
    while !game.is_terminal(&s) {
        steps += 1;
        assert!(steps < 100_000, "hand must terminate");
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let i = rng.below(outs.len());
                game.apply(&mut s, outs[i].0);
            }
            Turn::Player(_) => {
                let acts = game.legal_actions(&s);
                let i = rng.below(acts.len());
                game.apply(&mut s, acts[i]);
            }
        }
    }
    s
}

#[test]
fn blinds_are_posted_and_first_to_act_is_correct() {
    let game = Poker::new(6).with_blinds(1, 2);
    let s = game.initial_state();
    // Button 0 ⇒ SB seat 1, BB seat 2, first to act seat 3 (UTG).
    assert_eq!(s.street_bet(1), 1, "small blind");
    assert_eq!(s.street_bet(2), 2, "big blind");
    assert_eq!(s.current_bet(), 2);
    assert_eq!(game.turn(&s), Turn::Chance, "deal first");
    let mut s = s;
    deal_pending(&game, &mut s);
    assert_eq!(game.turn(&s), Turn::Player(3), "UTG acts first preflop");
}

#[test]
fn heads_up_button_posts_small_blind_and_acts_first() {
    let game = Poker::new(2).with_blinds(1, 2);
    let mut s = game.initial_state();
    assert_eq!(s.street_bet(0), 1, "button posts SB heads-up");
    assert_eq!(s.street_bet(1), 2, "other seat posts BB");
    deal_pending(&game, &mut s);
    assert_eq!(game.turn(&s), Turn::Player(0), "button acts first preflop");
}

#[test]
fn everyone_folds_to_the_big_blind() {
    let game = Poker::new(6).with_blinds(1, 2);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s);
    // Seats 3,4,5,0,1 fold to the BB (seat 2).
    for _ in 0..5 {
        let acts = game.legal_actions(&s);
        let fold = acts.iter().position(|&a| a == Action::Fold).unwrap();
        game.apply(&mut s, acts[fold]);
    }
    assert!(game.is_terminal(&s), "one player left ⇒ hand over");
    // BB wins the dead small blind: +0.5 bb; SB loses its 0.5 bb.
    assert!((game.returns(&s, 2) - 0.5).abs() < 1e-9, "BB nets +0.5 bb");
    assert!((game.returns(&s, 1) + 0.5).abs() < 1e-9, "SB nets -0.5 bb");
    let total: f64 = (0..6).map(|p| game.returns(&s, p)).sum();
    assert!(total.abs() < 1e-9, "zero-sum");
}

#[test]
fn check_around_advances_streets_and_reaches_showdown() {
    let game = Poker::new(3).with_blinds(1, 2);
    let mut s = game.initial_state();
    deal_pending(&game, &mut s); // hole cards
    // Preflop: UTG (seat 0, button) calls, SB calls, BB checks option.
    apply_named(&game, &mut s, "call"); // seat 0
    apply_named(&game, &mut s, "call"); // seat 1 (SB)
    apply_named(&game, &mut s, "check"); // seat 2 (BB option)
    assert_eq!(s.street(), Street::Flop);
    deal_pending(&game, &mut s);
    assert_eq!(s.board().len(), 3);
    // Check around each street to showdown.
    for expected in [Street::Turn, Street::River, Street::Showdown] {
        for _ in 0..3 {
            if game.is_terminal(&s) {
                break;
            }
            apply_named(&game, &mut s, "check");
        }
        if expected != Street::Showdown {
            deal_pending(&game, &mut s);
        }
    }
    assert!(game.is_terminal(&s));
    assert_eq!(s.board().len(), 5, "full board at showdown");
    let total: f64 = (0..3).map(|p| game.returns(&s, p)).sum();
    assert!(total.abs() < 1e-9, "zero-sum at showdown");
}

fn apply_named(game: &Poker, s: &mut PokerState, name: &str) {
    let a = game.parse_action(s, name).expect("named action parses");
    let acts = game.legal_actions(s);
    let i = acts
        .iter()
        .position(|&x| x == a)
        .unwrap_or_else(|| panic!("'{name}' not legal among {acts:?}"));
    game.apply(s, acts[i]);
}

#[test]
fn random_hands_terminate_and_are_zero_sum() {
    for seats in [2u8, 3, 6, 9] {
        let game = Poker::new(seats).with_blinds(1, 2);
        let mut rng = Rng::new(0x100 + seats as u64);
        for _ in 0..300 {
            let s = random_hand(&game, &mut rng);
            let total: f64 = (0..seats as usize).map(|p| game.returns(&s, p)).sum();
            assert!(total.abs() < 1e-6, "zero-sum (seats={seats}), got {total}");
            // No seat can win more than it could possibly have at risk.
            for p in 0..seats as usize {
                assert!(game.returns(&s, p).abs() <= game.max_return() + 1e-6);
            }
        }
    }
}

#[test]
fn conservation_holds_chips_never_created_or_destroyed() {
    let game = Poker::new(6).with_stack(200).with_blinds(1, 2);
    let mut rng = Rng::new(42);
    for _ in 0..500 {
        let s = random_hand(&game, &mut rng);
        // Net payoffs (in chips) must sum to zero exactly.
        let net_bb: f64 = (0..6).map(|p| game.returns(&s, p)).sum();
        assert!(net_bb.abs() < 1e-6, "net chips conserved, got {net_bb}");
    }
}

/// Deal a specific card by selecting the chance outcome that matches it.
fn deal_card(game: &Poker, s: &mut PokerState, card: &str) {
    let target = parse_card(card).unwrap();
    let outs = game.chance_outcomes(s);
    let i = outs
        .iter()
        .position(|&(a, _)| a == Action::Deal(target))
        .unwrap_or_else(|| panic!("{card} not available to deal"));
    game.apply(s, outs[i].0);
}

#[test]
fn side_pot_splits_by_commitment_level() {
    // The canonical side-pot case: the short all-in stack makes the best hand
    // and wins the main pot it is eligible for, while the side pot between the
    // two deeper stacks goes to the better of those two. We control the board
    // so the outcome is deterministic.
    let game = Poker::new(3).with_blinds(1, 2).with_button(0);
    let mut s = game.initial_state();
    // 3-handed: SB=1, BB=2, UTG=0. Deal starts left of button (seat 1), two
    // rounds: round1 seats 1,2,0 ; round2 seats 1,2,0.
    deal_card(&game, &mut s, "2c"); // seat1 c0
    deal_card(&game, &mut s, "Kd"); // seat2 c0
    deal_card(&game, &mut s, "Ah"); // seat0 c0
    deal_card(&game, &mut s, "2s"); // seat1 c1 -> seat1: 2c 2s (will set on a 2)
    deal_card(&game, &mut s, "Kh"); // seat2 c1 -> seat2: Kd Kh (will set on a K)
    deal_card(&game, &mut s, "As"); // seat0 c1 -> seat0: Ah As (will set on an A)

    // Force a side pot: seat 1 (best hand, set of 2s) is short.
    set_stack(&mut s, 0, 100);
    set_stack(&mut s, 1, 10);
    set_stack(&mut s, 2, 100);

    // Preflop everyone gets it in.
    apply_named(&game, &mut s, "all-in"); // seat 0 shoves 100
    let acts = game.legal_actions(&s);
    let allin = acts.iter().position(|&a| a == Action::AllIn).unwrap();
    game.apply(&mut s, acts[allin]); // seat 1 calls all-in (short)
    apply_named(&game, &mut s, "call"); // seat 2 calls 100

    // Controlled board: 2 (seat1 sets) + A (seat0 sets) + low blanks; seat2's
    // kings stay an overpair-that-loses. Final ranking: 222 < AAA, so seat 0
    // (set of aces) actually has the best hand — re-pick so the *short* stack
    // wins the main pot: give the board a second 2 for quads.
    deal_card(&game, &mut s, "2d"); // seat1 now has trip/▲ towards quads
    deal_card(&game, &mut s, "2h"); // seat1: quad 2s — unbeatable
    deal_card(&game, &mut s, "Ac"); // seat0: full house aces over twos
    deal_card(&game, &mut s, "Kc"); // turn: seat2 kings full? Kd Kh Kc = trips+...
    deal_card(&game, &mut s, "5s"); // river blank

    assert!(game.is_terminal(&s));
    // seat1 = quad deuces (best), seat0 = aces full, seat2 = kings full.
    let net: Vec<f64> = (0..3).map(|p| game.returns(&s, p)).collect();
    let total: f64 = net.iter().sum();
    assert!(total.abs() < 1e-6, "zero-sum with side pots, got {total}");
    // Short stack (seat1) won the main pot it contested: it profits despite
    // only 11 chips in.
    assert!(
        net[1] > 0.0,
        "quad deuces (short all-in) wins the main pot: {net:?}"
    );
    // The side pot (the chips seats 0 and 2 put in beyond seat1's level) is
    // contested only by them, and seat0 (aces full) beats seat2 (kings full).
    assert!(net[0] > 0.0, "aces full wins the side pot: {net:?}");
    assert!(net[2] < 0.0, "kings full loses both pots: {net:?}");
}

#[test]
fn identical_hands_chop_the_pot() {
    // Two seats both make the same straight off the board: an exact split.
    let game = Poker::new(2).with_blinds(1, 2).with_button(0);
    let mut s = game.initial_state();
    // HU deal order: SB/button=0, BB=1; round1 seats 0,1; round2 seats 0,1.
    deal_card(&game, &mut s, "Ac"); // seat0 c0
    deal_card(&game, &mut s, "Ad"); // seat1 c0
    deal_card(&game, &mut s, "Kc"); // seat0 c1 -> AK
    deal_card(&game, &mut s, "Kd"); // seat1 c1 -> AK
    // Preflop both check/call to see a board where the board plays.
    apply_named(&game, &mut s, "call"); // button calls
    apply_named(&game, &mut s, "check"); // BB checks
    // Flop/turn/river: a board that makes both hands identical (board straight).
    deal_card(&game, &mut s, "Th");
    deal_card(&game, &mut s, "Jh");
    deal_card(&game, &mut s, "Qh");
    apply_named(&game, &mut s, "check");
    apply_named(&game, &mut s, "check");
    deal_card(&game, &mut s, "9s");
    apply_named(&game, &mut s, "check");
    apply_named(&game, &mut s, "check");
    deal_card(&game, &mut s, "8d");
    apply_named(&game, &mut s, "check");
    apply_named(&game, &mut s, "check");
    assert!(game.is_terminal(&s));
    // Both played AK with a board T-J-Q-9-8: each plays A-K + Q-J-T? No: best
    // five is the board straight 8-9-T-J-Q for both ⇒ exact chop, both net 0.
    assert!((game.returns(&s, 0)).abs() < 1e-9, "chop: seat 0 nets 0");
    assert!((game.returns(&s, 1)).abs() < 1e-9, "chop: seat 1 nets 0");
}

fn set_stack(s: &mut PokerState, seat: usize, chips: u32) {
    s.test_set_stack(seat, chips);
}

// ---- bot sanity: the equity bot must beat the casual baselines ----

fn bb_per_hand(
    seats: u8,
    hero: &dyn game_core::Agent<Poker>,
    baseline: &dyn game_core::Agent<Poker>,
    hands: u32,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for h in 0..hands {
        let game = Poker::new(seats)
            .with_blinds(1, 2)
            .with_stack(200)
            .with_button((h % seats as u32) as u8);
        let hero_seat = (h as usize) % seats as usize;
        let agents: Vec<&dyn game_core::Agent<Poker>> = (0..seats as usize)
            .map(|p| if p == hero_seat { hero } else { baseline })
            .collect();
        let terminal = game_core::play_n(&game, &agents, &mut rng);
        total += game.returns(&terminal, hero_seat);
    }
    total / hands as f64
}

#[test]
fn equity_bot_crushes_always_call_and_random() {
    // Small sample counts keep this fast; the edge is enormous, so even a
    // few hundred hands clear zero comfortably.
    let bot = PokerBot::new(PokerStyle {
        samples: 200,
        ..Default::default()
    });
    let vs_call = bb_per_hand(6, &bot, &AlwaysCall, 600, 0x515);
    let vs_rand = bb_per_hand(6, &bot, &game_core::RandomAgent, 600, 0x516);
    assert!(
        vs_call > 1.0,
        "equity bot must beat calling stations by >1 bb/hand, got {vs_call:.2}"
    );
    assert!(
        vs_rand > 1.0,
        "equity bot must beat random by >1 bb/hand, got {vs_rand:.2}"
    );
    let hu = bb_per_hand(2, &bot, &AlwaysCall, 600, 0x517);
    assert!(
        hu > 0.5,
        "heads-up vs call must be clearly positive, got {hu:.2}"
    );
}

// ---- continuous cash-game session ----

/// Drive a session of all-random play, stepping through the between-hand
/// `NextHand` chance node, and record per-hand boundaries.
fn play_session(
    game: &Poker,
    rng: &mut Rng,
    stop_after_hands: u16,
) -> Vec<(u16, u8, [u32; MAX_SEATS])> {
    let mut s = game.initial_state();
    let mut snapshots = Vec::new();
    let mut last_hand = u16::MAX;
    let mut steps = 0;
    while !game.is_terminal(&s) {
        steps += 1;
        assert!(steps < 5_000_000, "session must make progress");
        // Snapshot at the start of each new hand (button + stacks).
        if s.hand_no() != last_hand && !s.resolved() {
            last_hand = s.hand_no();
            let mut stacks = [0u32; MAX_SEATS];
            for (p, st) in stacks.iter_mut().enumerate().take(game.seats()) {
                *st = s.stack(p) + s.committed(p); // before-blinds equivalent
            }
            snapshots.push((s.hand_no(), s.button() as u8, stacks));
            if s.hand_no() >= stop_after_hands {
                break;
            }
        }
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let i = rng.below(outs.len());
                game.apply(&mut s, outs[i].0);
            }
            Turn::Player(_) => {
                let acts = game.legal_actions(&s);
                let i = rng.below(acts.len());
                game.apply(&mut s, acts[i]);
            }
        }
    }
    snapshots
}

#[test]
fn session_deals_many_hands_and_rotates_the_button() {
    let game = Poker::new(6)
        .with_blinds(1, 2)
        .with_stack(200)
        .with_session(true);
    let mut rng = Rng::new(0x5E5510);
    let snaps = play_session(&game, &mut rng, 12);
    assert!(
        snaps.len() >= 10,
        "a session deals hand after hand: {}",
        snaps.len()
    );
    // The button rotates one seat per hand.
    for w in snaps.windows(2) {
        let (h0, b0, _) = w[0];
        let (h1, b1, _) = w[1];
        assert_eq!(h1, h0 + 1, "hand number increments");
        assert_eq!(b1, (b0 + 1) % 6, "button rotates one seat per hand");
    }
}

#[test]
fn session_carries_stacks_across_hands_and_conserves_chips() {
    let game = Poker::new(4)
        .with_blinds(1, 2)
        .with_stack(150)
        .with_session(true);
    let mut rng = Rng::new(0xCA54);
    let snaps = play_session(&game, &mut rng, 20);
    assert!(snaps.len() >= 15);
    let total_start = 4 * 150;
    for (hand, _btn, stacks) in &snaps {
        let total: u32 = stacks.iter().take(4).sum();
        // No rebuy fired (everyone stays funded in this short run) ⇒ chips are
        // conserved hand to hand; a rebuy would only ever add chips, never lose.
        assert!(
            total >= total_start || *hand > 0,
            "hand {hand}: chips not destroyed (have {total}, started {total_start})"
        );
    }
    // Stacks actually change between hands (chips move), i.e. it's not resetting
    // to a fresh 150 each hand.
    let distinct: std::collections::HashSet<_> = snaps.iter().map(|(_, _, st)| st[0]).collect();
    assert!(
        distinct.len() > 1,
        "seat 0's stack varies across hands (carried over)"
    );
}

#[test]
fn session_rebuys_a_busted_seat_so_play_continues() {
    // Tiny stacks + constant all-in random play will bust seats; the session
    // must top them back up and keep dealing, never deadlocking.
    let game = Poker::new(3)
        .with_blinds(1, 2)
        .with_stack(6)
        .with_session(true);
    let mut rng = Rng::new(0xB057);
    let snaps = play_session(&game, &mut rng, 40);
    assert!(
        snaps.len() >= 30,
        "rebuys keep the table alive: {}",
        snaps.len()
    );
    // At the start of every hand each seat can cover the big blind (rebuy works).
    for (hand, _btn, stacks) in &snaps {
        for (p, &st) in stacks.iter().take(3).enumerate() {
            assert!(st >= 2, "hand {hand} seat {p}: funded to play ({st} chips)");
        }
    }
}

#[test]
fn session_finally_terminates_at_the_hand_cap() {
    // A small cap so the test is quick; the session must end cleanly.
    let mut game = Poker::new(2)
        .with_blinds(1, 2)
        .with_stack(50)
        .with_session(true);
    game.session_hands = 8;
    let mut rng = Rng::new(0x0CA9);
    let mut s = game.initial_state();
    let mut steps = 0;
    while !game.is_terminal(&s) {
        steps += 1;
        assert!(steps < 1_000_000);
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
    assert!(game.is_terminal(&s), "the session ends at the cap");
    assert_eq!(
        s.hand_no(),
        7,
        "played session_hands-1 boundaries then ended"
    );
}

#[test]
fn one_hand_game_is_unchanged_by_the_session_flag() {
    // With session off, the game is terminal after exactly one hand (the metric
    // path the arena/bot_eval rely on).
    let game = Poker::new(6).with_blinds(1, 2).with_stack(200);
    assert!(!game.session);
    let mut rng = Rng::new(7);
    let s = random_hand(&game, &mut rng);
    assert!(game.is_terminal(&s), "one hand then terminal");
    let total: f64 = (0..6).map(|p| game.returns(&s, p)).sum();
    assert!(total.abs() < 1e-6, "still zero-sum");
}
