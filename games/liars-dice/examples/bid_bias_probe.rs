//! Standing diagnostic: does a Liar's Dice policy (default: the deployed
//! history-net champion) bid face 6 in a *hand-blind* way, and is that costing
//! it value?
//!
//! The raise ladder has a real asymmetry (verified against `legal_actions` /
//! `apply` below): a bid of `q x faces` (the maximum face) is the unique bid
//! whose every legal raise increases quantity — `RaiseQuantity` obviously does,
//! and `RaiseFace` wraps `faces -> 1` with `+1` quantity instead of the usual
//! same-quantity face bump. Every other face has a quantity-preserving
//! `RaiseFace` continuation. So the top face is a genuinely stronger bid to
//! park on: it costs the responder more to raise past. A policy that has
//! learned this can either lean on it *opportunistically* (bid the top face
//! more when it's plausible) or collapse to it *unconditionally* regardless of
//! its own hand — the latter is exploitable (it conveys no information) and
//! risks bids that are provably false from the bidder's own information.
//!
//! Four metrics distinguish the two:
//!   1. bid-face distribution (all bids, and opens only) and `P(bid = top
//!      face)` by total-dice-remaining phase (early/mid/endgame).
//!   2. open-on-top-face holdings correlation: `P(open on top face | own count
//!      of it)` and its Pearson r, versus the same computed pooled over every
//!      other face.
//!   3. endorse-vs-move-on by holdings: at a live bid on face `f`, `P(stick
//!      with f via RaiseQuantity, vs move on via RaiseFace)` as a function of
//!      the actor's own count of `f` — top face vs the other faces pooled.
//!   4. provably-impossible-bid rate: bids whose quantity exceeds the bidder's
//!      own count plus every unseen die (false even in the best case for the
//!      bidder), overall and in the endgame (total dice <= 6).
//!
//! Re-run this after every new league champion; v2 success criteria are the
//! impossible-bid rate trending to ~0 and a positive open-on-top-face holdings
//! correlation.
//!
//!     cargo run --release -p liars-dice --example bid_bias_probe
//!     cargo run --release -p liars-dice --example bid_bias_probe -- checkpoint=runs/ld_history/best.bin players=5 dice=5 faces=6

use game_core::{Agent, Game, Rng, Turn};
use liars_dice::{Action, HistoryNetAgent, LdState, LiarsDice, ProbConfig, ProbabilisticAgent};

const DEFAULT_CHECKPOINT: &str = "web/app/public/artifacts/ld-history-champion.bin";

struct Args {
    checkpoint: String,
    players: u8,
    dice: u8,
    faces: u8,
    selfplay_games: u32,
    mixed_games: u32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            checkpoint: DEFAULT_CHECKPOINT.to_string(),
            players: 5,
            dice: 5,
            faces: 6,
            selfplay_games: 1500,
            mixed_games: 1500,
            seed: 0x0B1A_5ED6,
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut args = Self::default();
        for raw in std::env::args().skip(1) {
            let Some((key, value)) = raw.split_once('=') else {
                panic!("expected key=value argument, got '{raw}'");
            };
            match key {
                "checkpoint" => args.checkpoint = value.to_string(),
                "players" => {
                    args.players = value
                        .parse()
                        .unwrap_or_else(|_| panic!("bad players={value}"))
                }
                "dice" => args.dice = value.parse().unwrap_or_else(|_| panic!("bad dice={value}")),
                "faces" => {
                    args.faces = value
                        .parse()
                        .unwrap_or_else(|_| panic!("bad faces={value}"))
                }
                "selfplay_games" => {
                    args.selfplay_games = value
                        .parse()
                        .unwrap_or_else(|_| panic!("bad selfplay_games={value}"));
                }
                "mixed_games" => {
                    args.mixed_games = value
                        .parse()
                        .unwrap_or_else(|_| panic!("bad mixed_games={value}"));
                }
                "seed" => args.seed = value.parse().unwrap_or_else(|_| panic!("bad seed={value}")),
                other => panic!("unknown argument key '{other}'"),
            }
        }
        args
    }
}

/// One bidding action taken by the policy under test, with enough context to
/// derive all four metrics after the fact.
#[derive(Clone, Copy)]
struct BidEvent {
    is_open: bool,
    /// True at a live raise (qty > 0) where both `RaiseQuantity` and
    /// `RaiseFace` were legal — the real endorse-vs-move-on choice point.
    endorse_choice: bool,
    /// Only meaningful when `endorse_choice`: chose `RaiseQuantity` (stick
    /// with `prev_face`) rather than `RaiseFace` (move on).
    endorsed: bool,
    hand: [u8; 6],
    my_dice: u8,
    total_dice: u32,
    prev_face: u8,
    resulting_face: u8,
    resulting_qty: u8,
}

fn total_dice(s: &LdState) -> u32 {
    s.dice_left().iter().map(|&d| u32::from(d)).sum()
}

/// Plays one game, logging every bidding action taken by a seat where
/// `is_under_test[seat]` is true.
fn play_and_log(
    game: &LiarsDice,
    agents: &[&dyn Agent<LiarsDice>],
    is_under_test: &[bool],
    rng: &mut Rng,
    log: &mut Vec<BidEvent>,
) {
    let mut s = game.initial_state();
    while !game.is_terminal(&s) {
        match game.turn(&s) {
            Turn::Chance => {
                let a = game.sample_chance_action(&s, rng);
                game.apply(&mut s, a);
            }
            Turn::Player(p) => {
                let (prev_qty, prev_face) = s.current_bid();
                let hand: [u8; 6] = std::array::from_fn(|i| s.my_count(p, i as u8 + 1));
                let my_dice = s.dice_left()[p];
                let td = total_dice(&s);
                let total: u8 = s.dice_left().iter().sum();
                let both_legal = prev_qty > 0 && prev_qty < total;
                let i = agents[p].act(game, &s, p, rng);
                let action = game.action_at(&s, i);
                game.apply(&mut s, action);
                if !is_under_test[p] {
                    continue;
                }
                let rec = match action {
                    Action::Open(q, f) => Some((true, false, false, q, f)),
                    Action::RaiseQuantity => {
                        Some((false, both_legal, true, prev_qty + 1, prev_face))
                    }
                    Action::RaiseFace => {
                        let (q, f) = if prev_face < game.faces {
                            (prev_qty, prev_face + 1)
                        } else {
                            (prev_qty + 1, 1)
                        };
                        Some((false, both_legal, false, q, f))
                    }
                    _ => None,
                };
                if let Some((is_open, endorse_choice, endorsed, rq, rf)) = rec {
                    log.push(BidEvent {
                        is_open,
                        endorse_choice,
                        endorsed,
                        hand,
                        my_dice,
                        total_dice: td,
                        prev_face,
                        resulting_face: rf,
                        resulting_qty: rq,
                    });
                }
            }
        }
    }
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// `(n_decisions, n_hit)` buckets by own count of `face` (0, 1, 2, 3+) at
/// opening decisions, where "hit" means the open landed on `face`.
fn open_modulation(events: &[BidEvent], face: u8) -> [(u32, u32); 4] {
    let mut buckets = [(0u32, 0u32); 4];
    for e in events.iter().filter(|e| e.is_open) {
        let k = e.hand[face as usize - 1].min(3) as usize;
        buckets[k].0 += 1;
        if e.resulting_face == face {
            buckets[k].1 += 1;
        }
    }
    buckets
}

/// Same shape, but "hit" means endorsing (staying on) `face` at a live raise
/// where `face` was the current bid and both raise directions were legal.
fn endorse_modulation(events: &[BidEvent], face: u8) -> [(u32, u32); 4] {
    let mut buckets = [(0u32, 0u32); 4];
    for e in events
        .iter()
        .filter(|e| e.endorse_choice && e.prev_face == face)
    {
        let k = e.hand[face as usize - 1].min(3) as usize;
        buckets[k].0 += 1;
        if e.endorsed {
            buckets[k].1 += 1;
        }
    }
    buckets
}

fn pool(faces: impl Iterator<Item = u8>, table: impl Fn(u8) -> [(u32, u32); 4]) -> [(u32, u32); 4] {
    let mut pooled = [(0u32, 0u32); 4];
    for f in faces {
        let t = table(f);
        for k in 0..4 {
            pooled[k].0 += t[k].0;
            pooled[k].1 += t[k].1;
        }
    }
    pooled
}

fn print_table(label: &str, buckets: [(u32, u32); 4]) {
    println!("  {label}:");
    for (k, (n, hit)) in buckets.iter().enumerate() {
        let bucket_label = if k == 3 {
            "3+".to_string()
        } else {
            k.to_string()
        };
        println!(
            "    own_count={bucket_label:<3} n={n:>7}  P={:.3}",
            *hit as f64 / (*n).max(1) as f64
        );
    }
}

fn main() {
    let args = Args::parse();
    let game = LiarsDice::new(args.players, args.dice, args.faces);

    println!("=== structural raise-graph check ===");
    // At bid q x faces, every legal raise increases quantity; at q x f with
    // f < faces, RaiseFace never does. Drive real states through `apply`
    // (LdState's fields are private) rather than constructing them by hand.
    {
        let mut probe_rng = Rng::new(1);
        let mut s = game.initial_state();
        while matches!(game.turn(&s), Turn::Chance) {
            let a = game.sample_chance_action(&s, &mut probe_rng);
            game.apply(&mut s, a);
        }
        game.apply(&mut s, Action::RaiseQuantity);
        game.apply(&mut s, Action::RaiseQuantity);
        for _ in 0..(args.faces - 1) {
            game.apply(&mut s, Action::RaiseFace);
        }
        assert_eq!(
            s.current_bid(),
            (3, args.faces),
            "expected to land on 3 x top face"
        );
        let acts = game.legal_actions(&s);
        println!("  bid 3x{}: {acts:?}", args.faces);
        for a in &acts {
            match a {
                Action::RaiseQuantity => println!("    RaiseQuantity -> 4x{} (qty+1)", args.faces),
                Action::RaiseFace => println!("    RaiseFace -> 4x1 (wraps, qty+1)"),
                _ => {}
            }
        }

        let mut s2 = game.initial_state();
        while matches!(game.turn(&s2), Turn::Chance) {
            let a = game.sample_chance_action(&s2, &mut probe_rng);
            game.apply(&mut s2, a);
        }
        game.apply(&mut s2, Action::RaiseQuantity);
        game.apply(&mut s2, Action::RaiseQuantity);
        game.apply(&mut s2, Action::RaiseFace);
        game.apply(&mut s2, Action::RaiseFace);
        let mid_face = s2.current_bid().1;
        assert!(mid_face < args.faces, "expected a mid-ladder, non-top face");
        let acts2 = game.legal_actions(&s2);
        println!("  bid 3x{mid_face}: {acts2:?}");
        for a in &acts2 {
            match a {
                Action::RaiseQuantity => println!("    RaiseQuantity -> 4x{mid_face} (qty+1)"),
                Action::RaiseFace => {
                    println!("    RaiseFace -> 3x{} (qty unchanged)", mid_face + 1)
                }
                _ => {}
            }
        }
    }
    println!();

    println!(
        "=== bid-distribution instrumentation: {} ===",
        args.checkpoint
    );
    let champion = HistoryNetAgent::load(std::path::Path::new(&args.checkpoint))
        .unwrap_or_else(|e| panic!("failed to load checkpoint at {}: {e}", args.checkpoint));
    let mut log: Vec<BidEvent> = Vec::new();
    let mut rng = Rng::new(args.seed);

    {
        let agents: Vec<&dyn Agent<LiarsDice>> = (0..args.players as usize)
            .map(|_| &champion as &dyn Agent<LiarsDice>)
            .collect();
        let mask = vec![true; args.players as usize];
        for _ in 0..args.selfplay_games {
            play_and_log(&game, &agents, &mask, &mut rng, &mut log);
        }
    }
    println!("  self-play games logged: {}", args.selfplay_games);

    {
        let scripted = [
            ProbabilisticAgent::new(ProbConfig::default()),
            ProbabilisticAgent::new(ProbConfig::aggressive_bluffer()),
            ProbabilisticAgent::new(ProbConfig::conservative_caller()),
            ProbabilisticAgent::new(ProbConfig::honest_bayes()),
        ];
        for g in 0..args.mixed_games {
            let champ_seat = (g as usize) % args.players as usize;
            let mut agents: Vec<&dyn Agent<LiarsDice>> = Vec::with_capacity(args.players as usize);
            let mut mask = vec![false; args.players as usize];
            let mut scripted_idx = 0;
            for (p, slot) in mask.iter_mut().enumerate() {
                if p == champ_seat {
                    agents.push(&champion as &dyn Agent<LiarsDice>);
                    *slot = true;
                } else {
                    agents.push(&scripted[scripted_idx] as &dyn Agent<LiarsDice>);
                    scripted_idx += 1;
                }
            }
            play_and_log(&game, &agents, &mask, &mut rng, &mut log);
        }
    }
    println!("  mixed-field games logged: {}", args.mixed_games);
    println!("  total logged bid events: {}", log.len());

    let top = args.faces;

    // 1. face distributions, and P(top face) by phase.
    println!("\n-- 1. face distribution: all bids --");
    let mut hist = vec![0u32; args.faces as usize];
    for e in &log {
        hist[e.resulting_face as usize - 1] += 1;
    }
    let total_bids: u32 = hist.iter().sum();
    for f in 1..=args.faces {
        let c = hist[f as usize - 1];
        println!(
            "  face {f}: {c:>7}  ({:.1}%)",
            100.0 * c as f64 / total_bids.max(1) as f64
        );
    }

    println!("\n-- 1. face distribution: opening bids only --");
    let opens: Vec<BidEvent> = log.iter().filter(|e| e.is_open).copied().collect();
    let mut ohist = vec![0u32; args.faces as usize];
    for e in &opens {
        ohist[e.resulting_face as usize - 1] += 1;
    }
    let total_opens: u32 = ohist.iter().sum();
    for f in 1..=args.faces {
        let c = ohist[f as usize - 1];
        println!(
            "  face {f}: {c:>7}  ({:.1}%)",
            100.0 * c as f64 / total_opens.max(1) as f64
        );
    }

    println!("\n-- 1. P(bid = top face) by total-dice-remaining phase --");
    // Non-overlapping quintile-ish cuts of [7, total_start], with the
    // endgame bucket fixed at the analytically meaningful `<= 6` threshold.
    let total_start = u32::from(args.players) * u32::from(args.dice);
    let e1 = total_start * 4 / 5;
    let e2 = total_start * 3 / 5;
    let e3 = total_start * 2 / 5;
    let phase_bounds = [
        (e1 + 1, total_start, "early"),
        (e2 + 1, e1, "mid"),
        (e3 + 1, e2, "mid-late"),
        (7, e3, "late"),
        (2, 6, "endgame"),
    ];
    for (lo, hi, label) in phase_bounds {
        let in_phase: Vec<&BidEvent> = log
            .iter()
            .filter(|e| e.total_dice >= lo && e.total_dice <= hi)
            .collect();
        let n = in_phase.len();
        let n_top = in_phase.iter().filter(|e| e.resulting_face == top).count();
        println!(
            "  {label:<10} ({lo}-{hi} dice)  n={n:>7}  P(top face)={:.3}",
            n_top as f64 / n.max(1) as f64
        );
    }

    // 2. open-on-top-face holdings correlation.
    println!("\n-- 2. P(open on face f | own_count(f)=k) --");
    print_table("top face", open_modulation(&log, top));
    print_table(
        "other faces pooled",
        pool(1..top, |f| open_modulation(&log, f)),
    );

    {
        let xs_top: Vec<f64> = opens
            .iter()
            .map(|e| f64::from(e.hand[top as usize - 1]))
            .collect();
        let ys_top: Vec<f64> = opens
            .iter()
            .map(|e| if e.resulting_face == top { 1.0 } else { 0.0 })
            .collect();
        let r_top = pearson(&xs_top, &ys_top);
        let rs_other: Vec<f64> = (1..top)
            .map(|f| {
                let xs: Vec<f64> = opens
                    .iter()
                    .map(|e| f64::from(e.hand[f as usize - 1]))
                    .collect();
                let ys: Vec<f64> = opens
                    .iter()
                    .map(|e| if e.resulting_face == f { 1.0 } else { 0.0 })
                    .collect();
                pearson(&xs, &ys)
            })
            .collect();
        let avg_other = rs_other.iter().sum::<f64>() / rs_other.len().max(1) as f64;
        println!(
            "\n  Pearson r(own_count(f), opened_on_f): top={r_top:.3}  avg(other faces)={avg_other:.3}"
        );
    }

    // 3. endorse-vs-move-on by holdings.
    println!("\n-- 3. P(endorse current face f | own_count(f)=k) --");
    print_table("top face", endorse_modulation(&log, top));
    print_table(
        "other faces pooled",
        pool(1..top, |f| endorse_modulation(&log, f)),
    );

    // 4. provably-impossible-bid rate.
    println!("\n-- 4. provably-impossible bid rate --");
    let is_impossible = |e: &BidEvent| -> bool {
        let unseen = e.total_dice - u32::from(e.my_dice);
        u32::from(e.resulting_qty) > u32::from(e.hand[e.resulting_face as usize - 1]) + unseen
    };
    let n_all = log.len();
    let n_imp = log.iter().filter(|e| is_impossible(e)).count();
    println!(
        "  overall: {n_imp}/{n_all} = {:.4}",
        n_imp as f64 / n_all.max(1) as f64
    );

    let endgame: Vec<&BidEvent> = log.iter().filter(|e| e.total_dice <= 6).collect();
    let n_end = endgame.len();
    let n_end_imp = endgame.iter().filter(|e| is_impossible(e)).count();
    println!(
        "  endgame (total<=6): {n_end_imp}/{n_end} = {:.4}",
        n_end_imp as f64 / n_end.max(1) as f64
    );

    let endgame_top_zero: Vec<&&BidEvent> = endgame
        .iter()
        .filter(|e| e.resulting_face == top && e.hand[top as usize - 1] == 0)
        .collect();
    let n_etz = endgame_top_zero.len();
    let n_etz_imp = endgame_top_zero.iter().filter(|e| is_impossible(e)).count();
    println!(
        "  endgame, top face, own_count(top)=0: {n_etz_imp}/{n_etz} = {:.4}",
        n_etz_imp as f64 / n_etz.max(1) as f64
    );
}
