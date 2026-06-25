//! Does the trained net over-call thin 1-quantity bids? Compares the net's
//! call frequency to the EXACT 2p1d6f equilibrium at the same information sets.
//!
//!     cargo run --release -p liars-dice --example probe_check -- runs/ld_final/best.bin

use game_core::{Game, Turn};
use liars_dice::features::net_policy;
use liars_dice::{
    Action, DiceShareValue, FitConfig, LiarsDice, MAX_FACES, NetAgent, RoundSubgame, fit_two_player,
};
use solvers::Cfr;

fn hand_with(face: u8) -> [u8; MAX_FACES] {
    let mut h = [0u8; MAX_FACES];
    h[face as usize - 1] = 1;
    h
}

/// Build the free-open 2p1d6f state where seat 0 holds a 1 and has opened
/// `1 x bid_face`, seat 1 holds `my_face`, and it is seat 1's turn.
fn probe_state(my_face: u8, bid_face: u8) -> liars_dice::LdState {
    let round = RoundSubgame::new(
        2,
        1,
        6,
        [1, 1, 0, 0, 0, 0, 0, 0],
        0,
        false,
        4,
        DiceShareValue,
    );
    let mut s = round.initial_state();
    let hands = [hand_with(1), hand_with(my_face)];
    let mut rolled = 0;
    while let Turn::Chance = round.turn(&s) {
        round.apply(&mut s, hands[rolled].into_roll());
        rolled += 1;
    }
    round.apply(&mut s, Action::Open(1, bid_face));
    s
}

trait IntoRoll {
    fn into_roll(self) -> Action;
}
impl IntoRoll for [u8; MAX_FACES] {
    fn into_roll(self) -> Action {
        Action::Roll(self)
    }
}

fn call_liar_prob(probs: &[f64], acts: &[Action]) -> f64 {
    acts.iter()
        .zip(probs)
        .find(|(a, _)| matches!(a, Action::CallLiar))
        .map(|(_, &p)| p)
        .unwrap_or(0.0)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("runs/ld_final/best.bin".into());
    let agent = NetAgent::load(std::path::Path::new(&path)).expect("load net");
    let net = agent.net();
    let cache = net.infer_cache();
    let game = LiarsDice::new(2, 1, 6);

    // Exact equilibrium: solve the 2p1d6f free-open round with the converged
    // lattice continuation.
    let fit = fit_two_player(1, 6, FitConfig::default());
    let round = RoundSubgame::new(2, 1, 6, [1, 1, 0, 0, 0, 0, 0, 0], 0, false, 4, fit.lattice);
    let mut cfr = Cfr::new(round);
    cfr.solve(200_000);

    println!("2p1d6f: net (seat 1) facing seat-0 open `1 x F`, holding `my`");
    println!(
        "  exact eq exploitability check: cfr nashconv/2 = {:.5}",
        cfr.exploitability().2 / 2.0
    );
    println!();
    println!(
        "  {:<28} {:>10} {:>10}",
        "scenario", "exact P(call)", "net P(call)"
    );
    println!("  {}", "-".repeat(52));
    for (my_face, bid_face, note) in [
        (2u8, 2u8, "believable (bid = own die)"),
        (2, 5, "thin (hold none of bid face)"),
        (2, 6, "thin (hold none of bid face)"),
        (5, 5, "I HOLD the bid face"),
    ] {
        let s = probe_state(my_face, bid_face);
        let acts = game.legal_actions(&s);
        let exact = cfr.policy(&s, 1);
        let netp = net_policy(net, &cache, &game, &s, 1);
        println!(
            "  hold {my_face}, face 1x{bid_face} {:<10} {:>10.3} {:>10.3}",
            format!("[{note}]"),
            call_liar_prob(&exact, &acts),
            call_liar_prob(&netp, &acts),
        );
    }
}
