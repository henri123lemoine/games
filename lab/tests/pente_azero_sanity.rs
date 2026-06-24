//! End-to-end smoke check for the native hybrid Pente bot: build the registered
//! `bot=azero` match against the exported net and step it, confirming the
//! VCF → Search<Pente> → nn_infer path runs without panicking and returns a
//! legal move. Skipped (not failed) when the exported net is absent, so it does
//! not break a checkout that has not run the trainer's `export`.

use std::collections::HashMap;

use lab::registry::{Opts, entries};

const NET: &str = "../data/azpente/run1/azero-pente.azweb";

#[test]
fn azero_pente_bot_plays_two_moves_end_to_end() {
    if !std::path::Path::new(NET).exists() {
        eprintln!("skipping: exported net {NET} not present (run the trainer's export)");
        return;
    }

    let entry = entries()
        .into_iter()
        .find(|e| e.id == "pente")
        .expect("pente is registered");

    let mut map = HashMap::new();
    map.insert("bot".into(), "azero".into());
    map.insert("seat".into(), "watch".into());
    map.insert("net".into(), NET.into());
    map.insert("sims".into(), "24".into());
    map.insert("size".into(), "13".into());
    map.insert("seed".into(), "7".into());
    let opts = Opts::new(map);

    let mut m = (entry.make)(&opts).expect("build the azero pente match");

    // Move 1: Black's forced center. Move 2: White's genuine net-guided search
    // (encoder → Search<Pente> + nn_infer leaf eval → argmax, behind the VCF).
    let first = m.step().expect("black plays the forced opening");
    assert_eq!(first.seat, 0);
    let second = m.step().expect("white plays a searched reply");
    assert_eq!(second.seat, 1);
    assert!(
        !second.label.is_empty(),
        "the searched move carries a board label"
    );
    eprintln!(
        "azero pente opened: black {} / white {}",
        first.label, second.label
    );
}
