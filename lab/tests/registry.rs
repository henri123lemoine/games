//! The option-parsing seams: spec splitting, the unused-option guard, and
//! the typo rejection the compare paths now enforce.

use std::collections::HashMap;

use lab::compare::split_specs;
use lab::registry::{Opts, entries};

fn opts(kvs: &[(&str, &str)]) -> Opts {
    Opts::new(
        kvs.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>(),
    )
}

#[test]
fn split_specs_keeps_spec_internal_commas() {
    assert_eq!(split_specs("a,b"), ["a", "b"]);
    assert_eq!(
        split_specs("azero:net=x.bin,sims=256,alphabeta:depth=5"),
        ["azero:net=x.bin,sims=256", "alphabeta:depth=5"]
    );
    assert_eq!(split_specs("alphabeta"), ["alphabeta"]);
}

#[test]
fn unused_options_are_reported() {
    let o = opts(&[("depth", "5"), ("depht", "7")]);
    assert_eq!(o.get("depth", 1).unwrap(), 5);
    assert_eq!(o.unused(), ["depht"]);
    assert!(o.ensure_consumed("test").is_err());
    assert_eq!(o.get("depht", 0).unwrap(), 7);
    assert!(o.ensure_consumed("test").is_ok());
}

#[test]
fn compare_paths_reject_bot_spec_typos() {
    let eval = entries()
        .into_iter()
        .find(|e| e.id == "connect4")
        .and_then(|e| e.eval)
        .unwrap();
    let err = (eval.pairs)(&opts(&[]), "alphabeta:depht=7", "alphabeta", 1, 0..0)
        .expect_err("misspelled spec option must error");
    assert!(err.contains("depht"), "error names the typo: {err}");

    let err = (eval.pairs)(&opts(&[("depth", "7")]), "alphabeta", "alphabeta", 1, 0..0)
        .expect_err("game-level depth is not a bot option in compare");
    assert!(err.contains("depth"), "error names the key: {err}");

    let ok = (eval.pairs)(&opts(&[]), "alphabeta:depth=2", "alphabeta", 1, 0..0);
    assert_eq!(ok.unwrap(), (0, 0, 0), "an empty pair range plays nothing");
}

#[test]
fn manifest_fields_cover_every_entry() {
    for e in entries() {
        assert!(!e.name.is_empty(), "{} needs a display name", e.id);
        assert!(
            !e.solo || !e.watch_bot.is_empty(),
            "solo game {} needs a watch bot",
            e.id
        );
    }
}
