use game_core::{Eval, Game, SearchSpec};
use go::{Go, GoAction, GoEval, GoSpec};

fn hint(g: &Go, s: &<Go as Game>::State, coord: &str) -> i64 {
    GoSpec.order_hint(g, s, GoAction::Place(g.point(coord).unwrap()))
}

#[test]
fn capture_orders_first() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". . X . .",
            ". X O X .",
            ". . . . .",
            ". . . . .",
        ],
        0,
    );
    let capture = hint(&g, &s, "c2");
    assert!(capture > 1_000, "capture hint {capture}");
    for a in g.legal_actions(&s) {
        assert!(GoSpec.order_hint(&g, &s, a) <= capture);
    }
}

#[test]
fn bigger_captures_order_earlier() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". X X . .",
            "X O O X .",
            ". X . X .",
            ". . . . .",
        ],
        0,
    );
    let two_stone = hint(&g, &s, "c2");
    let s1 = g.parse_state(
        &[
            ". . . . .",
            ". . X . .",
            ". X O X .",
            ". . . . .",
            ". . . . .",
        ],
        0,
    );
    assert!(two_stone > hint(&g, &s1, "c2"));
}

#[test]
fn escaping_atari_beats_quiet_moves() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". O X O .",
            ". . O . .",
            ". . . . .",
        ],
        0,
    );
    let escape = hint(&g, &s, "c4");
    let quiet = hint(&g, &s, "a1");
    assert!(escape >= 800, "escape hint {escape}");
    assert_eq!(quiet, 0);
}

#[test]
fn extending_into_a_second_atari_is_no_escape() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". O . O .",
            ". O X O .",
            ". . O . .",
            ". . . . .",
        ],
        0,
    );
    assert!(hint(&g, &s, "c4") < 0, "self-atari extension must rank low");
}

#[test]
fn putting_opponent_in_atari_beats_quiet_moves() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". X O X .",
            ". . . . .",
            ". . . . .",
        ],
        0,
    );
    let threat = hint(&g, &s, "c4");
    assert!((600..1_000).contains(&threat), "threat hint {threat}");
    assert_eq!(hint(&g, &s, "a1"), 0);
}

#[test]
fn filling_own_true_eye_ranks_below_pass() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". . X . .",
            ". X . X .",
            ". . X . .",
            ". . . . .",
        ],
        0,
    );
    let eye_fill = hint(&g, &s, "c3");
    let pass = GoSpec.order_hint(&g, &s, GoAction::Pass);
    assert!(eye_fill < pass);
    assert!(eye_fill < -1_000);
}

#[test]
fn filling_a_false_eye_is_not_penalized() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". O X O .",
            ". X . X .",
            ". . X . .",
            ". . . . .",
        ],
        0,
    );
    assert!(hint(&g, &s, "c3") >= 0);
}

#[test]
fn eval_is_komi_aware_and_antisymmetric() {
    let g = Go::new(9);
    let s = g.initial_state();
    let black = GoEval.eval(&g, &s, 0);
    let white = GoEval.eval(&g, &s, 1);
    assert!(black < 0.0, "komi means an even board favors White");
    assert!((black + white).abs() < 1e-12);

    let ahead = g.parse_state(
        &[
            "X X X X X X X X X",
            "X X X X X X X X X",
            "X X X X X X X X X",
            "X X X X X X X X X",
            ". . . . . . . . .",
            ". . . . . . . . .",
            "O O O O O O O O O",
            ". . . . . . . . .",
            ". . . . . . . . .",
        ],
        0,
    );
    assert!(GoEval.eval(&g, &ahead, 0) > 0.0);
    assert!(GoEval.eval(&g, &ahead, 0) < 1.0);
}
