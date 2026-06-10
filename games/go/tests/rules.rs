use game_core::{Game, GameUi, Rng, Turn};
use go::{Go, GoAction};

fn play(g: &Go, s: &mut <Go as Game>::State, coord: &str) {
    let a = g
        .parse_action(s, coord)
        .unwrap_or_else(|| panic!("{coord} should be legal"));
    g.apply(s, a);
}

fn place_at(g: &Go, coord: &str) -> GoAction {
    GoAction::Place(g.point(coord).unwrap())
}

fn is_empty(g: &Go, s: &<Go as Game>::State, coord: &str) -> bool {
    s.stone(g.point(coord).unwrap() as usize).is_none()
}

#[test]
fn lone_stone_without_liberties_is_captured() {
    let g = Go::new(5);
    let mut s = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". X O X .",
            ". . X . .",
            ". . . . .",
        ],
        0,
    );
    assert!(matches!(g.turn(&s), Turn::Player(0)));
    play(&g, &mut s, "c4");
    assert!(is_empty(&g, &s, "c3"));
    assert_eq!(s.captures(), [1, 0]);
}

#[test]
fn corner_group_is_captured() {
    let g = Go::new(5);
    let mut s = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". . . . .",
            "X X . . .",
            "O O . . .",
        ],
        0,
    );
    play(&g, &mut s, "c1");
    assert!(is_empty(&g, &s, "a1"));
    assert!(is_empty(&g, &s, "b1"));
    assert_eq!(s.captures(), [2, 0]);
}

#[test]
fn suicide_is_rejected() {
    let g = Go::new(5);
    let s = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". . . . .",
            "X . . . .",
            ". X . . .",
        ],
        1,
    );
    assert!(!g.legal_actions(&s).contains(&place_at(&g, "a1")));
    assert!(g.parse_action(&s, "a1").is_none());
    let black_view = g.parse_state(
        &[
            ". . . . .",
            ". . . . .",
            ". . . . .",
            "X . . . .",
            ". X . . .",
        ],
        0,
    );
    assert!(g.legal_actions(&black_view).contains(&place_at(&g, "a1")));
}

#[test]
fn simple_ko_is_rejected_until_a_move_intervenes() {
    let g = Go::new(5);
    let mut s = g.parse_state(
        &[
            ". . . . .",
            ". . X O .",
            ". X O . O",
            ". . X O .",
            ". . . . .",
        ],
        0,
    );
    play(&g, &mut s, "d3");
    assert!(is_empty(&g, &s, "c3"));

    let retake = place_at(&g, "c3");
    assert!(!g.legal_actions(&s).contains(&retake));
    assert!(g.parse_action(&s, "c3").is_none());

    let same_board_no_ko = g.parse_state(
        &[
            ". . . . .",
            ". . X O .",
            ". X . X O",
            ". . X O .",
            ". . . . .",
        ],
        1,
    );
    assert_ne!(
        g.state_key(&s).unwrap(),
        g.state_key(&same_board_no_ko).unwrap(),
        "state key must encode the ko restriction"
    );

    play(&g, &mut s, "e1");
    play(&g, &mut s, "a5");
    assert!(g.legal_actions(&s).contains(&retake));
    play(&g, &mut s, "c3");
    assert!(is_empty(&g, &s, "d3"));
    assert!(!g.legal_actions(&s).contains(&place_at(&g, "d3")));
}

#[test]
fn two_passes_end_the_game_with_area_scoring() {
    let g = Go::new(5);
    let mut s = g.parse_state(
        &[
            ". . X O .",
            ". . X O .",
            ". . X O .",
            ". . X O .",
            ". . X O .",
        ],
        1,
    );
    g.apply(&mut s, GoAction::Pass);
    assert!(!g.is_terminal(&s));
    g.apply(&mut s, GoAction::Pass);
    assert!(g.is_terminal(&s));
    assert_eq!(g.area_scores(&s), (15, 10));
    assert_eq!(g.returns(&s, 0), -1.0, "white wins on komi: 10 + 7.5 > 15");
    assert_eq!(g.returns(&s, 1), 1.0);

    let mut s = g.parse_state(
        &[
            ". . . X O",
            ". . . X O",
            ". . . X O",
            ". . . X O",
            ". . . X O",
        ],
        0,
    );
    g.apply(&mut s, GoAction::Pass);
    g.apply(&mut s, GoAction::Pass);
    assert!(g.is_terminal(&s));
    assert_eq!(g.area_scores(&s), (20, 5));
    assert_eq!(g.returns(&s, 0), 1.0);
}

#[test]
fn state_key_includes_side_to_move() {
    let g = Go::new(5);
    let s0 = g.initial_state();
    let mut s1 = s0.clone();
    g.apply(&mut s1, GoAction::Pass);
    assert_ne!(g.state_key(&s0).unwrap(), g.state_key(&s1).unwrap());
    assert_eq!(g.state_key(&s0).unwrap(), g.infoset_key(&s0, 0));
    assert_eq!(g.infoset_key(&s0, 0), g.infoset_key(&s0, 1));
}

#[test]
fn random_playthroughs_terminate() {
    for size in [5, 9] {
        let g = Go::new(size);
        let cap = 4 * size * size;
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..20 {
            let mut s = g.initial_state();
            let mut plies = 0;
            while !g.is_terminal(&s) {
                assert!(plies < cap, "draw-guard cap exceeded on {size}x{size}");
                let actions = g.legal_actions(&s);
                assert!(!actions.is_empty());
                let i = ((rng.unit() * actions.len() as f64) as usize).min(actions.len() - 1);
                g.apply(&mut s, actions[i]);
                plies += 1;
            }
            let r = g.returns(&s, 0);
            assert!(r == 1.0 || r == -1.0, "fractional komi forbids draws");
            assert_eq!(g.returns(&s, 1), -r);
        }
    }
}

#[test]
fn ply_cap_ends_a_cycling_game() {
    let g = Go::new(2);
    let mut s = g.initial_state();
    let mut plies = 0;
    while !g.is_terminal(&s) {
        assert!(plies < 16, "cap is 4 * 2 * 2 = 16");
        let actions = g.legal_actions(&s);
        g.apply(&mut s, actions[0]);
        plies += 1;
    }
    assert_eq!(
        plies, 16,
        "first-placement policy cycles on 2x2 until capped"
    );
    let r = g.returns(&s, 0);
    assert!(r == 1.0 || r == -1.0);
}
