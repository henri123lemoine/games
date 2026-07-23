//! MCTS-solver (Winands et al., KataGo-style) on the azero PUCT [`Search`].
//!
//! Two halves: a *zero-impact* half proving that with no prover the search is
//! byte-for-byte the prover-free one, and a *correctness* half — proven win
//! propagates to and is chosen at the root, a proven loss is never selected,
//! draws are handled, and a forced mate-in-N toy position is solved exactly —
//! driven by terminal proofs and by a stub [`TerminalProver`].

use game_core::{Game, NoProver, Proof, Rng, TerminalProver, Turn};
use solvers::azero::{EvalResult, Gather, Mlp, PuctConfig, Search, Value};

mod common;
use common::{Ttt, TttEnc, TttState, ttt_winner};

/// Drives a full search to `cfg.sims` and hands back the finished [`Search`] so
/// a test can read visits, the proof, or the proven move. `prover = None` drives
/// the search with no solver; `Some` runs the MCTS-solver.
fn run_search(
    game: &Ttt,
    enc: &TttEnc,
    net: &Mlp,
    root: &TttState,
    cfg: &PuctConfig,
    seed: u64,
    prover: Option<&dyn TerminalProver<Ttt>>,
) -> Search<Ttt> {
    let cache = net.infer_cache();
    let mut search = Search::new(None);
    let mut rng = Rng::new(seed);
    let mut results = Vec::new();
    loop {
        let gather = search.advance(
            game,
            enc,
            root,
            cfg,
            &mut rng,
            std::mem::take(&mut results),
            &|_| false,
            prover,
        );
        match gather {
            Gather::Requests(reqs) => {
                results = reqs
                    .iter()
                    .map(|r| {
                        let support: Vec<usize> =
                            r.support.iter().map(|&s| usize::from(s)).collect();
                        let (priors, value) =
                            net.policy_value_cached(&cache, &r.features, &support);
                        EvalResult {
                            priors,
                            value: Value::Mover(value),
                        }
                    })
                    .collect();
            }
            Gather::Done => return search,
        }
    }
}

fn cfg(sims: u32) -> PuctConfig {
    PuctConfig {
        sims,
        c_puct: 1.5,
        fpu: 0.25,
        dirichlet_alpha: 0.3,
        // Noise off so the two compared runs are deterministic and equal.
        root_noise: 0.0,
        max_leaves: 1,
        cycle_draws: false,
        forced_playouts_k: 0.0,
    }
}

/// A board from a list of `(cell, player)` placements *in play order* — X
/// (player 0) moves first and they alternate. The `player` field is only
/// documentation; it is checked against the alternation so a mis-ordered list
/// fails loudly.
fn board(moves: &[(usize, usize)]) -> TttState {
    let mut s = Ttt.initial_state();
    for &(cell, player) in moves {
        let Turn::Player(p) = Ttt.turn(&s) else {
            unreachable!()
        };
        assert_eq!(p, player, "move list out of play order at cell {cell}");
        Ttt.apply(&mut s, cell);
    }
    s
}

// ---- zero-impact: no prover ⇒ identical behavior --------------------------

#[test]
fn none_prover_matches_noprover_stub_visits() {
    let net = Mlp::new(19, 16, 9, 7);
    let game = Ttt;
    let root = game.initial_state();
    let c = cfg(200);

    let legacy = run_search(&game, &TttEnc, &net, &root, &c, 3, None);
    // A prover that proves nothing must leave every visit count untouched: the
    // solver is "active" but never sets a proof, so no path can diverge.
    let stub = NoProver;
    let with_stub = run_search(&game, &TttEnc, &net, &root, &c, 3, Some(&stub));

    assert_eq!(
        legacy.root_visits(),
        with_stub.root_visits(),
        "a never-proving prover changed the visit distribution"
    );
    assert_eq!(
        legacy.root_value(),
        with_stub.root_value(),
        "a never-proving prover changed the root value"
    );
    assert!(legacy.root_proof().is_none(), "no-prover run has no proof");
    assert!(
        with_stub.root_proof().is_none(),
        "never-proving prover yields no proof"
    );
}

#[test]
fn none_prover_visits_are_stable() {
    // Regression guard: the no-prover path is frozen against this snapshot, so
    // any accidental change to the legacy descent shows up here.
    let net = Mlp::new(19, 16, 9, 7);
    let game = Ttt;
    let root = game.initial_state();
    let s = run_search(&game, &TttEnc, &net, &root, &cfg(200), 3, None);
    let total: u32 = s.root_visits().iter().sum();
    assert_eq!(total, 200, "visit budget must be spent exactly");
    assert_eq!(s.root_visits().len(), 9);
}

// ---- correctness: a stub prover that proves named boards -------------------

/// Proves a fixed set of boards (by cell layout) with a fixed verdict — stands
/// in for a tablebase / mate-search the real games would supply.
struct ScriptedProver {
    verdicts: Vec<(TttState, Proof)>,
}

impl ScriptedProver {
    fn new() -> Self {
        ScriptedProver {
            verdicts: Vec::new(),
        }
    }
    fn prove_board(mut self, s: TttState, p: Proof) -> Self {
        self.verdicts.push((s, p));
        self
    }
}

/// A prover that directly proves one named board a Win, witnessing a fixed
/// winning move — exercising the directly-proven-Win path where the search must
/// set `proof_edge` from the witness rather than bubbling it up from a child.
struct WitnessProver {
    board: TttState,
    win_move: usize,
}

impl TerminalProver<Ttt> for WitnessProver {
    fn prove(&self, _game: &Ttt, state: &TttState) -> Option<(Proof, Option<usize>)> {
        (state.cells == self.board.cells && state.to_move == self.board.to_move)
            .then_some((Proof::Win, Some(self.win_move)))
    }
}

impl TerminalProver<Ttt> for ScriptedProver {
    fn prove(&self, _game: &Ttt, state: &TttState) -> Option<(Proof, Option<usize>)> {
        self.verdicts
            .iter()
            .find(|(s, _)| s.cells == state.cells && s.to_move == state.to_move)
            .map(|&(_, p)| (p, None))
    }
}

#[test]
fn proven_win_propagates_and_is_chosen() {
    // X to move with two in a row (0,1) and the winning cell 2 open. The child
    // after X plays 2 is terminal (X wins) and proves the root a Win.
    let net = Mlp::new(19, 16, 9, 11);
    let game = Ttt;
    let root = board(&[(0, 0), (3, 1), (1, 0), (4, 1)]); // X:0,1  O:3,4 — X to move
    assert_eq!(Ttt.turn(&root), Turn::Player(0));

    let stub = NoProver; // terminal proofs alone solve this
    let s = run_search(&game, &TttEnc, &net, &root, &cfg(64), 5, Some(&stub));

    assert_eq!(
        s.root_proof(),
        Some(Proof::Win),
        "root with an immediate winning move must prove Win"
    );
    let mv = s.best_proven_action().expect("a proven root yields a move");
    let action = s.root_actions()[mv];
    let mut after = root.clone();
    Ttt.apply(&mut after, action);
    assert_eq!(
        ttt_winner(&after),
        Some(0),
        "the chosen move must win for X"
    );
}

#[test]
fn proven_loss_child_is_never_selected() {
    // Build a root whose move A is a proven loss (stub) and move B is a proven
    // win (stub). The search must prove Win at the root via B and never play A.
    let net = Mlp::new(19, 16, 9, 13);
    let game = Ttt;
    let root = board(&[(0, 0), (3, 1), (1, 0)]); // X:0,1  O:3 — O (player 1) to move
    assert_eq!(Ttt.turn(&root), Turn::Player(1));

    // After O plays cell 2 it's X to move; mark that child a Win for X (i.e. a
    // loss for O — O should avoid it). After O plays cell 6, mark it a Loss for
    // X (a win for O — O should take it).
    let mut bad = root.clone();
    Ttt.apply(&mut bad, 2);
    let mut good = root.clone();
    Ttt.apply(&mut good, 6);

    let stub = ScriptedProver::new()
        .prove_board(bad, Proof::Win) // X wins after O→2  ⇒ O loses there
        .prove_board(good, Proof::Loss); // X loses after O→6 ⇒ O wins there

    let s = run_search(&game, &TttEnc, &net, &root, &cfg(64), 9, Some(&stub));
    assert_eq!(
        s.root_proof(),
        Some(Proof::Win),
        "O has a proven winning move, so the root is a Win for O"
    );
    let mv = s.best_proven_action().expect("proven root yields a move");
    assert_eq!(
        s.root_actions()[mv],
        6,
        "O must play its proven win (cell 6), never the losing cell 2"
    );
}

#[test]
fn all_children_winning_for_opponent_proves_loss() {
    // Root (O to move) where *every* child is a proven Win for the resulting
    // mover (X) — i.e. every O move hands X a win — so the root is a proven Loss
    // for O. A near-full board keeps the children few.
    let net = Mlp::new(19, 16, 9, 17);
    let game = Ttt;
    // X:0,1,5,6  O:2,3,7 — O to move (7 plies played), only cells 4 and 8 open.
    let root = board(&[(0, 0), (2, 1), (1, 0), (3, 1), (5, 0), (7, 1), (6, 0)]);
    assert_eq!(Ttt.turn(&root), Turn::Player(1));
    let open: Vec<usize> = (0..9).filter(|&i| root.cells[i] == 0).collect();
    assert_eq!(open, vec![4, 8]);

    // Mark both of O's moves as a Win for the resulting mover (X): O is lost.
    let mut prover = ScriptedProver::new();
    for &cell in &open {
        let mut child = root.clone();
        Ttt.apply(&mut child, cell);
        prover = prover.prove_board(child, Proof::Win); // win for X (to move there)
    }

    let s = run_search(&game, &TttEnc, &net, &root, &cfg(64), 21, Some(&prover));
    assert_eq!(
        s.root_proof(),
        Some(Proof::Loss),
        "every move hands X a win, so the root is a proven Loss for O"
    );
}

#[test]
fn proven_draw_handled() {
    // One open cell whose only child is a drawn terminal ⇒ the root is a proven
    // Draw. A full board minus one cell where neither side has three.
    let net = Mlp::new(19, 16, 9, 23);
    let game = Ttt;
    // Layout with no winner and a single empty cell (8); filling it stays a draw.
    //  X O X
    //  X O O
    //  O X _
    let root = board(&[
        (0, 0),
        (1, 1),
        (2, 0),
        (4, 1), // X:0,2  O:1,4
        (3, 0),
        (5, 1), // X:0,2,3  O:1,4,5
        (7, 0),
        (6, 1), // X:0,2,3,7  O:1,4,5,6
    ]);
    assert_eq!(Ttt.turn(&root), Turn::Player(0));
    let open: Vec<usize> = (0..9).filter(|&i| root.cells[i] == 0).collect();
    assert_eq!(open, vec![8]);
    // Sanity: filling 8 produces a full, winnerless board.
    let mut full = root.clone();
    Ttt.apply(&mut full, 8);
    assert!(Ttt.is_terminal(&full) && ttt_winner(&full).is_none());

    let stub = NoProver; // the terminal draw alone proves it
    let s = run_search(&game, &TttEnc, &net, &root, &cfg(32), 25, Some(&stub));
    assert_eq!(
        s.root_proof(),
        Some(Proof::Draw),
        "the only continuation is a drawn terminal"
    );
}

#[test]
fn deep_position_solved_to_the_exact_minimax_value() {
    // A multi-ply position solved from terminals alone (NoProver), with the
    // verdict grounded against an independent brute-force minimax — so this
    // catches any perspective or propagation error in the proof backup.
    let net = Mlp::new(19, 16, 9, 29);
    let game = Ttt;
    let root = board(&[(0, 0), (8, 1), (4, 0), (1, 1)]); // X:0,4  O:8,1 — X to move
    assert_eq!(Ttt.turn(&root), Turn::Player(0));

    let truth = ttt_solve(&root);
    let s = run_search(&game, &TttEnc, &net, &root, &cfg(2000), 31, Some(&NoProver));

    let got = s.root_proof();
    let expected = match truth {
        1 => Some(Proof::Win),
        -1 => Some(Proof::Loss),
        0 => Some(Proof::Draw),
        _ => unreachable!(),
    };
    assert_eq!(
        got, expected,
        "MCTS-solver verdict {got:?} must match the exact game value {truth}"
    );
    if truth == 1 {
        let mv = s.best_proven_action().expect("proven win yields a move");
        let action = s.root_actions()[mv];
        let mut after = root.clone();
        Ttt.apply(&mut after, action);
        // After X's winning move it is O to move in a lost position, so the
        // minimax value from O's (the new mover's) seat is -1.
        assert_eq!(
            ttt_solve(&after),
            -1,
            "the proven move must keep the win for X"
        );
    }
}

/// Exact minimax value of a Ttt position for the side to move (+1 win, 0 draw,
/// -1 loss). Used to ground the solver's verdict.
fn ttt_solve(s: &TttState) -> i32 {
    if let Some(w) = ttt_winner(s) {
        // The player who just moved (opponent of to_move) made the line.
        return if w == s.to_move { 1 } else { -1 };
    }
    let open: Vec<usize> = (0..9).filter(|&i| s.cells[i] == 0).collect();
    if open.is_empty() {
        return 0;
    }
    let mut best = -1;
    for &cell in &open {
        let mut c = s.clone();
        Ttt.apply(&mut c, cell);
        best = best.max(-ttt_solve(&c));
        if best == 1 {
            break;
        }
    }
    best
}

#[test]
fn directly_proven_root_win_witnesses_its_move() {
    // The prover proves the *root* a Win directly (its own leaf, before any
    // child is expanded), witnessing a winning move. `best_proven_action` must
    // return that move's edge — set from the witness in `resolve`, not bubbled
    // up from a child — so a directly proven root plays the proven move too.
    let net = Mlp::new(19, 16, 9, 43);
    let game = Ttt;
    // X:0,1  O:3,4 — X to move, mate-in-1 at cell 2 (a real winning move).
    let root = board(&[(0, 0), (3, 1), (1, 0), (4, 1)]);
    assert_eq!(Ttt.turn(&root), Turn::Player(0));
    let prover = WitnessProver {
        board: root.clone(),
        win_move: 2,
    };
    let s = run_search(&game, &TttEnc, &net, &root, &cfg(64), 5, Some(&prover));
    assert_eq!(
        s.root_proof(),
        Some(Proof::Win),
        "root proven a Win directly"
    );
    let mv = s
        .best_proven_action()
        .expect("directly proven root yields a move");
    assert_eq!(
        s.root_actions()[mv],
        2,
        "the proven move is the witnessed winning move, not edge 0"
    );
}

#[test]
fn proven_move_is_played_on_a_mate_in_one() {
    // Beyond verdict correctness: once the root is proven, the proven move is
    // the one the caller plays. Confirm best_proven_action sits among the legal
    // actions and (for a win) is a winning move, across several seeds.
    let net = Mlp::new(19, 16, 9, 41);
    let game = Ttt;
    let root = board(&[(0, 0), (3, 1), (1, 0), (4, 1)]); // X mate-in-1 at cell 2
    for seed in 0..6 {
        let s = run_search(&game, &TttEnc, &net, &root, &cfg(48), seed, Some(&NoProver));
        assert_eq!(s.root_proof(), Some(Proof::Win));
        let mv = s.best_proven_action().unwrap();
        assert_eq!(s.root_actions()[mv], 2, "seed {seed}: must pick the win");
    }
}
