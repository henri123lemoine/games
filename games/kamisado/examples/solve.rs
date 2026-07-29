//! Weakly solves single-round Kamisado from the opening position.
//!
//! The tool is the lab's own generic alpha-beta: mate-scaled terminal scores
//! make a decisive root score a *proof* (a heuristic leaf can never reach the
//! mate scale, so a mate-scale minimax value means every opponent alternative
//! was refuted by an actual terminal win), while the heuristic evaluation
//! keeps the windows tight enough to prune the undecided parts of the tree —
//! the trick a pure win/loss/unknown prover lacks, which is why it stalls.
//! Iterative deepening reports the smallest action depth that decides the
//! opening.
//!
//! Kamisado is friendly to this: no draws, obligation chains collapse into
//! non-alternating turns, every action advances a tower (so no repetitions,
//! rounds end within 112 actions), and the forced-tower rule keeps branching
//! near 10.
//!
//! The known result (the hamisado project, echoed on Wikipedia) is a
//! first-player win, proven there at depth 17 of its move counting, which
//! includes explicit "pass" plies for blocked obligations; depths here count
//! actual tower moves only, so the same proof lands a little shallower.
//!
//! After the alpha-beta proof, an independent verifier re-derives the result
//! with none of that machinery: a plain bounded AND-OR search over the rules
//! (exists a Black move / for all White moves), memoized on state keys. Run
//! against budgets 15, 16 and 17 it establishes the forced-win depth exactly.
//!
//! ```bash
//! cargo run --release -p kamisado --example solve             # prove + verify the opening
//! cargo run --release -p kamisado --example solve -- 12 all   # + classify all 102 openings at depth 12
//! ```

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use game_core::{Game, GameUi, SearchSpec, Turn};
use kamisado::{Kamisado, KamisadoEval, KamisadoMove, KamisadoSpec, KamisadoState};
use solvers::{AlphaBeta, loss_distance, win_distance};

type Search = AlphaBeta<Kamisado, KamisadoEval, KamisadoSpec>;

/// Memo keys are the *exact* canonical 101-bit position identities (no
/// hashing on the equality path, so the verification cannot be corrupted by
/// a key collision); the hasher only buckets them.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("u128 keys only");
    }
    fn write_u128(&mut self, v: u128) {
        self.0 = game_core::hash::combine(v as u64, (v >> 64) as u64);
    }
}

const NO_REFUTER: u16 = u16::MAX;

/// Memo for the verifier: per canonical position, the smallest budget known
/// to be a Black win, the largest known not to be, and — for refuted White
/// nodes — the reply that refuted, stored in canonical orientation and tried
/// first when the node is re-searched at a higher budget.
#[derive(Clone, Copy)]
struct Bounds {
    win_within: u8,
    no_win_within: u8,
    refuter: u16,
}

struct Verifier {
    game: Kamisado,
    memo: HashMap<u128, Bounds, BuildHasherDefault<IdHasher>>,
    nodes: u64,
}

impl Verifier {
    fn new() -> Self {
        Verifier {
            game: Kamisado,
            memo: HashMap::default(),
            nodes: 0,
        }
    }

    fn store(&mut self, key: u128, update: impl FnOnce(&mut Bounds)) {
        let entry = self.memo.entry(key).or_insert(Bounds {
            win_within: u8::MAX,
            no_win_within: 0,
            refuter: NO_REFUTER,
        });
        update(entry);
    }

    /// Does Black force a win within `budget` further actions? Exact bounded
    /// AND-OR search: no evaluation, no pruning beyond the bound itself, so a
    /// `true` here is a proof from the rules alone.
    fn black_wins_within(&mut self, state: &KamisadoState, budget: u8) -> bool {
        self.nodes += 1;
        if self.game.is_terminal(state) {
            return self.game.returns(state, 0) > 0.0;
        }
        if budget == 0 {
            return false;
        }
        let (key, mirrored) = state.canonical();
        let mut refuter = NO_REFUTER;
        if let Some(b) = self.memo.get(&key) {
            if b.win_within <= budget {
                return true;
            }
            if b.no_win_within >= budget {
                return false;
            }
            refuter = b.refuter;
        }
        let mover = mover_of(&self.game, state);
        let goal = if mover == 0 { 7 } else { 0 };
        let mut acts = self.game.legal_actions(state);
        // A move onto the goal rank decides the node without any recursion:
        // Black wins outright; a White winning reply refutes at every budget.
        if acts.iter().any(|a| a.to >> 3 == goal) {
            if mover == 0 {
                self.store(key, |b| b.win_within = b.win_within.min(1));
                return true;
            }
            self.store(key, |b| b.no_win_within = u8::MAX);
            return false;
        }
        if mover == 0 {
            // Try the most forcing moves first — only Black's side benefits
            // from ordering, White's replies must all be refuted anyway.
            acts.sort_by_cached_key(|&a| -KamisadoSpec.order_hint(&self.game, state, a));
        } else if refuter != NO_REFUTER {
            // The reply that refuted at a lower budget usually still does.
            let flip = if mirrored { 7 } else { 0 };
            let mv = KamisadoMove {
                from: (refuter >> 8) as u8 ^ flip,
                to: refuter as u8 ^ flip,
            };
            if let Some(pos) = acts.iter().position(|&a| a == mv) {
                acts.swap(0, pos);
            }
        }
        let mut win = mover != 0;
        let mut refuted_by = NO_REFUTER;
        for a in acts {
            let mut child = state.clone();
            self.game.apply(&mut child, a);
            let w = self.black_wins_within(&child, budget - 1);
            if mover == 0 && w {
                win = true;
                break;
            }
            if mover != 0 && !w {
                win = false;
                let flip = if mirrored { 7 } else { 0 };
                refuted_by = u16::from(a.from ^ flip) << 8 | u16::from(a.to ^ flip);
                break;
            }
        }
        self.store(key, |b| {
            if win {
                b.win_within = b.win_within.min(budget);
            } else {
                b.no_win_within = b.no_win_within.max(budget);
                if refuted_by != NO_REFUTER {
                    b.refuter = refuted_by;
                }
            }
        });
        win
    }
}

fn mover_of(game: &Kamisado, state: &KamisadoState) -> usize {
    match game.turn(state) {
        Turn::Player(p) => p,
        Turn::Chance => unreachable!(),
    }
}

/// Replay the proof line: at each position re-search to the remaining mate
/// distance (hot transposition table, so this is cheap) and play the move.
fn proof_line(game: &Kamisado, ab: &mut Search, root: &KamisadoState) -> Vec<String> {
    let mut line = Vec::new();
    let mut state = root.clone();
    while !game.is_terminal(&state) && line.len() < 128 {
        let (i, score) = ab.best_scored(game, &state);
        let dist = match (win_distance(score), loss_distance(score)) {
            (Some(d), _) | (_, Some(d)) => d,
            _ => break, // line drifted off the proof — should not happen
        };
        ab.depth = dist.max(1);
        let action = game.legal_actions(&state)[i];
        let mover = if mover_of(game, &state) == 0 {
            "B"
        } else {
            "W"
        };
        line.push(format!("{mover}:{}", game.action_label(&state, action)));
        game.apply(&mut state, action);
    }
    line
}

fn main() {
    let mut max_depth: u32 = 40;
    let mut classify_all = false;
    let mut verify = true;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "all" => classify_all = true,
            "no-verify" => verify = false,
            d => max_depth = d.parse().expect("args: [max depth] [all] [no-verify]"),
        }
    }

    let game = Kamisado;
    let root = game.initial_state();
    let mut ab = Search::new(1, KamisadoEval, KamisadoSpec);
    let start = Instant::now();
    let mut nodes = 0u64;
    let mut proven = None;

    for depth in 1..=max_depth {
        ab.depth = depth;
        let (best, score) = ab.best_scored(&game, &root);
        nodes += ab.node_count();
        let verdict = match (win_distance(score), loss_distance(score)) {
            (Some(d), _) => format!("WIN in {d}"),
            (_, Some(d)) => format!("LOSS in {d}"),
            _ => format!("eval {score:+.3}"),
        };
        println!(
            "depth {depth:>2}  {verdict:<12}  best {:<7}  nodes {nodes:>12}  {:.1}s",
            game.action_label(&root, game.legal_actions(&root)[best]),
            start.elapsed().as_secs_f64()
        );
        if win_distance(score).is_some() || loss_distance(score).is_some() {
            proven = Some((depth, score));
            break;
        }
    }

    let Some((depth, score)) = proven else {
        println!("\nNot decided within depth {max_depth}.");
        return;
    };
    match win_distance(score) {
        Some(d) => println!(
            "\nKamisado is a first-player (Black) WIN: forced in {d} actions, found at search depth {depth}."
        ),
        None => {
            println!("\nKamisado is a first-player (Black) LOSS, proven at search depth {depth}.")
        }
    }
    ab.depth = depth;
    println!(
        "proof line: {}",
        proof_line(&game, &mut ab, &root).join(" ")
    );

    if let Some(dist) = win_distance(score).filter(|_| verify) {
        println!("\nIndependent verification (bounded AND-OR search over the rules alone):");
        let mut verifier = Verifier::new();
        for budget in dist.saturating_sub(2)..=dist {
            let t = Instant::now();
            let won = verifier.black_wins_within(&root, budget as u8);
            println!(
                "  win within {budget:>2} actions: {won:<5}  nodes {:>12}  {:.1}s",
                verifier.nodes,
                t.elapsed().as_secs_f64()
            );
        }
    }

    if classify_all {
        let d = max_depth.saturating_sub(1).max(1);
        println!("\nOpenings for Black, each searched to depth {d} (verdict for Black):");
        let (mut wins, mut losses, mut open) = (0, 0, 0);
        for a in game.legal_actions(&root) {
            let mut child = root.clone();
            game.apply(&mut child, a);
            ab.depth = d;
            let (_, s) = ab.best_scored(&game, &child);
            // Score is from the perspective of whoever moves in `child`.
            let for_black = if mover_of(&game, &child) == 0 { s } else { -s };
            let verdict = match (win_distance(for_black), loss_distance(for_black)) {
                (Some(k), _) => {
                    wins += 1;
                    format!("win in {}", k + 1)
                }
                (_, Some(k)) => {
                    losses += 1;
                    format!("loss in {}", k + 1)
                }
                _ => {
                    open += 1;
                    "undecided".into()
                }
            };
            println!("  {}  {verdict}", game.action_label(&root, a));
        }
        println!("({wins} winning, {losses} losing, {open} undecided at this depth)");
    }
}
