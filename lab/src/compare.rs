//! The generic comparison driver: statistically rigorous bot-vs-bot
//! evaluation for any registered game. Two-player games run paired,
//! seat-swapped matches (same seed both orientations, optional shared random
//! opening) feeding a GSPRT; N-player games run hero-vs-field with rotated
//! seats feeding a binomial SPRT; tournaments are round-robins fitted to a
//! mean-anchored Elo scale. Games supply only a bot-spec parser.

use std::collections::HashMap;

use game_core::stats::{BinomialSprt, Sprt, Verdict, elo_estimate};
use game_core::{Agent, Game, Rng, Turn, play_n};
use rayon::prelude::*;

use crate::registry::Opts;

/// A parsed bot specification, e.g. `alphabeta:depth=5` or
/// `azero:net=data/azero/chess.bin,sims=256`.
pub struct BotSpec {
    pub name: String,
    pub opts: Opts,
}

pub fn parse_spec(s: &str) -> Result<BotSpec, String> {
    let (name, rest) = match s.split_once(':') {
        Some((n, r)) => (n, Some(r)),
        None => (s, None),
    };
    let mut map = HashMap::new();
    if let Some(rest) = rest {
        for kv in rest.split(',') {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("bot option must be key=value, got '{kv}' in '{s}'"))?;
            map.insert(k.to_string(), v.to_string());
        }
    }
    Ok(BotSpec {
        name: name.to_string(),
        opts: Opts(map),
    })
}

/// Splits `bots=` lists on commas while keeping commas that belong to a spec's
/// own options: a segment with `=` but no `:` continues the previous spec
/// (`azero:net=x.bin,sims=256` stays one spec).
pub fn split_specs(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for seg in s.split(',') {
        if seg.contains('=')
            && !seg.contains(':')
            && let Some(last) = out.last_mut()
        {
            last.push(',');
            last.push_str(seg);
        } else {
            out.push(seg.to_string());
        }
    }
    out
}

pub type BoxedAgent<G> = Box<dyn Agent<G>>;
/// Builds a fresh agent from a per-pair seed. Builders are shared across
/// rayon workers; the agents they return are used by one worker only, so
/// agents with interior mutability (MCTS) work unchanged. Expensive resources
/// (nets, trained solvers) should be loaded once at parse time and shared via
/// `Arc` inside the builder.
pub type BotBuilder<G> = Box<dyn Fn(u64) -> BoxedAgent<G> + Send + Sync>;
/// Game-supplied parser from a [`BotSpec`] (plus the game-level options) to a
/// [`BotBuilder`].
pub type BotParser<G> = fn(&BotSpec, &Opts) -> Result<BotBuilder<G>, String>;

pub struct CompareArgs {
    pub a: String,
    pub b: String,
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
    pub max_games: u64,
    pub batch: u64,
    /// Field mode (N>2 players): H1 win share is `1/players + delta`.
    pub delta: f64,
    pub seed: u64,
    pub opts: Opts,
}

pub struct TourneyArgs {
    pub bots: Vec<String>,
    pub games: u64,
    pub seed: u64,
    pub opts: Opts,
}

fn mix(seed: u64, k: u64) -> u64 {
    let mut x = seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) | 1
}

/// Plays one two-player game with `first` at seat 0 and returns seat 0's
/// terminal utility. The first `open_plies` decisions are uniform-random — a
/// seed-derived opening book, identical across the two orientations of a pair,
/// which decorrelates deterministic bots the way engine testers use openings.
fn play_scored<G: Game>(
    game: &G,
    first: &dyn Agent<G>,
    second: &dyn Agent<G>,
    open_plies: u64,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut s = game.initial_state();
    let mut plies = 0u64;
    while !game.is_terminal(&s) {
        match game.turn(&s) {
            Turn::Chance => {
                let outs = game.chance_outcomes(&s);
                let r = rng.unit();
                let mut acc = 0.0;
                let mut chosen = outs[outs.len() - 1].0;
                for (a, p) in &outs {
                    acc += *p;
                    if r < acc {
                        chosen = *a;
                        break;
                    }
                }
                game.apply(&mut s, chosen);
            }
            Turn::Player(p) => {
                let actions = game.legal_actions(&s);
                let r = rng.unit();
                let i = if plies < open_plies {
                    ((r * actions.len() as f64) as usize).min(actions.len() - 1)
                } else {
                    let agent = if p == 0 { first } else { second };
                    agent.act(game, &s, p, r)
                };
                game.apply(&mut s, actions[i]);
                plies += 1;
            }
        }
    }
    game.returns(&s, 0)
}

fn wdl(utility: f64) -> (u64, u64, u64) {
    if utility > 1e-9 {
        (1, 0, 0)
    } else if utility < -1e-9 {
        (0, 0, 1)
    } else {
        (0, 1, 0)
    }
}

/// `pair_count` seat-swapped pairs in parallel; W-D-L from A's perspective.
fn play_pairs<G: Game + Sync>(
    game: &G,
    a: &BotBuilder<G>,
    b: &BotBuilder<G>,
    open_plies: u64,
    seed: u64,
    pairs: std::ops::Range<u64>,
) -> (u64, u64, u64) {
    pairs
        .into_par_iter()
        .map(|k| {
            let s = mix(seed, k);
            let pa = a(s ^ 0xA11CE);
            let pb = b(s ^ 0xB0B);
            let u1 = play_scored(game, &*pa, &*pb, open_plies, s);
            let u2 = -play_scored(game, &*pb, &*pa, open_plies, s);
            let (w1, d1, l1) = wdl(u1);
            let (w2, d2, l2) = wdl(u2);
            (w1 + w2, d1 + d2, l1 + l2)
        })
        .reduce(|| (0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2))
}

/// Two-player compare: paired seat-swapped games into a GSPRT on
/// H0: elo = `elo0` vs H1: elo = `elo1`.
pub fn head_to_head<G: Game + Sync>(
    game: &G,
    args: &CompareArgs,
    default_open: u64,
    parse: BotParser<G>,
) -> Result<(), String> {
    let a = parse(&parse_spec(&args.a)?, &args.opts)?;
    let b = parse(&parse_spec(&args.b)?, &args.opts)?;
    let open = args.opts.get("open", default_open);
    let mut sprt = Sprt::new(args.elo0, args.elo1, args.alpha, args.beta);
    let max_pairs = (args.max_games / 2).max(1);
    let batch_pairs = (args.batch / 2).max(1);
    println!(
        "compare: '{}' vs '{}'  H0 elo={}  H1 elo={}  alpha={} beta={}  open={} plies  seed={}",
        args.a, args.b, args.elo0, args.elo1, args.alpha, args.beta, open, args.seed
    );
    let mut next = 0u64;
    while next < max_pairs {
        let hi = (next + batch_pairs).min(max_pairs);
        let (w, d, l) = play_pairs(game, &a, &b, open, args.seed, next..hi);
        next = hi;
        sprt.update(w, d, l);
        let (tw, td, tl) = sprt.counts();
        let e = elo_estimate(tw, td, tl);
        println!(
            "games {:>5}  {}-{}-{}  elo {:>+7.1} +/- {:>5.1}  llr {:>6.2}",
            sprt.games(),
            tw,
            td,
            tl,
            e.elo,
            e.margin(),
            sprt.llr()
        );
        if sprt.verdict() != Verdict::Open {
            break;
        }
    }
    let (tw, td, tl) = sprt.counts();
    let e = elo_estimate(tw, td, tl);
    match sprt.verdict() {
        Verdict::AcceptH1 => println!(
            "verdict: '{}' is stronger than '{}' — accepted H1 (elo >= {}) after {} games; \
             measured elo {:+.0} +/- {:.0}",
            args.a,
            args.b,
            args.elo1,
            sprt.games(),
            e.elo,
            e.margin()
        ),
        Verdict::RejectH1 => println!(
            "verdict: no evidence '{}' is stronger than '{}' — accepted H0 (elo <= {}) after {} \
             games; measured elo {:+.0} +/- {:.0}",
            args.a,
            args.b,
            args.elo0,
            sprt.games(),
            e.elo,
            e.margin()
        ),
        Verdict::Open => println!(
            "verdict: inconclusive after {} games (llr {:.2} inside [{:.2}, {:.2}]); measured \
             elo {:+.0} +/- {:.0} — raise max-games to decide",
            sprt.games(),
            sprt.llr(),
            sprt.bounds().0,
            sprt.bounds().1,
            e.elo,
            e.margin()
        ),
    }
    Ok(())
}

/// N-player compare: hero A rotated through every seat against a field of B,
/// binomial SPRT on H0: p = 1/n vs H1: p = 1/n + delta.
pub fn vs_field<G: Game + Sync>(
    game: &G,
    args: &CompareArgs,
    parse: BotParser<G>,
) -> Result<(), String> {
    let a = parse(&parse_spec(&args.a)?, &args.opts)?;
    let b = parse(&parse_spec(&args.b)?, &args.opts)?;
    let n = game.num_players();
    let p0 = 1.0 / n as f64;
    let p1 = (p0 + args.delta).min(1.0 - 1e-6);
    let mut sprt = BinomialSprt::new(p0, p1, args.alpha, args.beta);
    println!(
        "compare (field of {}): hero '{}' vs field of '{}'  H0 share={:.3}  H1 share={:.3}  \
         alpha={} beta={}  seed={}",
        n - 1,
        args.a,
        args.b,
        p0,
        p1,
        args.alpha,
        args.beta,
        args.seed
    );
    let batch = args.batch.max(1);
    let mut next = 0u64;
    while next < args.max_games {
        let hi = (next + batch).min(args.max_games);
        let (wins, losses) = (next..hi)
            .into_par_iter()
            .map(|g| {
                let s = mix(args.seed, g);
                let hero_seat = (g as usize) % n;
                let hero = a(s ^ 0xA11CE);
                let field: Vec<BoxedAgent<G>> = (0..n - 1)
                    .map(|i| b(s ^ 0xB0B ^ (i as u64) << 17))
                    .collect();
                let mut fi = 0;
                let agents: Vec<&dyn Agent<G>> = (0..n)
                    .map(|p| {
                        if p == hero_seat {
                            &*hero
                        } else {
                            fi += 1;
                            &*field[fi - 1]
                        }
                    })
                    .collect();
                if play_n(game, &agents, &mut Rng::new(s)) == hero_seat {
                    (1u64, 0u64)
                } else {
                    (0, 1)
                }
            })
            .reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
        next = hi;
        sprt.update(wins, losses);
        let (w, l) = sprt.counts();
        let share = w as f64 / sprt.games() as f64;
        println!(
            "games {:>5}  {}-{}  share {:.3} (fair {:.3})  llr {:>6.2}",
            sprt.games(),
            w,
            l,
            share,
            p0,
            sprt.llr()
        );
        if sprt.verdict() != Verdict::Open {
            break;
        }
    }
    let (w, _) = sprt.counts();
    let share = w as f64 / sprt.games().max(1) as f64;
    match sprt.verdict() {
        Verdict::AcceptH1 => println!(
            "verdict: hero '{}' beats the field of '{}' — accepted H1 (win share >= {:.3}) after \
             {} games; measured share {:.3}",
            args.a,
            args.b,
            p1,
            sprt.games(),
            share
        ),
        Verdict::RejectH1 => println!(
            "verdict: hero '{}' is not ahead of the field — accepted H0 (win share <= {:.3}) \
             after {} games; measured share {:.3}",
            args.a,
            p0,
            sprt.games(),
            share
        ),
        Verdict::Open => println!(
            "verdict: inconclusive after {} games (llr {:.2} inside [{:.2}, {:.2}]); measured \
             share {:.3} — raise max-games to decide",
            sprt.games(),
            sprt.llr(),
            sprt.bounds().0,
            sprt.bounds().1,
            share
        ),
    }
    Ok(())
}

/// Round-robin tournament over two-player configurations: every pairing plays
/// `games` seat-swapped games, then a Bradley-Terry fit (draws as half-wins,
/// lightly regularized) produces a mean-anchored Elo table.
pub fn round_robin<G: Game + Sync>(
    game: &G,
    args: &TourneyArgs,
    default_open: u64,
    parse: BotParser<G>,
) -> Result<(), String> {
    if game.num_players() != 2 {
        return Err(
            "tourney needs a 2-player configuration (e.g. set players=2 for liars-dice)".into(),
        );
    }
    if args.bots.len() < 2 {
        return Err("tourney needs at least two bots (bots=<spec1>,<spec2>,...)".into());
    }
    let open = args.opts.get("open", default_open);
    let builders: Vec<BotBuilder<G>> = args
        .bots
        .iter()
        .map(|s| parse(&parse_spec(s)?, &args.opts))
        .collect::<Result<_, _>>()?;
    let n = builders.len();
    let pairs_per = (args.games / 2).max(1);
    let mut records = vec![vec![(0u64, 0u64, 0u64); n]; n];
    println!("pairings ({} games each):", pairs_per * 2);
    for i in 0..n {
        for j in i + 1..n {
            let seed = mix(args.seed, ((i * n + j) as u64) << 32);
            let (w, d, l) = play_pairs(game, &builders[i], &builders[j], open, seed, 0..pairs_per);
            records[i][j] = (w, d, l);
            records[j][i] = (l, d, w);
            println!(
                "  {:<28} vs {:<28} {}-{}-{}",
                args.bots[i], args.bots[j], w, d, l
            );
        }
    }
    let elos = fit_elo(&records);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| elos[b].partial_cmp(&elos[a]).unwrap());
    println!("\nelo table (mean-anchored at 0):");
    for (rank, &i) in order.iter().enumerate() {
        let (w, d, l) = records[i]
            .iter()
            .fold((0, 0, 0), |acc, r| (acc.0 + r.0, acc.1 + r.1, acc.2 + r.2));
        println!(
            "  {}. {:<28} elo {:>+6.0}   {}-{}-{}",
            rank + 1,
            args.bots[i],
            elos[i],
            w,
            d,
            l
        );
    }
    Ok(())
}

/// Bradley-Terry maximum-likelihood ratings (draws as half-wins, each pairing
/// regularized with half a draw) converted to Elo and mean-anchored at 0.
fn fit_elo(records: &[Vec<(u64, u64, u64)>]) -> Vec<f64> {
    let n = records.len();
    let points = |i: usize, j: usize| {
        let (w, d, _) = records[i][j];
        w as f64 + d as f64 / 2.0 + 0.25
    };
    let mut gamma = vec![1.0f64; n];
    for _ in 0..500 {
        let prev = gamma.clone();
        for i in 0..n {
            let wins: f64 = (0..n).filter(|&j| j != i).map(|j| points(i, j)).sum();
            let denom: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let games = points(i, j) + points(j, i);
                    games / (prev[i] + prev[j])
                })
                .sum();
            gamma[i] = wins / denom;
        }
        let log_mean = gamma.iter().map(|g| g.ln()).sum::<f64>() / n as f64;
        let scale = log_mean.exp();
        for g in &mut gamma {
            *g /= scale;
        }
    }
    let elos: Vec<f64> = gamma.iter().map(|g| 400.0 * g.log10()).collect();
    let mean = elos.iter().sum::<f64>() / n as f64;
    elos.into_iter().map(|e| e - mean).collect()
}
