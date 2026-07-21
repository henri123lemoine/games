//! Statistical comparison for simultaneous-action games.
//!
//! This mirrors the ordinary arena's paired GSPRT, field SPRT, and tournament
//! surfaces, but every turn is collected through [`SimultaneousGame`].

use game_core::stats::{BinomialSprt, Sprt, Verdict, elo_estimate, fit_elo};
use game_core::{Rng, SimultaneousAgent, SimultaneousGame, SimultaneousTurn, hash};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::compare::{BotSpec, CompareArgs, TourneyArgs, parse_spec};
use crate::registry::Opts;

/// Arena adjudication only. Canonical rules have no artificial turn cap, but a
/// statistically bounded comparison must not hang on two repetition-prone
/// deterministic agents; an unresolved game at this horizon is a draw.
const TURN_CAP: u64 = 1_000;

pub type BoxedSimultaneousAgent<G> = Box<dyn SimultaneousAgent<G>>;
pub type SimultaneousBotBuilder<G> = Box<dyn Fn(u64) -> BoxedSimultaneousAgent<G> + Send + Sync>;
pub type SimultaneousBotParser<G> =
    fn(&BotSpec, &Opts) -> Result<SimultaneousBotBuilder<G>, String>;

fn parse_bot<G: SimultaneousGame>(
    text: &str,
    parse: SimultaneousBotParser<G>,
    opts: &Opts,
) -> Result<SimultaneousBotBuilder<G>, String> {
    let spec = parse_spec(text)?;
    let builder = parse(&spec, opts)?;
    spec.opts.ensure_consumed(&format!("bot '{text}'"))?;
    Ok(builder)
}

fn mix(seed: u64, index: u64) -> u64 {
    hash::combine(seed, index) | 1
}

fn play_scored<G: SimultaneousGame>(
    game: &G,
    agents: &[&dyn SimultaneousAgent<G>],
    open_turns: u64,
    seed: u64,
    perspective: usize,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut state = game.initial_state_with_rng(&mut rng);
    let mut turns = 0;
    while !game.is_terminal(&state) && turns < TURN_CAP {
        match game.turn(&state) {
            SimultaneousTurn::Chance => {
                let action = game.sample_chance_action(&state, &mut rng);
                game.apply_chance(&mut state, action);
            }
            SimultaneousTurn::Players => {
                let actions: Vec<_> = (0..game.num_players())
                    .map(|player| {
                        let index = if !game.is_active(&state, player) || turns < open_turns {
                            rng.below(game.num_actions(&state, player))
                        } else {
                            agents[player].act(game, &state, player, &mut rng)
                        };
                        game.action_at(&state, player, index)
                    })
                    .collect();
                game.apply_joint(&mut state, &actions);
                turns += 1;
            }
        }
    }
    if !game.is_terminal(&state) {
        return 0.0;
    }
    game.returns(&state, perspective)
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

pub fn play_one_pair<G: SimultaneousGame>(
    game: &G,
    a: &SimultaneousBotBuilder<G>,
    b: &SimultaneousBotBuilder<G>,
    open_turns: u64,
    seed: u64,
) -> (u64, u64, u64) {
    let a0 = a(seed ^ 0xA11CE);
    let b1 = b(seed ^ 0xB0B);
    let first: [&dyn SimultaneousAgent<G>; 2] = [&*a0, &*b1];
    let u0 = play_scored(game, &first, open_turns, seed, 0);

    let b0 = b(seed ^ 0xB0B);
    let a1 = a(seed ^ 0xA11CE);
    let second: [&dyn SimultaneousAgent<G>; 2] = [&*b0, &*a1];
    let u1 = play_scored(game, &second, open_turns, seed, 1);
    let (w0, d0, l0) = wdl(u0);
    let (w1, d1, l1) = wdl(u1);
    (w0 + w1, d0 + d1, l0 + l1)
}

fn play_pairs<G: SimultaneousGame + Sync>(
    game: &G,
    a: &SimultaneousBotBuilder<G>,
    b: &SimultaneousBotBuilder<G>,
    open_turns: u64,
    seed: u64,
    pairs: std::ops::Range<u64>,
) -> (u64, u64, u64) {
    let one = |index| play_one_pair(game, a, b, open_turns, mix(seed, index));
    let sum = |a: (u64, u64, u64), b: (u64, u64, u64)| (a.0 + b.0, a.1 + b.1, a.2 + b.2);
    #[cfg(feature = "parallel")]
    return pairs.into_par_iter().map(one).reduce(|| (0, 0, 0), sum);
    #[cfg(not(feature = "parallel"))]
    pairs.map(one).fold((0, 0, 0), sum)
}

fn play_one_field_game<G: SimultaneousGame>(
    game: &G,
    hero_builder: &SimultaneousBotBuilder<G>,
    field_builder: &SimultaneousBotBuilder<G>,
    game_index: u64,
    seed: u64,
) -> bool {
    let game_seed = mix(seed, game_index);
    let players = game.num_players();
    let hero_seat = game_index as usize % players;
    let hero = hero_builder(game_seed ^ 0xA11CE);
    let field: Vec<BoxedSimultaneousAgent<G>> = (0..players - 1)
        .map(|index| field_builder(game_seed ^ 0xB0B ^ (index as u64) << 17))
        .collect();
    let mut next_field = 0;
    let agents: Vec<&dyn SimultaneousAgent<G>> = (0..players)
        .map(|seat| {
            if seat == hero_seat {
                &*hero
            } else {
                let agent = &*field[next_field];
                next_field += 1;
                agent
            }
        })
        .collect();
    play_scored(game, &agents, 0, game_seed, hero_seat) > 0.0
}

#[allow(clippy::too_many_arguments)]
pub fn run_pairs<G: SimultaneousGame + Sync>(
    game: &G,
    opts: &Opts,
    a: &str,
    b: &str,
    default_open: u64,
    parse: SimultaneousBotParser<G>,
    seed: u64,
    pairs: std::ops::Range<u64>,
) -> Result<(u64, u64, u64), String> {
    let a = parse_bot(a, parse, opts)?;
    let b = parse_bot(b, parse, opts)?;
    let open = opts.get("open", default_open)?;
    opts.ensure_consumed("simultaneous pairs")?;
    Ok(play_pairs(game, &a, &b, open, seed, pairs))
}

pub fn run_field<G: SimultaneousGame + Sync>(
    game: &G,
    opts: &Opts,
    a: &str,
    b: &str,
    parse: SimultaneousBotParser<G>,
    seed: u64,
    games: std::ops::Range<u64>,
) -> Result<(u64, u64), String> {
    let a = parse_bot(a, parse, opts)?;
    let b = parse_bot(b, parse, opts)?;
    opts.ensure_consumed("simultaneous field games")?;
    let mut wins = 0;
    let mut losses = 0;
    for index in games {
        if play_one_field_game(game, &a, &b, index, seed) {
            wins += 1;
        } else {
            losses += 1;
        }
    }
    Ok((wins, losses))
}

pub fn head_to_head<G: SimultaneousGame + Sync>(
    game: &G,
    args: &CompareArgs,
    default_open: u64,
    parse: SimultaneousBotParser<G>,
) -> Result<(), String> {
    let a = parse_bot(&args.a, parse, &args.opts)?;
    let b = parse_bot(&args.b, parse, &args.opts)?;
    let open = args.opts.get("open", default_open)?;
    args.opts.ensure_consumed("simultaneous compare")?;
    let mut sprt = Sprt::new(args.elo0, args.elo1, args.alpha, args.beta);
    let max_pairs = (args.max_games / 2).max(1);
    let batch_pairs = (args.batch / 2).max(1);
    println!(
        "compare: '{}' vs '{}'  H0 elo={}  H1 elo={}  open={} joint turns  seed={}",
        args.a, args.b, args.elo0, args.elo1, open, args.seed
    );
    let mut next = 0;
    while next < max_pairs {
        let end = (next + batch_pairs).min(max_pairs);
        let (wins, draws, losses) = play_pairs(game, &a, &b, open, args.seed, next..end);
        next = end;
        sprt.update(wins, draws, losses);
        let (wins, draws, losses) = sprt.counts();
        let estimate = elo_estimate(wins, draws, losses);
        println!(
            "games {:>5}  {}-{}-{}  elo {:>+7.1} +/- {:>5.1}  llr {:>6.2}",
            sprt.games(),
            wins,
            draws,
            losses,
            estimate.elo,
            estimate.margin(),
            sprt.llr()
        );
        if sprt.verdict() != Verdict::Open {
            break;
        }
    }
    let (wins, draws, losses) = sprt.counts();
    let estimate = elo_estimate(wins, draws, losses);
    println!(
        "verdict: {:?} after {} games; measured elo {:+.0} +/- {:.0}",
        sprt.verdict(),
        sprt.games(),
        estimate.elo,
        estimate.margin()
    );
    Ok(())
}

pub fn vs_field<G: SimultaneousGame + Sync>(
    game: &G,
    args: &CompareArgs,
    parse: SimultaneousBotParser<G>,
) -> Result<(), String> {
    let a = parse_bot(&args.a, parse, &args.opts)?;
    let b = parse_bot(&args.b, parse, &args.opts)?;
    args.opts.ensure_consumed("simultaneous field compare")?;
    let fair = 1.0 / game.num_players() as f64;
    let target = (fair + args.delta).min(1.0 - 1e-6);
    let mut sprt = BinomialSprt::new(fair, target, args.alpha, args.beta);
    let mut next = 0;
    while next < args.max_games {
        let end = (next + args.batch.max(1)).min(args.max_games);
        let one = |index| {
            if play_one_field_game(game, &a, &b, index, args.seed) {
                (1u64, 0u64)
            } else {
                (0, 1)
            }
        };
        let sum = |a: (u64, u64), b: (u64, u64)| (a.0 + b.0, a.1 + b.1);
        #[cfg(feature = "parallel")]
        let (wins, losses) = (next..end).into_par_iter().map(one).reduce(|| (0, 0), sum);
        #[cfg(not(feature = "parallel"))]
        let (wins, losses) = (next..end).map(one).fold((0, 0), sum);
        next = end;
        sprt.update(wins, losses);
        let (wins, losses) = sprt.counts();
        println!(
            "games {:>5}  {}-{}  share {:.3} (fair {:.3})  llr {:>6.2}",
            sprt.games(),
            wins,
            losses,
            wins as f64 / sprt.games() as f64,
            fair,
            sprt.llr()
        );
        if sprt.verdict() != Verdict::Open {
            break;
        }
    }
    println!(
        "verdict: {:?} after {} games (target share {:.3})",
        sprt.verdict(),
        sprt.games(),
        target
    );
    Ok(())
}

pub fn round_robin<G: SimultaneousGame + Sync>(
    game: &G,
    args: &TourneyArgs,
    default_open: u64,
    parse: SimultaneousBotParser<G>,
) -> Result<(), String> {
    if game.num_players() != 2 {
        return Err("tourney requires players=2".into());
    }
    if args.bots.len() < 2 {
        return Err("tourney needs at least two bots".into());
    }
    let open = args.opts.get("open", default_open)?;
    let builders: Vec<_> = args
        .bots
        .iter()
        .map(|bot| parse_bot(bot, parse, &args.opts))
        .collect::<Result<_, _>>()?;
    args.opts.ensure_consumed("simultaneous tourney")?;
    let count = builders.len();
    let pairs_per = (args.games / 2).max(1);
    let mut records = vec![vec![(0, 0, 0); count]; count];
    for first in 0..count {
        for second in first + 1..count {
            let seed = mix(args.seed, ((first * count + second) as u64) << 32);
            let (wins, draws, losses) = play_pairs(
                game,
                &builders[first],
                &builders[second],
                open,
                seed,
                0..pairs_per,
            );
            records[first][second] = (wins, draws, losses);
            records[second][first] = (losses, draws, wins);
            println!(
                "  {:<28} vs {:<28} {}-{}-{}",
                args.bots[first], args.bots[second], wins, draws, losses
            );
        }
    }
    let ratings = fit_elo(&records);
    let mut order: Vec<_> = (0..count).collect();
    order.sort_by(|&a, &b| ratings[b].total_cmp(&ratings[a]));
    println!("\nelo table (mean-anchored at 0):");
    for (rank, &index) in order.iter().enumerate() {
        println!(
            "  {}. {:<28} elo {:>+6.0}",
            rank + 1,
            args.bots[index],
            ratings[index]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use game_core::{SimultaneousGame, SimultaneousTurn};

    use super::*;

    #[derive(Clone)]
    struct State(Option<[u8; 2]>);
    struct Matching;

    impl SimultaneousGame for Matching {
        type State = State;
        type Action = u8;
        type ChanceAction = ();
        fn num_players(&self) -> usize {
            2
        }
        fn initial_state(&self) -> State {
            State(None)
        }
        fn turn(&self, _state: &State) -> SimultaneousTurn {
            SimultaneousTurn::Players
        }
        fn is_terminal(&self, state: &State) -> bool {
            state.0.is_some()
        }
        fn is_active(&self, _state: &State, _player: usize) -> bool {
            true
        }
        fn returns(&self, state: &State, player: usize) -> f64 {
            let actions = state.0.expect("terminal");
            let first = if actions[0] == actions[1] { 1.0 } else { -1.0 };
            if player == 0 { first } else { -first }
        }
        fn legal_actions(&self, _state: &State, _player: usize) -> Vec<u8> {
            vec![0, 1]
        }
        fn apply_joint(&self, state: &mut State, actions: &[u8]) {
            state.0 = Some([actions[0], actions[1]]);
        }
        fn chance_outcomes(&self, _state: &State) -> Vec<((), f64)> {
            Vec::new()
        }
        fn apply_chance(&self, _state: &mut State, _action: ()) {
            unreachable!()
        }
    }

    #[test]
    fn paired_games_swap_seats() {
        let zero: SimultaneousBotBuilder<Matching> =
            Box::new(|_| Box::new(|_: &Matching, _: &State, _: usize, _: &mut Rng| 0));
        let one: SimultaneousBotBuilder<Matching> =
            Box::new(|_| Box::new(|_: &Matching, _: &State, _: usize, _: &mut Rng| 1));
        assert_eq!(play_one_pair(&Matching, &zero, &one, 0, 7), (1, 0, 1));
    }
}
