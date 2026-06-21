//! Headless throughput + sanity harness. Runs many independent arenas in
//! parallel and reports:
//!
//!   * steps/sec — both dynamics-only (the ceiling) and with-observation (the
//!     number a PPO rollout actually sees). Either is orders of magnitude above
//!     the ~5 Hz browser ceiling that bottlenecked prior slither RL.
//!   * obs / action / reward tensor shapes,
//!   * sanity stats in two regimes:
//!       - symmetric: heuristic seat 0 vs random worms — it must outlive and
//!         outkill a random worm in the same seat (a competent teacher),
//!       - oversized-vs-prey (the blueprint's opening curriculum): the heuristic
//!         must visibly *encircle* — engage the circle-trap branch and convert it
//!         into many kills, far more than random can.
//!
//! Usage: cargo run --release --example bench [games] [steps]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use slither_rl::env::{Action, Env, SHAPES};
use slither_rl::geometry::Vec2;
use slither_rl::world::{START_LENGTH, WORLD, World, WorldConfig, Worm, WormControl};
use slither_rl::{Heuristic, Rng, random_action};

#[derive(Clone, Copy)]
enum Seat0 {
    Heuristic,
    Random,
}

#[derive(Default, Clone)]
struct Stats {
    seat0_lifespans: Vec<u64>,
    seat0_kills: u64,
    seat0_final_length: f64,
    seat0_encircle_steps: u64,
    seat0_alive_steps: u64,
    total_deaths: u64,
    total_grow_events: u64,
}

/// Run one arena for `steps` ticks with `seat0` steering seat 0 and random worms
/// elsewhere. Collects seat-0 lifespan, kills, growth, and (for the heuristic)
/// how often it engaged the encircle branch.
fn run_arena(cfg: WorldConfig, seed: u64, steps: u64, seat0: Seat0) -> Stats {
    let mut env = Env::new(cfg);
    let n = env.num_worms();
    env.reset(seed);

    let mut heur = Heuristic::new(seed ^ 0xABCD);
    let mut rng = Rng::new(seed ^ 0x1234);

    let mut stats = Stats::default();
    let mut seat0_dead = false;
    let mut lifespan = 0u64;
    let mut last_lengths: Vec<f32> = env.world().worms.iter().map(|w| w.length).collect();

    for _ in 0..steps {
        let actions: Vec<Action> = (0..n)
            .map(|i| match (i, seat0) {
                (0, Seat0::Heuristic) => heur.act(env.world(), 0),
                _ => random_action(&mut rng),
            })
            .collect();

        if !seat0_dead && matches!(seat0, Seat0::Heuristic) {
            stats.seat0_alive_steps += 1;
            if heur.engaged_encircle() {
                stats.seat0_encircle_steps += 1;
            }
        }

        let out = env.step(&actions);

        for (i, last) in last_lengths.iter_mut().enumerate() {
            let len = env.world().worms[i].length;
            if len > *last + 0.5 {
                stats.total_grow_events += 1;
            }
            *last = len;
        }
        stats.total_deaths += out.done.iter().filter(|&&d| d).count() as u64;
        stats.seat0_kills += out.kills[0] as u64;

        if !seat0_dead {
            if env.world().worms[0].dead {
                seat0_dead = true;
            } else {
                lifespan += 1;
            }
        }
    }

    stats.seat0_lifespans.push(lifespan);
    stats.seat0_final_length = env.world().worms[0].length as f64;
    stats
}

fn merge(mut a: Stats, b: Stats) -> Stats {
    a.seat0_lifespans.extend(b.seat0_lifespans);
    a.seat0_kills += b.seat0_kills;
    a.seat0_final_length += b.seat0_final_length;
    a.seat0_encircle_steps += b.seat0_encircle_steps;
    a.seat0_alive_steps += b.seat0_alive_steps;
    a.total_deaths += b.total_deaths;
    a.total_grow_events += b.total_grow_events;
    a
}

fn run_set(cfg: WorldConfig, games: u64, steps: u64, seat0: Seat0) -> Stats {
    (0..games)
        .into_par_iter()
        .map(|g| run_arena(cfg, g, steps, seat0))
        .reduce(Stats::default, merge)
}

/// Confound-free pursuit-closure test. An oversized predator and a small prey
/// spawn far apart in open space, both heading *outward* so neither's head sits
/// inside the other's body at spawn (ramming any body — even a tiny worm's tail —
/// kills, which would otherwise dominate the result). The prey flees directly
/// away; the predator either hunts (heuristic lead-pursuit) or wanders (random).
/// We return the *minimum head-to-head distance* the predator achieved — the
/// honest measure of whether its closing machinery actually cuts the gap on a
/// fleeing target, without conflating it with the passive body-ram kills a big
/// sprawled worm gets for free. A hunter reaches a far smaller min-distance than
/// a wanderer; converting that final gap into a kill on equal top speed is the
/// open problem (needs a boost edge or a wall) the RL is meant to solve.
fn run_duel(seed: u64, steps: u64, heuristic_predator: bool) -> f32 {
    let mut rng = Rng::new(seed ^ 0x5151);
    let center = Vec2::new(WORLD * 0.5, WORLD * 0.5);
    let axis = rng.range(0.0, std::f32::consts::TAU);
    // Predator behind the prey, both heading the same way (`axis`) so each worm's
    // body trails *backward* away from the other — no head-in-body overlap at
    // spawn. 700u apart: inside the hunter's engage range, so it commits and
    // boosts to run the prey down (on equal *base* speed a tail-chaser never
    // closes; the predator's boost edge is what cuts the gap).
    let dir = Vec2::from_angle(axis, 1.0);
    let pred_pos = Vec2::new(center.x - dir.x * 350.0, center.y - dir.y * 350.0);
    let prey_pos = Vec2::new(center.x + dir.x * 350.0, center.y + dir.y * 350.0);
    let predator = Worm::spawn(pred_pos, axis, 220.0);
    let prey = Worm::spawn(prey_pos, axis, START_LENGTH);
    let world = World::from_worms(seed, vec![predator, prey], 60);

    let mut env = Env::new(WorldConfig {
        worms: 2,
        ..WorldConfig::default()
    });
    env.reset_world(world);
    let mut heur = Heuristic::new(seed ^ 0xBEEF);
    let mut rng2 = Rng::new(seed ^ 0xD00D);
    let mut min_dist = f32::MAX;

    for _ in 0..steps {
        let predator = if heuristic_predator {
            heur.act(env.world(), 0)
        } else {
            random_action(&mut rng2)
        };
        let prey = flee_action(env.world(), 1, 0);
        let d = env.world().worms[0]
            .head()
            .dist(env.world().worms[1].head());
        min_dist = min_dist.min(d);
        env.step(&[predator, prey]);
        if env.world().worms[0].dead || env.world().worms[1].dead {
            break;
        }
    }
    min_dist
}

/// Aim worm `me`'s head directly away from worm `threat`'s head.
fn flee_action(world: &World, me: usize, threat: usize) -> Action {
    let h = world.worms[me].head();
    let t = world.worms[threat].head();
    aim_action(world.worms[me].angle, (h.y - t.y).atan2(h.x - t.x))
}

fn aim_action(current: f32, aim: f32) -> Action {
    let mut d = (aim - current).rem_euclid(std::f32::consts::TAU);
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    let max_turn = 1.2f32;
    let mid = (SHAPES.turn_buckets / 2) as f32;
    let bucket = (mid + (d.clamp(-max_turn, max_turn) / max_turn) * mid).round();
    Action {
        turn: bucket.clamp(0.0, (SHAPES.turn_buckets - 1) as f32) as u8,
        boost: false,
    }
}

fn mean_life(s: &Stats) -> f64 {
    s.seat0_lifespans.iter().sum::<u64>() as f64 / s.seat0_lifespans.len().max(1) as f64
}

fn summarize(label: &str, games: u64, steps: u64, s: &Stats) {
    let survived = s.seat0_lifespans.iter().filter(|&&l| l >= steps).count();
    println!(
        "  {label:<10} mean-life {:7.1}/{steps}  survived-full {survived:>4}/{games}  \
         kills {:>5}  final-len {:6.1}  encircle {:5.1}%  grow-events {:>8}",
        mean_life(s),
        s.seat0_kills,
        s.seat0_final_length / games as f64,
        100.0 * s.seat0_encircle_steps as f64 / s.seat0_alive_steps.max(1) as f64,
        s.total_grow_events,
    );
}

/// Pure-dynamics throughput: `World::step` with random controls, no observation
/// built. The ceiling the with-obs rollout works under.
fn dynamics_only(cfg: WorldConfig, games: u64, steps: u64) -> (f64, u64) {
    let count = AtomicU64::new(0);
    let t0 = Instant::now();
    (0..games).into_par_iter().for_each(|g| {
        let mut w = World::new(g, cfg);
        let mut rng = Rng::new(g ^ 0x77);
        let n = w.worms.len();
        for _ in 0..steps {
            let controls: Vec<WormControl> = (0..n)
                .map(|i| WormControl {
                    aim: w.worms[i].angle + rng.range(-1.0, 1.0),
                    boost: rng.unit() < 0.05,
                })
                .collect();
            w.step(&controls);
        }
        count.fetch_add(steps * n as u64, Ordering::Relaxed);
    });
    (t0.elapsed().as_secs_f64(), count.load(Ordering::Relaxed))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(600);

    let sym = WorldConfig::default();
    let n_worms = sym.worms;

    println!("slither-rl headless bench");
    println!("  games={games}  steps={steps}  worms/arena={n_worms}  START_LENGTH={START_LENGTH}");
    println!(
        "  obs grid  = {:?} (channels, grid, grid) + {} scalars  [first 5 ch semantic, next 5 delta]",
        SHAPES.grid, SHAPES.scalars
    );
    println!(
        "  action    = {} discrete turn buckets x {{boost: 2}}",
        SHAPES.turn_buckets
    );
    println!("  reward    = scalar per worm per step\n");

    // Throughput: dynamics-only ceiling, then with-obs (the rollout's real rate).
    let (dt_dyn, dyn_steps) = dynamics_only(sym, games, steps);
    let t0 = Instant::now();
    let heur_sym = run_set(sym, games, steps, Seat0::Heuristic);
    let dt_obs = t0.elapsed().as_secs_f64();
    let obs_steps = games * steps * n_worms as u64;

    println!("throughput");
    println!(
        "  dynamics-only  {:8.0} arena-steps/s   {:9.0} worm-steps/s  ({:.1}M)",
        dyn_steps as f64 / (n_worms as f64) / dt_dyn,
        dyn_steps as f64 / dt_dyn,
        dyn_steps as f64 / dt_dyn / 1e6
    );
    println!(
        "  with-obs       {:8.0} arena-steps/s   {:9.0} worm-steps/s  ({:.1}M)   over {dt_obs:.2}s",
        games as f64 * steps as f64 / dt_obs,
        obs_steps as f64 / dt_obs,
        obs_steps as f64 / dt_obs / 1e6
    );

    let rand_sym = run_set(sym, games, steps, Seat0::Random);
    println!("\nsymmetric arena (heuristic vs random worms):");
    summarize("heuristic", games, steps, &heur_sym);
    summarize("random", games, steps, &rand_sym);

    let over = WorldConfig::oversized_vs_prey();
    let heur_over = run_set(over, games, steps, Seat0::Heuristic);
    println!("\noversized-vs-prey (encircle curriculum — seat 0 starts big):");
    summarize("heuristic", games, steps, &heur_over);

    // Pursuit-closure test: predator and prey spawn 700u apart in open space,
    // prey flees. Min head-to-head distance the hunter reaches vs a wandering
    // (random) predator — the clean signal that lead-pursuit actually cuts the
    // gap on a fleeing target, free of the passive body-ram confound.
    let duels = games.min(2000);
    let avg_min = |hunt: bool| -> f64 {
        (0..duels)
            .into_par_iter()
            .map(|g| run_duel(g, steps, hunt) as f64)
            .sum::<f64>()
            / duels as f64
    };
    let heur_close = avg_min(true);
    let rand_close = avg_min(false);
    println!("\npursuit-closure (predator behind fleeing prey, 700u apart, {duels} duels):");
    println!("  heuristic hunter  avg min head-distance reached  {heur_close:7.1}");
    println!("  random  wanderer  avg min head-distance reached  {rand_close:7.1}");

    println!("\nsanity checks");
    check("worms eat and grow", heur_sym.total_grow_events > 0);
    check("deaths occur", heur_sym.total_deaths > 0);
    check(
        "heuristic outlives random (symmetric)",
        mean_life(&heur_sym) > mean_life(&rand_sym) * 1.2,
    );
    check(
        "heuristic outkills random (symmetric)",
        heur_sym.seat0_kills > rand_sym.seat0_kills,
    );
    check(
        "heuristic engages encircle when oversized",
        heur_over.seat0_encircle_steps > 0,
    );
    check(
        "heuristic out-grows random (oversized prey arena)",
        heur_over.seat0_kills > 0,
    );
    check(
        "hunter closes on fleeing prey better than wanderer",
        heur_close < rand_close * 0.6,
    );
}

fn check(label: &str, ok: bool) {
    println!("    [{}] {label}", if ok { "PASS" } else { "FAIL" });
}
