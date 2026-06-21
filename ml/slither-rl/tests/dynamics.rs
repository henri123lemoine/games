//! Invariants for the headless dynamics and env: determinism from a seed,
//! head-vs-body death semantics (the rammer dies, the body's owner survives),
//! food growth, boost length-burn, and observation/action/reward shapes.

use slither_rl::env::{Action, Env, SHAPES, TURN_BUCKETS};
use slither_rl::geometry::Vec2;
use slither_rl::obs::{CHANNELS, GRID, SCALARS};
use slither_rl::world::{
    START_LENGTH, WORLD, World, WorldConfig, Worm, WormControl, head_hits_body,
};
use slither_rl::{Heuristic, Rng, random_action};

fn straight(n: usize) -> Vec<WormControl> {
    vec![
        WormControl {
            aim: 0.0,
            boost: false
        };
        n
    ]
}

#[test]
fn deterministic_from_seed() {
    let cfg = WorldConfig::default();
    let run = || {
        let mut w = World::new(42, cfg);
        let mut rng = Rng::new(7);
        let n = w.worms.len();
        for _ in 0..400 {
            let controls: Vec<_> = (0..n)
                .map(|i| WormControl {
                    aim: w.worms[i].angle + rng.range(-1.0, 1.0),
                    boost: rng.unit() < 0.1,
                })
                .collect();
            w.step(&controls);
        }
        (
            w.worms.iter().map(|x| x.head()).collect::<Vec<_>>(),
            w.pellets.len(),
            w.steps,
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a.1, b.1, "pellet count diverged");
    assert_eq!(a.2, b.2, "step count diverged");
    for (x, y) in a.0.iter().zip(b.0.iter()) {
        assert_eq!(x.x.to_bits(), y.x.to_bits(), "trajectory diverged");
        assert_eq!(x.y.to_bits(), y.y.to_bits(), "trajectory diverged");
    }
}

#[test]
fn head_into_body_kills_rammer_not_body() {
    // Worm 1 (the blocker) heads south from (2000, 2000), so its body trails north
    // along the column x=2000, y in [~1470, 2000]. The rammer starts west of that
    // column at y=1800 (inside the body's span) and heads east straight into it.
    // The rammer's head owner must die; the blocker survives.
    let blocker = Worm::spawn(
        Vec2::new(2000.0, 2000.0),
        std::f32::consts::FRAC_PI_2,
        120.0,
    );
    let rammer = Worm::spawn(Vec2::new(1880.0, 1800.0), 0.0, START_LENGTH);
    let mut w = World::from_worms(1, vec![rammer, blocker], 0);

    let mut killed = false;
    for _ in 0..200 {
        // Hold the blocker still-ish (it just keeps heading south) and drive the
        // rammer due east into the body column.
        w.step(&[
            WormControl {
                aim: 0.0,
                boost: false,
            },
            WormControl {
                aim: std::f32::consts::FRAC_PI_2,
                boost: false,
            },
        ]);
        if w.worms[0].dead {
            killed = true;
            break;
        }
    }
    assert!(killed, "rammer should have hit the blocker's body");
    assert!(w.worms[0].dead, "rammer (head owner) dies");
    assert!(!w.worms[1].dead, "blocker (body owner) survives");
    assert_eq!(w.worms[0].killed_by, Some(1));
}

#[test]
fn wall_kills_on_contact() {
    // Heading west from near the left wall ends in death against it.
    let w0 = Worm::spawn(Vec2::new(60.0, 2000.0), std::f32::consts::PI, START_LENGTH);
    let mut w = World::from_worms(2, vec![w0], 0);
    let mut died = false;
    for _ in 0..200 {
        w.step(&[WormControl {
            aim: std::f32::consts::PI,
            boost: false,
        }]);
        if w.worms[0].dead {
            died = true;
            break;
        }
    }
    assert!(died, "worm should die hitting the wall");
    assert_eq!(w.worms[0].killed_by, None, "wall death has no killer worm");
}

#[test]
fn eating_food_grows_the_worm() {
    let mut w = World::new(
        5,
        WorldConfig {
            worms: 1,
            pellet_target: 2000,
            ..WorldConfig::default()
        },
    );
    let start = w.worms[0].length;
    for _ in 0..300 {
        w.step(&straight(1));
    }
    assert!(
        w.worms[0].length > start,
        "a worm crossing dense pellets should grow"
    );
}

#[test]
fn boost_burns_length() {
    let mut w = World::new(
        9,
        WorldConfig {
            worms: 1,
            pellet_target: 0,
            ..WorldConfig::default()
        },
    );
    // Make it long enough to boost, then boost without any food to absorb.
    w.worms[0].length = 200.0;
    let before = w.worms[0].length;
    for _ in 0..120 {
        w.step(&[WormControl {
            aim: w.worms[0].angle,
            boost: true,
        }]);
    }
    assert!(w.worms[0].length < before, "sustained boost burns length");
}

/// Boost must conserve mass: the length burned is shed as pellets at the same
/// rate, so a worm can't boost in a circle, eat its own shed pellets, and net
/// mass. With no ambient food, total mass (worm length + field pellet value)
/// must never rise above where it started.
#[test]
fn boost_conserves_mass() {
    let mut w = World::new(
        13,
        WorldConfig {
            worms: 1,
            pellet_target: 0,
            ..WorldConfig::default()
        },
    );
    w.worms[0].length = 300.0;
    let field: f32 = w.pellets.iter().map(|p| p.value).sum();
    let start_mass = w.worms[0].length + field;

    // Boost in a tight circle so the head keeps crossing its own shed pellets.
    for _ in 0..600 {
        w.step(&[WormControl {
            aim: w.worms[0].angle + 0.6,
            boost: true,
        }]);
        let field: f32 = w.pellets.iter().map(|p| p.value).sum();
        let mass = w.worms[0].length + field;
        assert!(
            mass <= start_mass + 1e-3,
            "boost created mass: {mass} > {start_mass}"
        );
    }
}

#[test]
fn head_hits_body_skips_neck() {
    // The first two body points are the worm's own head/neck and must never count.
    let body = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(100.0, 0.0),
    ];
    assert!(
        !head_hits_body(Vec2::new(0.0, 0.0), 5.0, &body),
        "neck must not self-trigger"
    );
    assert!(
        head_hits_body(Vec2::new(100.0, 0.0), 5.0, &body),
        "a real body point triggers"
    );
}

#[test]
fn env_shapes_and_reward_sign() {
    let mut env = Env::new(WorldConfig::default());
    let obs = env.reset(11);
    assert_eq!(SHAPES.grid, (CHANNELS, GRID, GRID));
    assert_eq!(SHAPES.scalars, SCALARS);
    assert_eq!(SHAPES.turn_buckets, TURN_BUCKETS);
    assert_eq!(obs[0].grid.len(), CHANNELS * GRID * GRID);
    assert_eq!(obs[0].scalars.len(), SCALARS);

    let n = env.num_worms();
    let out = env.step(&vec![Action::default(); n]);
    assert_eq!(out.obs.len(), n);
    assert_eq!(out.reward.len(), n);
    assert_eq!(out.done.len(), n);
    assert_eq!(out.kills.len(), n);
}

#[test]
fn delta_channels_zero_on_first_observation() {
    let mut env = Env::new(WorldConfig::default());
    let obs = env.reset(3);
    let delta_base = (CHANNELS / 2) * GRID * GRID;
    assert!(
        obs[0].grid[delta_base..].iter().all(|&v| v == 0.0),
        "the very first observation has no previous frame, so its delta is zero"
    );
}

#[test]
fn heuristic_outlives_random() {
    // Over several arenas the heuristic in seat 0 should clearly outlast a random
    // worm in the same seat — the basic competence the pool teacher needs.
    let lifespan = |use_heuristic: bool| -> u64 {
        let mut total = 0;
        for seed in 0..40u64 {
            let mut env = Env::new(WorldConfig::default());
            env.reset(seed);
            let n = env.num_worms();
            let mut heur = Heuristic::new(seed);
            let mut rng = Rng::new(seed ^ 0x99);
            for t in 0..600u64 {
                let actions: Vec<Action> = (0..n)
                    .map(|i| {
                        if i == 0 && use_heuristic {
                            heur.act(env.world(), 0)
                        } else {
                            random_action(&mut rng)
                        }
                    })
                    .collect();
                env.step(&actions);
                if env.world().worms[0].dead {
                    total += t;
                    break;
                }
                if t == 599 {
                    total += 600;
                }
            }
        }
        total
    };
    let heur = lifespan(true);
    let rand = lifespan(false);
    assert!(
        heur > rand,
        "heuristic ({heur}) should outlast random ({rand})"
    );
}

#[test]
fn body_radius_grows_sublinearly_with_length() {
    // The radius law is the single source of truth for collision and rendering,
    // and must grow gently (cube-root) with length — width should not balloon
    // with score the way a linear law makes it. Concretely: across a 10x jump in
    // length the radius must less than double, and across the whole playable
    // range (spawn → a giant) the spread stays inside ~3.5x.
    let r = |len: f32| Worm::spawn(Vec2::new(2000.0, 2000.0), 0.0, len).radius();
    let r_start = r(START_LENGTH);
    let r_mid = r(300.0);
    let r_big = r(3000.0);

    assert!(
        r_mid > r_start && r_big > r_mid,
        "radius must increase with length"
    );
    // 10x length (300 → 3000) under cube root is 10^(1/3) ≈ 2.15x on the growth
    // term, well under 2x on the whole radius once the constant base is included.
    assert!(
        r_big < 2.0 * r_mid,
        "10x length should less than double the radius, got {r_mid} → {r_big}"
    );
    assert!(
        r_big / r_start < 3.5,
        "spawn→giant spread should stay gentle, got {}x",
        r_big / r_start
    );
}

#[test]
fn worm_head_stays_in_arena() {
    let mut w = World::new(13, WorldConfig::default());
    let n = w.worms.len();
    let mut rng = Rng::new(13);
    for _ in 0..500 {
        let controls: Vec<_> = (0..n)
            .map(|i| WormControl {
                aim: w.worms[i].angle + rng.range(-2.0, 2.0),
                boost: false,
            })
            .collect();
        w.step(&controls);
        for worm in &w.worms {
            let h = worm.head();
            assert!(
                h.x >= 0.0 && h.x <= WORLD && h.y >= 0.0 && h.y <= WORLD,
                "head left the arena"
            );
        }
    }
}
