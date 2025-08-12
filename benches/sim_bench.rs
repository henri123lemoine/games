use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use twentyone::env::{Action, Env};

fn play_game(mut env: Env) -> u64 {
    let mut steps: u64 = 0;
    loop {
        env.start_new_round().unwrap();
        loop {
            let p = env.current_player();
            let obs = env.observation(p);
            let act = if obs.self_total < 17 {
                Action::Draw
            } else {
                Action::Stand
            };
            let res = env.step(act).unwrap();
            steps += 1;
            if res.round_over {
                break;
            }
        }
        if env.observation(0).self_hearts == 0 || env.observation(1).self_hearts == 0 {
            break;
        }
        if env.observation(0).round > 100 {
            break;
        }
    }
    steps
}

fn bench_random_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim");
    group.throughput(Throughput::Elements(10_000));
    group.bench_function("10k_games_draw17", |b| {
        b.iter_batched(
            || Env::new(0x1234_5678_9ABC_DEF0),
            |_| {
                let mut total_steps = 0u64;
                for i in 0..10_000u64 {
                    let e = Env::new(0x1234_5678_9ABC_DEF0 ^ i);
                    total_steps += play_game(e);
                }
                total_steps
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_random_policy);
criterion_main!(benches);
