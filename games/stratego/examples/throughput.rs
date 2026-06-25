//! Self-play throughput micro-benchmark: drives the batch [`Simulator`] with the
//! uniform reference evaluator and prints env-steps/sec so we know the data-gen
//! rate on this machine. The evaluator is a no-op, so this measures the sim +
//! encoder + buffer cost — the substrate the real trainer will sit on, minus the
//! GPU forward.
//!
//! Run: `cargo run --release -p stratego --example throughput`
//! Override the envs/steps: `cargo run --release -p stratego --example throughput -- 4096 200`

use std::time::Instant;

use stratego::{EncoderConfig, ReplayBuffer, Simulator, UniformEvaluator};

fn main() {
    let mut args = std::env::args().skip(1);
    let num_envs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);

    let cfg = EncoderConfig::default();
    let eval = UniformEvaluator;
    let mut sim = Simulator::new(num_envs, cfg, 0xA7A4A05, 4000);
    let mut buffer = ReplayBuffer::new(num_envs, 256, cfg);

    let warmup = sim.run(&eval, &mut buffer, 5);
    let _ = warmup;

    let start = Instant::now();
    let stats = sim.run(&eval, &mut buffer, steps);
    let elapsed = start.elapsed().as_secs_f64();

    let env_steps = stats.decision_steps as f64;
    let env_steps_per_sec = env_steps / elapsed;

    println!("envs={num_envs} steps={steps}");
    println!("decision_steps        {}", stats.decision_steps);
    println!("games_completed       {}", stats.games_completed);
    println!("reward_pl0_sum        {:.1}", stats.reward_pl0_sum);
    println!("wall                  {elapsed:.3} s");
    println!("env-steps/sec         {env_steps_per_sec:.0}");
    println!("steps/sec (batched)   {:.1}", steps as f64 / elapsed);
}
