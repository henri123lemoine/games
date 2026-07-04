//! Monitored long-run ReBeL deploy training — the overnight 5p5d6f run with the
//! metrics a human watcher actually needs. Trains the single config-invariant PBS
//! value net by bootstrapped self-play on the real multi-round game (the mixed
//! config sampler, biased to the 5p5d6f flagship), and streams:
//!
//!   * `OUTDIR/metrics.jsonl` — one JSON line per event (`step`, `wall_s`,
//!     `samples`, `train_steps`). TRAIN events (every `LOG_EVERY` steps) carry the
//!     windowed mean `loss` and `throughput_sps`. EVAL events (every `EVAL_EVERY`
//!     steps) additionally carry `expl_2p2d3f` — the net's worst per-round
//!     exploitability vs the EXACT 2p2d3f lattice (fit once at startup) — and,
//!     every 4th eval, `winshare`, a `RebelAgent`(this net) field win share vs
//!     the Rollout baseline on the configured target (default 5p5d6f). An eval
//!     line is a superset of a train line, so `step` is strictly increasing
//!     across all lines. Eval rows also include checkpoint paths so tournament
//!     sweeps can rank the actual training curve rather than only the latest
//!     artifact.
//!   * `OUTDIR/best.bin` — saved whenever measured target `winshare` improves
//!     (falling back to `expl_2p2d3f` only before any win-share eval exists).
//!   * `OUTDIR/ckpt_stepN.bin` — eval-time snapshots when `KEEP_CHECKPOINTS=1`.
//!   * `OUTDIR/ckpt.bin` — the latest net, every `CKPT_EVERY` steps (and at exit).
//!
//! Resume: `RESUME=<path>` loads that PbsNet and continues (skipping warmup);
//! without it the run starts fresh with the normal heuristic->net warmup.
//!
//!     cargo run --release -p liars-dice --features parallel --example monitored_train
//!
//! Env: STEPS HIDDEN NUM_ITERS DEPTH GEN_PER OUTDIR SEED LOG_EVERY EVAL_EVERY RESUME
//! (plus WARMUP TRAIN_RATIO BURN_IN BUFFER LR LR_HALFLIFE EVAL_ITERS FIT_ITERS WINSHARE_PLAYERS
//! WINSHARE_DICE WINSHARE_FACES WINSHARE_GAMES WINSHARE_ROLLOUTS WINSHARE_ITERS
//! KEEP_CHECKPOINTS CKPT_EVERY). Tail the run with `tail -f OUTDIR/metrics.jsonl`.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use game_core::winrate_vs_field;
use liars_dice::rebel::{DeployTrainConfig, DeployTrainer, ExploitProbe, PbsNet, RebelAgent};
use liars_dice::{BidConditioned, LiarsDice, ProbabilisticAgent};
use solvers::Rollout;

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|e| panic!("failed to parse {key}='{value}' as usize: {e}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => panic!("failed to read {key}: {e}"),
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    match std::env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|e| panic!("failed to parse {key}='{value}' as f64: {e}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => panic!("failed to read {key}: {e}"),
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => panic!(
                "failed to parse {key}='{value}' as boolean (use 1/0, true/false, yes/no, on/off)"
            ),
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => panic!("failed to read {key}: {e}"),
    }
}

/// Format a float as a JSON number, or `null` when non-finite (keeps the line
/// valid JSON even if a metric blows up).
fn jf(x: f64, prec: usize) -> String {
    if x.is_finite() {
        format!("{x:.prec$}")
    } else {
        "null".to_string()
    }
}

/// Field win share of a `RebelAgent`(this net) vs the Rollout baseline.
/// `PbsNet` is not `Clone`, so the net is round-tripped through a temp file to
/// hand the agent its own copy.
struct WinshareEval {
    players: u8,
    dice: u8,
    faces: u8,
    games: u32,
    rollouts: u32,
    iters: usize,
    seed: u64,
}

fn winshare_vs_rollout(net: &PbsNet, outdir: &Path, cfg: &WinshareEval) -> f64 {
    let tmp = outdir.join("_winshare_net.bin");
    if net.save(&tmp).is_err() {
        return f64::NAN;
    }
    let loaded = match PbsNet::load(&tmp) {
        Ok(n) => n,
        Err(_) => return f64::NAN,
    };
    let agent = RebelAgent::with_config(loaded, cfg.iters, 2);
    let game = LiarsDice::new(cfg.players, cfg.dice, cfg.faces);
    let baseline = Rollout::new(
        cfg.rollouts,
        ProbabilisticAgent::default_agent(),
        BidConditioned::default(),
    );
    winrate_vs_field(
        &game,
        &agent,
        &baseline,
        cfg.games,
        cfg.seed ^ 0x3D34_C0FF_EE15_600D,
    )
}

fn winshare_metric_key(players: u8, dice: u8, faces: u8) -> String {
    format!("winshare_{players}p{dice}d{faces}f")
}

fn main() {
    let threads = rayon::current_num_threads();
    let steps = env_usize("STEPS", 100_000);
    let hidden = env_usize("HIDDEN", 256);
    let num_iters = env_usize("NUM_ITERS", 256);
    let gen_per = env_usize("GEN_PER", threads);
    let depth = env_usize("DEPTH", 2) as u32;
    let warmup = env_usize("WARMUP", 200);
    let log_every = env_usize("LOG_EVERY", 25).max(1);
    let eval_every = env_usize("EVAL_EVERY", 250).max(1);
    let ckpt_every = env_usize("CKPT_EVERY", 100).max(1);
    let keep_checkpoints = env_bool("KEEP_CHECKPOINTS", true);
    let train_ratio = env_usize("TRAIN_RATIO", 8);
    let burn_in = env_usize("BURN_IN", 2048);
    let buffer = env_usize("BUFFER", 2_000_000);
    let lr = env_f64("LR", 3e-4) as f32;
    let lr_halflife = env_usize(
        "LR_HALFLIFE",
        DeployTrainConfig::default().lr_halflife as usize,
    ) as u64;
    let seed = env_usize("SEED", 0) as u64;

    // Probe / eval budgets — kept cheap so monitoring never dominates training.
    let eval_iters = env_usize("EVAL_ITERS", 256);
    let fit_iters = env_usize("FIT_ITERS", 1500) as u64;
    let ws_players = env_usize("WINSHARE_PLAYERS", 5) as u8;
    let ws_dice = env_usize("WINSHARE_DICE", 5) as u8;
    let ws_faces = env_usize("WINSHARE_FACES", 6) as u8;
    let ws_games = env_usize("WINSHARE_GAMES", 80) as u32;
    let ws_rollouts = env_usize("WINSHARE_ROLLOUTS", 80) as u32;
    let ws_iters = env_usize("WINSHARE_ITERS", 96);
    let ws_metric = winshare_metric_key(ws_players, ws_dice, ws_faces);
    let ws_eval = WinshareEval {
        players: ws_players,
        dice: ws_dice,
        faces: ws_faces,
        games: ws_games,
        rollouts: ws_rollouts,
        iters: ws_iters,
        seed,
    };

    let outdir =
        PathBuf::from(std::env::var("OUTDIR").unwrap_or_else(|_| "runs/ld_rebel_monitored".into()));
    std::fs::create_dir_all(&outdir).unwrap();
    let resume = std::env::var("RESUME").ok().filter(|s| !s.is_empty());

    let cfg = DeployTrainConfig {
        steps,
        warmup_steps: warmup,
        num_iters,
        max_depth: depth,
        batch: 512,
        lr,
        lr_halflife,
        gen_per_step: gen_per,
        train_gen_ratio: train_ratio,
        burn_in,
        eval_every,
        eval_iters,
        eval_fit_iters: fit_iters,
        hidden,
        n_layers: 2,
        buffer_cap: buffer,
        seed,
        log: false,
        outdir: outdir.clone(),
        fixed_config: None,
        ..DeployTrainConfig::default()
    };

    let metrics_path = outdir.join("metrics.jsonl");
    let mut metrics = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .unwrap();
    writeln!(
        metrics,
        "{{\"event\":\"rebel_monitored_config\",\"steps\":{steps},\"hidden\":{hidden},\
         \"num_iters\":{num_iters},\"depth\":{depth},\"gen_per\":{gen_per},\
         \"warmup\":{warmup},\"log_every\":{log_every},\"eval_every\":{eval_every},\
         \"ckpt_every\":{ckpt_every},\"keep_checkpoints\":{keep_checkpoints},\
         \"train_ratio\":{train_ratio},\"burn_in\":{burn_in},\"buffer\":{buffer},\
         \"lr\":{lr},\"lr_halflife\":{lr_halflife},\
         \"seed\":{seed},\"eval_iters\":{eval_iters},\"fit_iters\":{fit_iters},\
         \"winshare_players\":{ws_players},\"winshare_dice\":{ws_dice},\
         \"winshare_faces\":{ws_faces},\
         \"winshare_games\":{ws_games},\"winshare_rollouts\":{ws_rollouts},\
         \"winshare_iters\":{ws_iters},\"outdir\":\"{}\",\"resume\":\"{}\"}}",
        json_escape(&outdir.display().to_string()),
        json_escape(resume.as_deref().unwrap_or(""))
    )
    .unwrap();

    let mut trainer = DeployTrainer::new(cfg);
    let resuming = if let Some(ref p) = resume {
        trainer
            .load_net(Path::new(p))
            .unwrap_or_else(|e| panic!("RESUME load {p}: {e}"));
        true
    } else {
        false
    };

    // The exact 2p2d3f lattice — fit ONCE, reused every eval.
    println!("=== fitting 2p2d3f exploitability probe (fit_iters={fit_iters}) ===");
    let probe = ExploitProbe::fit(2, 3, fit_iters, eval_iters, true);

    println!(
        "=== monitored ReBeL deploy training (mixed sampler, 5p5d6f-biased) ===\n\
         hidden={hidden} num_iters={num_iters} depth={depth} gen_per={gen_per} train_ratio={train_ratio} \
         buffer={buffer} lr={lr} lr_halflife={lr_halflife} threads={threads}\n\
         steps={steps} warmup={warmup} log_every={log_every} eval_every={eval_every} \
         winshare_target={ws_players}p{ws_dice}d{ws_faces}f \
         ckpt_every={ckpt_every} keep_checkpoints={keep_checkpoints} resume={} outdir={}\n\
         metrics={}",
        resume.as_deref().unwrap_or("(none)"),
        outdir.display(),
        metrics_path.display(),
    );

    let t_start = Instant::now();
    let mut win_start = Instant::now();
    let mut win_samples = trainer.samples_generated();
    let mut win_loss_sum = 0.0f64;
    let mut win_loss_count = 0u64;
    let mut eval_count = 0u64;
    let mut best_expl = f64::INFINITY;
    let mut best_winshare = f64::NEG_INFINITY;

    for step in 0..steps {
        let use_net_cont = resuming || step >= warmup;
        let stat = trainer.step(use_net_cont);
        win_loss_sum += stat.loss_sum;
        win_loss_count += stat.loss_count;

        let s = step + 1;
        let is_eval = s.is_multiple_of(eval_every) || s == steps;
        let is_log = is_eval || s.is_multiple_of(log_every);

        if is_log {
            let wall = t_start.elapsed().as_secs_f64();
            let samples = trainer.samples_generated();
            let train_steps = trainer.train_steps();
            let win_wall = win_start.elapsed().as_secs_f64();
            let throughput = if win_wall > 0.0 {
                (samples - win_samples) as f64 / win_wall
            } else {
                0.0
            };
            let mean_loss = (win_loss_count > 0).then(|| win_loss_sum / win_loss_count as f64);

            let mut fields = vec![
                format!("\"step\":{s}"),
                format!("\"event\":\"{}\"", if is_eval { "eval" } else { "train" }),
                format!("\"wall_s\":{}", jf(wall, 3)),
                format!("\"samples\":{samples}"),
                format!("\"train_steps\":{train_steps}"),
                format!(
                    "\"loss\":{}",
                    mean_loss.map(|l| jf(l, 6)).unwrap_or_else(|| "null".into())
                ),
                format!("\"throughput_sps\":{}", jf(throughput, 1)),
            ];

            let mut expl = None;
            let mut winshare = None;
            if is_eval {
                eval_count += 1;
                let e = probe.max_exploitability(trainer.net());
                expl = Some(e);
                fields.push(format!("\"expl_2p2d3f\":{}", jf(e, 6)));

                if (eval_count - 1).is_multiple_of(4) {
                    let w = winshare_vs_rollout(trainer.net(), &outdir, &ws_eval);
                    winshare = Some(w);
                    fields.push(format!("\"winshare_config\":\"{}\"", ws_metric));
                    fields.push(format!("\"winshare\":{}", jf(w, 4)));
                    fields.push(format!("\"{}\":{}", ws_metric, jf(w, 4)));
                }

                let diagnostic_best = e < best_expl;
                if diagnostic_best {
                    best_expl = e;
                }
                let is_best = if let Some(w) = winshare {
                    if w > best_winshare {
                        best_winshare = w;
                        true
                    } else {
                        false
                    }
                } else {
                    best_winshare == f64::NEG_INFINITY && diagnostic_best
                };
                if is_best {
                    let _ = trainer.net().save(&outdir.join("best.bin"));
                }
                let latest_path = outdir.join("ckpt.bin");
                let _ = trainer.net().save(&latest_path);
                fields.push(format!("\"is_best\":{is_best}"));
                let selection_metric = if is_best && winshare.is_some() {
                    ws_metric.as_str()
                } else if is_best {
                    "expl_2p2d3f"
                } else {
                    "none"
                };
                fields.push(format!(
                    "\"selection_metric\":\"{}\"",
                    json_escape(selection_metric)
                ));
                let best_winshare_json = if best_winshare.is_finite() {
                    jf(best_winshare, 4)
                } else {
                    "null".to_string()
                };
                fields.push(format!("\"best_winshare\":{}", best_winshare_json));
                fields.push(format!("\"best_{}\":{}", ws_metric, best_winshare_json));
                fields.push(format!("\"best_expl_2p2d3f\":{}", jf(best_expl, 6)));
                fields.push(format!(
                    "\"latest_checkpoint\":\"{}\"",
                    json_escape(&latest_path.display().to_string())
                ));
                fields.push(format!(
                    "\"best_checkpoint\":\"{}\"",
                    json_escape(&outdir.join("best.bin").display().to_string())
                ));
                if keep_checkpoints {
                    let eval_path = outdir.join(format!("ckpt_step{s}.bin"));
                    let _ = trainer.net().save(&eval_path);
                    fields.push(format!(
                        "\"checkpoint\":\"{}\"",
                        json_escape(&eval_path.display().to_string())
                    ));
                }
            }

            writeln!(metrics, "{{{}}}", fields.join(",")).unwrap();
            metrics.flush().unwrap();

            win_start = Instant::now();
            win_samples = samples;
            win_loss_sum = 0.0;
            win_loss_count = 0;

            if let Some(e) = expl {
                let loss_str = mean_loss
                    .map(|l| format!("{l:.5}"))
                    .unwrap_or_else(|| "   na  ".into());
                let ws_str = winshare
                    .map(|w| format!("{w:.4}"))
                    .unwrap_or_else(|| "  -   ".into());
                println!(
                    "[mon] step={s:>6} samples={samples:>9} loss={loss_str} \
                     expl={e:.5} best_expl={best_expl:.5} winshare={ws_str} \
                     best_ws={} thr={throughput:.0}sps",
                    if best_winshare.is_finite() {
                        format!("{best_winshare:.4}")
                    } else {
                        "-".to_string()
                    }
                );
            }
        }

        if s.is_multiple_of(ckpt_every) {
            let _ = trainer.net().save(&outdir.join("ckpt.bin"));
        }
    }

    let _ = trainer.net().save(&outdir.join("ckpt.bin"));
    let best_ws = if best_winshare.is_finite() {
        format!("{best_winshare:.4}")
    } else {
        "n/a".to_string()
    };
    println!(
        "=== done: samples={} train_steps={} best_winshare={best_ws} \
         best_expl_2p2d3f={best_expl:.5} \
         — latest net at {}/ckpt.bin, best at {}/best.bin ===",
        trainer.samples_generated(),
        trainer.train_steps(),
        outdir.display(),
        outdir.display(),
    );
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winshare_metric_key_names_target_config() {
        assert_eq!(winshare_metric_key(5, 5, 6), "winshare_5p5d6f");
        assert_eq!(winshare_metric_key(2, 1, 2), "winshare_2p1d2f");
    }
}
