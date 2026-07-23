//! Export parity: the `nn_infer` forward over the shipped `ataraxios.bin`
//! artifact must reproduce the MLX trainer's outputs on the committed fixture
//! (`export_web.py --fixture`), the same "trained net plays identically in the
//! browser" gate `ml/aztrainer/src/verify.rs` provides for the conv nets.
//!
//! The fixture stores expected outputs only; inputs are regenerated here from
//! the shared LCG. Both sides run fp32 math over identical fp16-roundtripped
//! weights, so the tolerance covers accumulation order, never quantization.

use std::path::Path;

use stratego::encode::{DEPLOY_TYPE_WIDTH, EncoderConfig, NUM_OCCUPIABLE_CELLS};
use stratego::netbot::{NetBot, scatter_grid};

const FIXTURE_MAGIC: &[u8; 8] = b"ATRXFIX1";
const LCG_SEED: u64 = 0x243F_6A88_85A3_08D3;
/// Max |Rust − MLX| over raw logits; both are fp32 chains over ~10-magnitude
/// activations, where reduction-order drift stays well under this.
const TOLERANCE: f32 = 2e-2;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u32 << 24) as f32
    }
}

struct FixtureReader<'a>(&'a [u8]);

impl FixtureReader<'_> {
    fn u32(&mut self) -> usize {
        let (head, rest) = self.0.split_at(4);
        self.0 = rest;
        u32::from_le_bytes(head.try_into().unwrap()) as usize
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        let (head, rest) = self.0.split_at(4 * n);
        self.0 = rest;
        head.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn repo_file(rel: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The shipped artifact lives in the arcade-assets bucket, not in git;
/// tools/fetch-asset.sh caches it locally and prints the path.
fn fetched_asset(logical: &str) -> Vec<u8> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fetch-asset.sh");
    let out = std::process::Command::new(&script)
        .arg(logical)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", script.display()));
    assert!(
        out.status.success(),
        "fetch-asset {logical}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let path = String::from_utf8(out.stdout).expect("utf8 path");
    std::fs::read(path.trim()).unwrap_or_else(|e| panic!("read {}: {e}", path.trim()))
}

#[test]
fn exported_artifact_matches_the_mlx_forward() {
    let bot =
        NetBot::from_bytes(&fetched_asset("artifacts/ataraxios.bin")).expect("artifact parses");
    let fixture = repo_file("tests/fixtures/net_parity.fix");
    let (magic, body) = fixture.split_at(FIXTURE_MAGIC.len());
    assert_eq!(magic, FIXTURE_MAGIC);
    let mut r = FixtureReader(body);
    let mut lcg = Lcg(LCG_SEED);

    let in_dim = EncoderConfig::default().num_token_features();
    let mut worst = 0.0f32;
    let move_cases = r.u32();
    assert!(move_cases > 0);
    for case in 0..move_cases {
        let tokens: Vec<f32> = (0..NUM_OCCUPIABLE_CELLS * in_dim)
            .map(|_| lcg.next())
            .collect();
        let expect_logits = r.floats(1800);
        let expect_value_logp = r.floats(3);

        let out = bot.net().move_forward(&tokens);
        let got_logits = scatter_grid(&out.grid);
        for (slot, (&got, &want)) in got_logits.iter().zip(&expect_logits).enumerate() {
            if want < -1e37 {
                assert!(got < -1e37, "case {case} slot {slot}: lake fill lost");
                continue;
            }
            worst = worst.max((got - want).abs());
        }
        for (cat, (&p, &want)) in out.value_probs.iter().zip(&expect_value_logp).enumerate() {
            let got = p.ln();
            assert!(
                (got - want).abs() < TOLERANCE,
                "case {case} value category {cat}: {got} vs {want}"
            );
        }
    }

    let setup_cases = r.u32();
    assert!(setup_cases > 0);
    for case in 0..setup_cases {
        let prefix_len = r.u32();
        let expect = r.floats(DEPLOY_TYPE_WIDTH);
        let supply = stratego::board::CLASSIC_INITIAL_COUNTS;
        let order: Vec<usize> = supply
            .iter()
            .enumerate()
            .flat_map(|(t, &c)| std::iter::repeat_n(t, c as usize))
            .collect();
        let mut prefix = vec![0.0f32; prefix_len * DEPLOY_TYPE_WIDTH];
        for (slot, &kind) in order[..prefix_len].iter().enumerate() {
            prefix[slot * DEPLOY_TYPE_WIDTH + kind] = 1.0;
        }
        let got = bot.net().setup_forward(&prefix);
        for (t, (&g, &want)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (g - want).abs() < TOLERANCE,
                "setup case {case} type {t}: {g} vs {want}"
            );
            worst = worst.max((g - want).abs());
        }
    }
    assert!(r.0.is_empty(), "fixture fully consumed");

    println!("max |Rust - MLX| logit difference: {worst}");
    assert!(worst < TOLERANCE, "worst logit difference {worst}");
}
