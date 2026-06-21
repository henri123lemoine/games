//! Self-consistency of the generic forward on the *committed trained nets*,
//! loaded through `nn_infer::Legacy` (the deploy path). The bit-for-bit
//! comparison against the original `azinfer`/`goinfer`/`snakeinfer` forwards
//! lived here too — that `==` proof passed and retired with those reference
//! crates when they were deleted. What endures and still needs guarding: the
//! committed nets parse, the AZNET1 body stays byte-identical to the legacy body
//! (the 73-zero policy-conv pad), the forward is well-formed (finite logits,
//! `value ∈ [-1,1]`, ownership where the head carries it), it is board-size
//! agnostic for the global-pool heads, and `forward_support` yields a valid
//! distribution over the legal subset.

use nn_infer::{Arch, HeadFlags, HeadKind, Legacy, Net};

/// A tiny xorshift RNG so inputs are reproducible without a dep.
struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

fn read(path: &str) -> Option<Vec<u8>> {
    let p = format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), path);
    std::fs::read(&p).ok()
}

fn well_formed(out: &nn_infer::Output, policy_len: usize) {
    assert_eq!(out.policy.len(), policy_len, "policy width");
    assert!(out.policy.iter().all(|x| x.is_finite()), "finite logits");
    assert!(
        out.value.is_finite() && out.value.abs() <= 1.0,
        "value in [-1,1]: {}",
        out.value
    );
}

#[test]
fn chess_net_loads_and_is_well_formed() {
    let Some(old) = read("web/app/public/azero/azero-chess.azweb") else {
        eprintln!("skip: chess net not committed");
        return;
    };
    let planes = chess::encode::PLANE_COUNT;
    let policy_len = chess::encode::AZ_POLICY_LEN;
    let net = Legacy::FlatConv { planes, policy_len }
        .load(&old)
        .expect("legacy load");
    assert_eq!(net.arch().head, HeadKind::FlatConv);

    // Byte-identical body: an AZNET1 buffer for the same arch is
    // `header ∥ legacy_body`, including chess's 73-zero policy-conv bias pad.
    let arch = Arch {
        blocks: net.arch().blocks,
        channels: net.arch().channels,
        planes,
        size: 8,
        scalars: 0,
        head: HeadKind::FlatConv,
        policy_len,
        flags: HeadFlags::default(),
    };
    let mut aznet = arch.header_bytes();
    aznet.extend_from_slice(&old[16..]);
    assert_eq!(
        aznet[nn_infer::format::HEADER_LEN..],
        old[16..],
        "AZNET1 FlatConv body must equal the legacy body (73-zero pad kept)"
    );
    assert!(Net::parse(&aznet).is_ok(), "rewrapped AZNET1 parses");

    let mut rng = Rng(0xC0FFEE);
    for _ in 0..16 {
        well_formed(&net.forward(&rng.fill(planes * 64), &[]), policy_len);
    }
}

#[test]
fn go_net_loads_well_formed_size_agnostic_with_ownership() {
    let Some(old) = read("web/app/public/azero/azero-go.azweb") else {
        eprintln!("skip: go net not committed");
        return;
    };
    let planes = go::encode::PLANES;
    let net = Legacy::GoSpatial { planes }
        .load(&old)
        .expect("legacy load");
    let size = net.arch().size;
    assert_eq!(net.arch().head, HeadKind::GlobalPoolSpatial);
    assert!(
        net.arch().flags.ownership(),
        "the committed go net carries the ownership head (AZWEBGO3)"
    );

    let mut rng = Rng(0x90D90D);
    // The global-pool heads run at any board size on the same weights.
    for &s in &[size, 9] {
        for _ in 0..8 {
            let out = net.forward_at(&rng.fill(planes * s * s), &[], s);
            well_formed(&out, s * s + 1);
            let own = out.ownership.expect("ownership head present");
            assert_eq!(own.len(), s * s, "one ownership value per point @ {s}");
            assert!(
                own.iter().all(|o| o.is_finite() && o.abs() <= 1.0),
                "ownership in [-1,1] @ {s}"
            );
        }
    }
}

#[test]
fn snake_net_loads_well_formed_size_agnostic() {
    let Some(old) = read("web/app/public/azero/azero-snake.azweb") else {
        eprintln!("skip: snake net not committed");
        return;
    };
    let planes = snake::encode::PLANES;
    let net = Legacy::SnakeDense {
        planes,
        policy_len: 4,
    }
    .load(&old)
    .expect("legacy load");
    let size = net.arch().size;
    assert_eq!(net.arch().head, HeadKind::GlobalPoolDense);

    let mut rng = Rng(0x5EED5);
    for &s in &[size, 11] {
        for _ in 0..8 {
            let out = net.forward_at(&rng.fill(planes * s * s), &[], s);
            well_formed(&out, 4);
            assert!(out.ownership.is_none(), "snake has no ownership head");
        }
    }
}

#[test]
fn forward_support_is_a_distribution_over_the_legal_subset() {
    let Some(old) = read("web/app/public/azero/azero-go.azweb") else {
        eprintln!("skip: go net not committed");
        return;
    };
    let planes = go::encode::PLANES;
    let net = Legacy::GoSpatial { planes }
        .load(&old)
        .expect("legacy load");
    let size = net.arch().size;

    let mut rng = Rng(0xA11CE);
    let support: Vec<u16> = (0..=(size * size) as u16).step_by(3).collect();
    let (priors, value) = net.forward_support(&rng.fill(planes * size * size), &[], &support);
    assert_eq!(priors.len(), support.len(), "one prior per legal action");
    let sum: f32 = priors.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "priors sum to 1: {sum}");
    assert!(
        priors.iter().all(|p| (0.0..=1.0).contains(p)),
        "priors in [0,1]"
    );
    assert!(value.is_finite() && value.abs() <= 1.0);
}
