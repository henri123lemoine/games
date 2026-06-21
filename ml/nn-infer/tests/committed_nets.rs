//! Self-consistency of the generic forward on the *committed trained nets*,
//! which are now `AZNET1` (the per-game `azinfer`/`goinfer`/`snakeinfer` forwards
//! and the legacy magics they read are gone). Guards that each committed net
//! parses, its header round-trips, the forward is well-formed (finite logits,
//! `value ∈ [-1,1]`, ownership where the head carries it), it is board-size
//! agnostic for the global-pool heads, and `forward_support` yields a valid
//! distribution over the legal subset.

use nn_infer::{HeadKind, Net};

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

/// The committed nets are AZNET1: their magic is the unified one, and the header
/// re-serializes to the same bytes it parsed from (round-trip).
fn assert_aznet1_round_trip(bytes: &[u8]) {
    assert_eq!(
        &bytes[..8],
        nn_infer::format::MAGIC.as_slice(),
        "AZNET1 magic"
    );
    let net = Net::parse(bytes).expect("AZNET1 parse");
    let header = net.arch().header_bytes();
    assert_eq!(
        header,
        &bytes[..nn_infer::format::HEADER_LEN],
        "header round-trips to the same bytes"
    );
}

#[test]
fn chess_net_loads_and_is_well_formed() {
    let Some(bytes) = read("web/app/public/azero/azero-chess.azweb") else {
        eprintln!("skip: chess net not committed");
        return;
    };
    assert_aznet1_round_trip(&bytes);
    let net = Net::parse(&bytes).expect("parse");
    assert_eq!(net.arch().head, HeadKind::FlatConv);
    let planes = net.arch().planes;
    let policy_len = net.arch().policy_len;

    let mut rng = Rng(0xC0FFEE);
    for _ in 0..16 {
        well_formed(&net.forward(&rng.fill(planes * 64), &[]), policy_len);
    }
}

#[test]
fn go_net_loads_well_formed_size_agnostic_with_ownership() {
    let Some(bytes) = read("web/app/public/azero/azero-go.azweb") else {
        eprintln!("skip: go net not committed");
        return;
    };
    assert_aznet1_round_trip(&bytes);
    let net = Net::parse(&bytes).expect("parse");
    assert_eq!(net.arch().head, HeadKind::GlobalPoolSpatial);
    assert!(
        net.arch().flags.ownership(),
        "the committed go net carries the ownership head"
    );
    let planes = net.arch().planes;
    let size = net.arch().size;

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
    let Some(bytes) = read("web/app/public/azero/azero-snake.azweb") else {
        eprintln!("skip: snake net not committed");
        return;
    };
    assert_aznet1_round_trip(&bytes);
    let net = Net::parse(&bytes).expect("parse");
    assert_eq!(net.arch().head, HeadKind::GlobalPoolDense);
    let planes = net.arch().planes;
    let size = net.arch().size;

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
    let Some(bytes) = read("web/app/public/azero/azero-go.azweb") else {
        eprintln!("skip: go net not committed");
        return;
    };
    let net = Net::parse(&bytes).expect("parse");
    let planes = net.arch().planes;
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
