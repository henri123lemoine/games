//! Bit-for-bit parity of the generic `nn-infer` forward against the four
//! original per-game forwards on the *committed trained nets*. Each old export
//! (`AZWEB001`/`AZWEBGO3`/`AZSNK1`) is rewrapped into an `AZNET1` buffer — the
//! weight stream byte-order is unchanged, only the header is rewritten — parsed
//! by `nn-infer`, and its policy/value/ownership outputs are required to match
//! the old crate's `forward` *exactly* (`==`, not within a tolerance) over many
//! random inputs. This is the gate the migration is allowed to cross.

use nn_infer::{Arch, HeadFlags, HeadKind, Net};

/// A tiny xorshift RNG so the fixtures are reproducible without a dep.
struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // [-1, 1): plausible post-encoding feature magnitudes.
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

/// The committed nets' body (everything after the old header) is byte-identical
/// to the AZNET1 body, so an AZNET1 buffer is `new_header ∥ old_body`.
fn rewrap(old: &[u8], old_header_len: usize, arch: Arch) -> Vec<u8> {
    let mut b = arch.header_bytes();
    b.extend_from_slice(&old[old_header_len..]);
    b
}

#[test]
fn chess_flat_conv_matches_azinfer_bit_for_bit() {
    let Some(old) = read("web/app/public/azero/azero-chess.azweb") else {
        eprintln!("skip: chess net not committed");
        return;
    };
    let blocks = u32::from_le_bytes(old[8..12].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(old[12..16].try_into().unwrap()) as usize;
    let planes = chess::encode::PLANE_COUNT;
    let arch = Arch {
        blocks,
        channels,
        planes,
        size: 8,
        scalars: 0,
        head: HeadKind::FlatConv,
        policy_len: chess::encode::AZ_POLICY_LEN,
        flags: HeadFlags::default(),
    };
    let reference = azinfer::model::Model::parse(&old).expect("azinfer parse");
    let net = Net::parse(&rewrap(&old, 16, arch)).expect("nn-infer parse");

    let mut rng = Rng(0xC0FFEE);
    let mut worst = 0u32;
    for _ in 0..32 {
        let feats = rng.fill(planes * 64);
        let (rp, rv) = reference.forward(&feats);
        let out = net.forward(&feats, &[]);
        assert_eq!(out.policy, rp, "chess policy must match azinfer exactly");
        assert_eq!(out.value, rv, "chess value must match azinfer exactly");
        worst += 1;
    }
    assert_eq!(worst, 32);
}

#[test]
fn go_spatial_with_ownership_matches_goinfer_bit_for_bit() {
    let Some(old) = read("web/app/public/azero/azero-go.azweb") else {
        eprintln!("skip: go net not committed");
        return;
    };
    let blocks = u32::from_le_bytes(old[8..12].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(old[12..16].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(old[16..20].try_into().unwrap()) as usize;
    let planes = go::encode::PLANES;
    let ownership = &old[..8] == b"AZWEBGO3";
    let arch = Arch {
        blocks,
        channels,
        planes,
        size,
        scalars: 0,
        head: HeadKind::GlobalPoolSpatial,
        policy_len: 0,
        flags: HeadFlags(if ownership { HeadFlags::OWNERSHIP } else { 0 }),
    };
    let reference = goinfer::model::Model::parse(&old).expect("goinfer parse");
    let net = Net::parse(&rewrap(&old, 20, arch)).expect("nn-infer parse");
    assert!(
        ownership,
        "the committed go net is AZWEBGO3 (ownership head)"
    );

    let mut rng = Rng(0x90D90D);
    // Exercise the trained size and a smaller board (global-pool size-agnostic).
    for &s in &[size, 9] {
        for _ in 0..16 {
            let feats = rng.fill(planes * s * s);
            let (rp, rv) = reference.forward_at(&feats, s);
            let out = net.forward_at(&feats, &[], s);
            assert_eq!(out.policy, rp, "go policy must match goinfer exactly @ {s}");
            assert_eq!(out.value, rv, "go value must match goinfer exactly @ {s}");
            let ro = reference.ownership_at(&feats, s);
            assert_eq!(out.ownership, ro, "go ownership must match goinfer @ {s}");
        }
    }
}

#[test]
fn snake_dense_matches_snakeinfer_bit_for_bit() {
    let Some(old) = read("web/app/public/azero/azero-snake.azweb") else {
        eprintln!("skip: snake net not committed");
        return;
    };
    let blocks = u32::from_le_bytes(old[6..10].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(old[10..14].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(old[14..18].try_into().unwrap()) as usize;
    let planes = snake::encode::PLANES;
    let arch = Arch {
        blocks,
        channels,
        planes,
        size,
        scalars: 0,
        head: HeadKind::GlobalPoolDense,
        policy_len: 4,
        flags: HeadFlags::default(),
    };
    let reference = snakeinfer::model::Model::parse(&old).expect("snakeinfer parse");
    let net = Net::parse(&rewrap(&old, 18, arch)).expect("nn-infer parse");

    let mut rng = Rng(0x5EED5);
    for &s in &[size, 11] {
        for _ in 0..16 {
            let feats = rng.fill(planes * s * s);
            let (rp, rv) = reference.forward_at(&feats, s);
            let out = net.forward_at(&feats, &[], s);
            assert_eq!(out.policy, rp, "snake policy must match snakeinfer @ {s}");
            assert_eq!(out.value, rv, "snake value must match snakeinfer @ {s}");
            assert!(out.ownership.is_none());
        }
    }
}
