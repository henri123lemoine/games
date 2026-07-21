//! The `AZNET1` container: the self-describing header and the `Reader`
//! primitives that walk the BN-folded weight stream. One parser for every
//! exported conv-resnet net. The header names the architecture (`blocks`,
//! `channels`, `planes`, `size`, `scalars`) and the head topology (`HeadKind`,
//! `policy_len`, `HeadFlags`); the body is fp32 little-endian weights in a fixed
//! layer order. The `Reader{floats,conv,linear}` set and the no-trailing-bytes
//! check are the format's integrity boundary.
//!
//! **Body layout.** Every conv carries a bias — including chess's unused
//! zero-padded policy-conv bias (see [`HeadKind::FlatConv`]) — so the runtime
//! needs only conv+bias / linear / relu / tanh. The trainer's exporter writes
//! the header, then dumps the BN-folded weights in this order.

/// `AZNET1\0\0`: the unified magic, padded to 8 bytes.
pub const MAGIC: &[u8; 8] = b"AZNET1\0\0";

/// The only format version. New heads extend [`HeadFlags`] rather than bumping
/// this; a structural change to the container would.
pub const VERSION: u32 = 1;

/// Number of `u32` header fields after the magic.
const HEADER_FIELDS: usize = 10;
/// Total header byte count: the 8-byte magic plus the ten `u32` fields (48).
/// The weight stream begins here.
pub const HEADER_LEN: usize = MAGIC.len() + HEADER_FIELDS * 4;

/// Which policy/value head topology the trunk feeds. The trained nets pair a
/// policy shape with a value shape, so this is one closed enumeration rather
/// than independent flags that could express an unbuilt combination.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeadKind {
    /// Chess: a 1×1 conv policy flattened to a fixed move space (`policy_len`),
    /// and a value head that flattens a small conv to a dense MLP. Board-fixed.
    ///
    /// The policy conv `p2` (`C`→`policy_len/size²`) **carries a bias like every
    /// other conv**, even though chess never uses it: the exporter writes
    /// `policy_len/size²` (= 73) zero floats there to honor the runtime's "every
    /// conv has a bias" invariant. The exporter must emit those zeros; the reader
    /// consumes them via [`Reader::conv`].
    FlatConv,
    /// Go: a 1×1 conv policy biased by the global pool, one logit per board
    /// point plus a pooled pass logit (`size²+1` wide), and a global-pool value
    /// MLP. Board-size-agnostic.
    GlobalPoolSpatial,
    /// Snake: a 1×1 conv policy pooled to a small fixed action set
    /// (`policy_len`), and a global-pool value MLP. Board-size-agnostic.
    GlobalPoolDense,
}

impl HeadKind {
    fn from_u32(v: u32) -> Result<HeadKind, String> {
        match v {
            0 => Ok(HeadKind::FlatConv),
            1 => Ok(HeadKind::GlobalPoolSpatial),
            2 => Ok(HeadKind::GlobalPoolDense),
            other => Err(format!("unknown head_kind {other}")),
        }
    }

    pub(crate) fn as_u32(self) -> u32 {
        match self {
            HeadKind::FlatConv => 0,
            HeadKind::GlobalPoolSpatial => 1,
            HeadKind::GlobalPoolDense => 2,
        }
    }
}

/// Optional heads appended after the value head, one flag bit each. Old readers
/// reject unknown bits rather than misparsing trailing weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct HeadFlags(pub u32);

impl HeadFlags {
    /// Go's per-point ownership head (`o1`, a bias-less 1×1 `C`→1 conv → tanh).
    pub const OWNERSHIP: u32 = 1;
    /// Multi-seat value head: the reserved header word carries the seat count
    /// and the value head's final linear emits that many raw logits (softmax
    /// over seats at the consumer) instead of one tanh scalar.
    pub const VALUE_SEATS: u32 = 2;

    pub fn ownership(self) -> bool {
        self.0 & Self::OWNERSHIP != 0
    }

    pub fn value_seats(self) -> bool {
        self.0 & Self::VALUE_SEATS != 0
    }
}

/// The header: everything a parser or WebGPU driver needs to lay out the net,
/// read entirely from the file (never from a game identity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Arch {
    pub blocks: usize,
    pub channels: usize,
    pub planes: usize,
    pub size: usize,
    pub scalars: usize,
    pub head: HeadKind,
    /// Flat policy width for [`HeadKind::FlatConv`] / [`HeadKind::GlobalPoolDense`];
    /// `0` for [`HeadKind::GlobalPoolSpatial`], whose width is `size²+1`.
    pub policy_len: usize,
    pub flags: HeadFlags,
    /// Value-head width: 1 is the scalar tanh head; >1 emits raw per-seat
    /// logits (multiplayer win shares after a softmax).
    pub value_seats: usize,
}

impl Arch {
    /// Parses and bounds-checks the header, returning it alongside the byte
    /// offset where the weight stream begins.
    pub fn parse(data: &[u8]) -> Result<(Arch, usize), String> {
        if data.len() < HEADER_LEN {
            return Err("truncated AZNET1 header".into());
        }
        if &data[..MAGIC.len()] != MAGIC.as_slice() {
            return Err("not an AZNET1 export".into());
        }
        let u32_at = |i: usize| {
            let off = MAGIC.len() + i * 4;
            u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
        };
        let version = u32_at(0);
        if version != VERSION {
            return Err(format!("unsupported AZNET1 version {version}"));
        }
        let blocks = u32_at(1) as usize;
        let channels = u32_at(2) as usize;
        let planes = u32_at(3) as usize;
        let size = u32_at(4) as usize;
        let scalars = u32_at(5) as usize;
        let head = HeadKind::from_u32(u32_at(6))?;
        let policy_len = u32_at(7) as usize;
        let flags = HeadFlags(u32_at(8));
        let reserved = u32_at(9);

        if blocks == 0 || blocks > 64 || channels == 0 || channels > 1024 {
            return Err(format!("implausible architecture {blocks}x{channels}"));
        }
        if planes == 0 || planes > 1024 {
            return Err(format!("implausible planes {planes}"));
        }
        if !(1..=64).contains(&size) {
            return Err(format!("implausible size {size}"));
        }
        if flags.0 & !(HeadFlags::OWNERSHIP | HeadFlags::VALUE_SEATS) != 0 {
            return Err(format!("unknown head flags {:#x}", flags.0));
        }
        let value_seats = if flags.value_seats() {
            if !(2..=8).contains(&(reserved as usize)) {
                return Err(format!("implausible value seat count {reserved}"));
            }
            reserved as usize
        } else {
            if reserved != 0 {
                return Err(format!("nonzero reserved header word {reserved}"));
            }
            1
        };
        if value_seats > 1 && head == HeadKind::FlatConv {
            return Err("multi-seat value requires a global-pool head".into());
        }
        match head {
            HeadKind::GlobalPoolSpatial if policy_len != 0 => {
                return Err("spatial policy must carry policy_len 0".into());
            }
            HeadKind::FlatConv | HeadKind::GlobalPoolDense if policy_len == 0 => {
                return Err("flat/dense policy must carry a nonzero policy_len".into());
            }
            _ => {}
        }
        Ok((
            Arch {
                blocks,
                channels,
                planes,
                size,
                scalars,
                head,
                policy_len,
                flags,
                value_seats,
            },
            HEADER_LEN,
        ))
    }

    /// Serializes the 48-byte header (magic + ten `u32`s). The exporter writes
    /// this, then appends the BN-folded weight stream in the fixed layer order.
    pub fn header_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER_LEN);
        b.extend_from_slice(MAGIC);
        let (flags, reserved) = if self.value_seats > 1 {
            (
                self.flags.0 | HeadFlags::VALUE_SEATS,
                self.value_seats as u32,
            )
        } else {
            (self.flags.0 & !HeadFlags::VALUE_SEATS, 0)
        };
        for word in [
            VERSION,
            self.blocks as u32,
            self.channels as u32,
            self.planes as u32,
            self.size as u32,
            self.scalars as u32,
            self.head.as_u32(),
            self.policy_len as u32,
            flags,
            reserved,
        ] {
            b.extend_from_slice(&word.to_le_bytes());
        }
        b
    }
}

/// A `[c_out, c_in, k, k]` conv with a (possibly folded-in or zero) bias.
pub struct Conv {
    /// `[c_out, c_in, k, k]` flattened, k ∈ {1, 3}.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
}

/// A `[out, in]` dense layer with a bias.
pub struct Linear {
    /// `[out, in]` flattened.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
}

/// Sequential reader over the fp32 weight stream. The `floats`/`conv`/`linear`
/// primitives and the truncation check are the format's integrity boundary.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos }
    }

    pub fn floats(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = n
            .checked_mul(4)
            .filter(|b| self.data.len() - self.pos >= *b)
            .ok_or_else(|| format!("truncated export at offset {}", self.pos))?;
        let out = self.data[self.pos..self.pos + bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.pos += bytes;
        Ok(out)
    }

    pub fn conv(&mut self, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        Ok(Conv {
            w: self.floats(c_out * c_in * k * k)?,
            b: self.floats(c_out)?,
            c_in,
            c_out,
            k,
        })
    }

    /// A conv whose stored bias is omitted from the file (the placement /
    /// ownership convs); the in-memory bias is zero.
    pub fn conv_nobias(&mut self, c_in: usize, c_out: usize, k: usize) -> Result<Conv, String> {
        Ok(Conv {
            w: self.floats(c_out * c_in * k * k)?,
            b: vec![0.0; c_out],
            c_in,
            c_out,
            k,
        })
    }

    pub fn linear(&mut self, n_in: usize, n_out: usize) -> Result<Linear, String> {
        Ok(Linear {
            w: self.floats(n_out * n_in)?,
            b: self.floats(n_out)?,
            n_in,
            n_out,
        })
    }

    /// Rejects a body with weights left unread — the export must consume exactly.
    pub fn finish(&self) -> Result<(), String> {
        if self.pos != self.data.len() {
            return Err(format!(
                "{} trailing bytes in export",
                self.data.len() - self.pos
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch() -> Arch {
        Arch {
            blocks: 4,
            channels: 64,
            planes: 8,
            size: 19,
            scalars: 0,
            head: HeadKind::GlobalPoolSpatial,
            policy_len: 0,
            flags: HeadFlags(HeadFlags::OWNERSHIP),
            value_seats: 1,
        }
    }

    #[test]
    fn value_seats_round_trip() {
        let a = Arch {
            head: HeadKind::GlobalPoolDense,
            policy_len: 200,
            flags: HeadFlags::default(),
            size: 1,
            value_seats: 4,
            ..arch()
        };
        let (parsed, _) = Arch::parse(&a.header_bytes()).expect("parse");
        assert_eq!(parsed.value_seats, 4);
        assert!(parsed.flags.value_seats());
    }

    #[test]
    fn value_seats_rejects_flat_head_and_wild_counts() {
        let mut a = arch();
        a.value_seats = 4;
        a.head = HeadKind::FlatConv;
        a.policy_len = 4672;
        assert!(
            Arch::parse(&a.header_bytes()).is_err(),
            "flat head rejected"
        );
        let mut b = arch();
        b.head = HeadKind::GlobalPoolDense;
        b.policy_len = 8;
        b.value_seats = 9;
        assert!(Arch::parse(&b.header_bytes()).is_err(), "9 seats rejected");
    }

    #[test]
    fn header_round_trips_through_parse() {
        let a = arch();
        let bytes = a.header_bytes();
        let (parsed, body) = Arch::parse(&bytes).expect("parse");
        assert_eq!(parsed, a);
        assert_eq!(body, bytes.len(), "body begins right after the header");
    }

    #[test]
    fn every_head_kind_round_trips() {
        for (head, policy_len) in [
            (HeadKind::FlatConv, 4672),
            (HeadKind::GlobalPoolSpatial, 0),
            (HeadKind::GlobalPoolDense, 4),
        ] {
            let a = Arch {
                head,
                policy_len,
                flags: HeadFlags::default(),
                ..arch()
            };
            let (parsed, _) = Arch::parse(&a.header_bytes()).expect("parse");
            assert_eq!(parsed.head, head);
            assert_eq!(parsed.policy_len, policy_len);
        }
    }

    #[test]
    fn rejects_wrong_magic_and_version() {
        let mut bytes = arch().header_bytes();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(Arch::parse(&bad).is_err(), "wrong magic rejected");
        // Bump the version word (first u32 after the magic).
        bytes[MAGIC.len()] = 2;
        assert!(Arch::parse(&bytes).is_err(), "unknown version rejected");
    }

    #[test]
    fn rejects_truncated_header() {
        let bytes = arch().header_bytes();
        assert!(Arch::parse(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn rejects_unknown_head_kind_and_flags() {
        let set = |a: &Arch, field: usize, v: u32| {
            let mut b = a.header_bytes();
            let off = MAGIC.len() + field * 4;
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
            b
        };
        let a = arch();
        assert!(Arch::parse(&set(&a, 6, 3)).is_err(), "head_kind 3 rejected");
        assert!(
            Arch::parse(&set(&a, 8, 0xFF)).is_err(),
            "unknown flag bits rejected"
        );
        assert!(
            Arch::parse(&set(&a, 9, 1)).is_err(),
            "nonzero reserved word rejected"
        );
    }

    #[test]
    fn rejects_inconsistent_policy_len() {
        let mut spatial = arch();
        spatial.policy_len = 100;
        assert!(
            Arch::parse(&spatial.header_bytes()).is_err(),
            "spatial policy must carry policy_len 0"
        );
        let dense = Arch {
            head: HeadKind::GlobalPoolDense,
            policy_len: 0,
            ..arch()
        };
        assert!(
            Arch::parse(&dense.header_bytes()).is_err(),
            "dense policy needs a nonzero policy_len"
        );
    }

    #[test]
    fn reader_finish_rejects_trailing_bytes() {
        let data = [0u8; 12];
        let mut r = Reader::new(&data, 0);
        let _ = r.floats(2).unwrap();
        assert!(r.finish().is_err(), "4 trailing bytes rejected");
        let _ = r.floats(1).unwrap();
        assert!(r.finish().is_ok(), "exact consumption accepted");
    }
}
