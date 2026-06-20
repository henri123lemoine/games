//! Loading the pre-`AZNET1` exports (`AZWEB001` / `AZWEBGO2` / `AZWEBGO3` /
//! `AZSNK1`) through the generic engine, so the committed trained nets keep
//! playing before the trainers re-export. The legacy headers don't name the
//! input-plane count or the head topology — that is game knowledge — so the
//! game-aware caller supplies it; this module only recognizes the magic,
//! validates the legacy dims, and synthesizes the [`Arch`]. The weight stream
//! after the legacy header is byte-identical to the `AZNET1` body, so the
//! parsed [`Net`] is the same one the new format would produce.

use crate::Net;
use crate::format::{Arch, HeadFlags, HeadKind};

/// The legacy export families, keyed by magic. Each carries the per-game facts
/// the legacy header omits; the caller (which knows the game) picks the variant.
pub enum Legacy {
    /// `AZWEB001`: chess flat-conv. Board-fixed 8×8; `planes` input features and
    /// a `policy_len`-wide flat move space.
    FlatConv { planes: usize, policy_len: usize },
    /// `AZWEBGO2`/`AZWEBGO3`: go global-pool spatial policy; `AZWEBGO3` carries
    /// the ownership head. `planes` input features.
    GoSpatial { planes: usize },
    /// `AZSNK1`: snake global-pool dense policy over `policy_len` actions;
    /// `planes` input features.
    SnakeDense { planes: usize, policy_len: usize },
}

impl Legacy {
    /// Parses `data` as the chosen legacy family into a generic [`Net`]. The
    /// architecture dims come from the legacy header; the head topology and
    /// plane count come from `self`.
    pub fn load(&self, data: &[u8]) -> Result<Net, String> {
        let bytes = match self {
            Legacy::FlatConv { planes, policy_len } => {
                let (blocks, channels) = flat_dims(data)?;
                Arch {
                    blocks,
                    channels,
                    planes: *planes,
                    size: 8,
                    scalars: 0,
                    head: HeadKind::FlatConv,
                    policy_len: *policy_len,
                    flags: HeadFlags::default(),
                }
                .rewrap(data, FLAT_HEADER)
            }
            Legacy::GoSpatial { planes } => {
                let (blocks, channels, size, ownership) = go_dims(data)?;
                Arch {
                    blocks,
                    channels,
                    planes: *planes,
                    size,
                    scalars: 0,
                    head: HeadKind::GlobalPoolSpatial,
                    policy_len: 0,
                    flags: HeadFlags(if ownership { HeadFlags::OWNERSHIP } else { 0 }),
                }
                .rewrap(data, GO_HEADER)
            }
            Legacy::SnakeDense { planes, policy_len } => {
                let (blocks, channels, size) = snake_dims(data)?;
                Arch {
                    blocks,
                    channels,
                    planes: *planes,
                    size,
                    scalars: 0,
                    head: HeadKind::GlobalPoolDense,
                    policy_len: *policy_len,
                    flags: HeadFlags::default(),
                }
                .rewrap(data, SNAKE_HEADER)
            }
        }?;
        Net::parse(&bytes)
    }
}

impl Arch {
    /// `AZNET1` buffer = this header ∥ the legacy body (the bytes after its
    /// `old_header` length), which is already in `AZNET1` layer order.
    fn rewrap(&self, data: &[u8], old_header: usize) -> Result<Vec<u8>, String> {
        if data.len() < old_header {
            return Err("truncated legacy header".into());
        }
        let mut b = self.header_bytes();
        b.extend_from_slice(&data[old_header..]);
        Ok(b)
    }
}

/// `AZWEB001` header: magic(8) + blocks,channels (u32 each).
const FLAT_HEADER: usize = 8 + 2 * 4;
/// `AZWEBGO2/3` header: magic(8) + blocks,channels,size (u32 each).
const GO_HEADER: usize = 8 + 3 * 4;
/// `AZSNK1` header: magic(6) + blocks,channels,size (u32 each).
const SNAKE_HEADER: usize = 6 + 3 * 4;

fn u32_at(data: &[u8], i: usize) -> Result<usize, String> {
    data.get(i..i + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .ok_or_else(|| "truncated legacy header".into())
}

fn flat_dims(data: &[u8]) -> Result<(usize, usize), String> {
    if data.get(..8) != Some(b"AZWEB001".as_slice()) {
        return Err("not an AZWEB001 export".into());
    }
    Ok((u32_at(data, 8)?, u32_at(data, 12)?))
}

fn go_dims(data: &[u8]) -> Result<(usize, usize, usize, bool), String> {
    let ownership = match data.get(..8) {
        Some(b"AZWEBGO3") => true,
        Some(b"AZWEBGO2") => false,
        _ => return Err("not an AZWEBGO2/AZWEBGO3 export".into()),
    };
    Ok((
        u32_at(data, 8)?,
        u32_at(data, 12)?,
        u32_at(data, 16)?,
        ownership,
    ))
}

fn snake_dims(data: &[u8]) -> Result<(usize, usize, usize), String> {
    if data.get(..6) != Some(b"AZSNK1".as_slice()) {
        return Err("not an AZSNK1 export".into());
    }
    Ok((u32_at(data, 6)?, u32_at(data, 10)?, u32_at(data, 14)?))
}
