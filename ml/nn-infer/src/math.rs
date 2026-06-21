//! The fp32 forward primitives shared by every conv-resnet head: same-padding
//! convolution, global pooling, and dense layers. Plain loops — built for
//! correctness and wasm portability, not speed. The conv is an axpy-shaped sweep
//! that is bit-identical to the textbook per-output-cell gather: for any fixed
//! output cell the `(c_in, ky, kx)` terms are summed in the same order, so a
//! trained net's argmax/sign decisions are unchanged.

use crate::format::{Conv, Linear};

/// `19.0` centers the global-pool size-scale; matches every trainer.
pub const POOL_SIZE_REF: f32 = 19.0;

/// `size`×`size` same-padding stride-1 convolution, channel-major `[c, area]`
/// layout, writing into `out` (resized to `c_out·area`).
pub fn conv_fwd(conv: &Conv, x: &[f32], size: usize, relu: bool, out: &mut Vec<f32>) {
    let area = size * size;
    out.clear();
    out.resize(conv.c_out * area, 0.0);
    let k = conv.k;
    let half = (k / 2) as isize;
    let s = size as isize;
    for co in 0..conv.c_out {
        let out_plane = &mut out[co * area..(co + 1) * area];
        out_plane.fill(conv.b[co]);
        for ci in 0..conv.c_in {
            let wbase = (co * conv.c_in + ci) * k * k;
            let in_plane = &x[ci * area..(ci + 1) * area];
            for ky in 0..k {
                let dy = ky as isize - half;
                let y0 = (-dy).max(0);
                let y1 = (s - dy).min(s);
                for kx in 0..k {
                    let w = conv.w[wbase + ky * k + kx];
                    if w == 0.0 {
                        continue;
                    }
                    let dx = kx as isize - half;
                    let x0 = (-dx).max(0);
                    let x1 = (s - dx).min(s);
                    if x0 >= x1 {
                        continue;
                    }
                    let span = (x1 - x0) as usize;
                    for y in y0..y1 {
                        let o = (y * s + x0) as usize;
                        let i = ((y + dy) * s + (x0 + dx)) as usize;
                        let dst = &mut out_plane[o..o + span];
                        let src = &in_plane[i..i + span];
                        axpy(w, src, dst);
                    }
                }
            }
        }
        if relu {
            for v in out_plane.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
}

/// Allocating wrapper for one-shot callers (heads, ownership) that don't keep
/// scratch around.
pub fn conv_fwd_vec(conv: &Conv, x: &[f32], size: usize, relu: bool) -> Vec<f32> {
    let mut out = Vec::new();
    conv_fwd(conv, x, size, relu, &mut out);
    out
}

/// `dst[i] += w · src[i]`, a fused multiply-add over unit-stride slices the
/// compiler lowers to `f32x4` lanes under `+simd128`.
#[inline(always)]
fn axpy(w: f32, src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += w * s;
    }
}

/// Global pooling: channel-major `[c, area]` → `[3c]` = per-channel mean, then
/// board-size-scaled mean, then max. Mirrors every trainer's `global_pool`.
pub fn global_pool(x: &[f32], c: usize, area: usize) -> Vec<f32> {
    let scale = (area as f32).sqrt() / POOL_SIZE_REF;
    let mut out = vec![0.0f32; 3 * c];
    for ch in 0..c {
        let plane = &x[ch * area..(ch + 1) * area];
        let mut sum = 0.0f32;
        let mut mx = f32::NEG_INFINITY;
        for &v in plane {
            sum += v;
            mx = mx.max(v);
        }
        let mean = sum / area as f32;
        out[ch] = mean;
        out[c + ch] = mean * scale;
        out[2 * c + ch] = mx;
    }
    out
}

pub fn linear_fwd(l: &Linear, x: &[f32], relu: bool) -> Vec<f32> {
    (0..l.n_out)
        .map(|o| {
            let acc = l.b[o]
                + l.w[o * l.n_in..(o + 1) * l.n_in]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| w * v)
                    .sum::<f32>();
            if relu { acc.max(0.0) } else { acc }
        })
        .collect()
}
