//! Pack a slice of [`Obs`] into the two tensors the net wants: a grid tensor
//! `[B, CHANNELS, GRID, GRID]` and a scalar tensor `[B, SCALARS]`. The env stores
//! each grid already flattened channel-major/row-major, so the grid copy is a
//! straight memcpy of contiguous slices; `reshape` then reinterprets it with no
//! data movement.

use slither_rl::obs::Obs;
use tch::{Device, Tensor};

use crate::net::{CHANNELS, GRID, SCALARS};

const GRID_LEN: usize = (CHANNELS * GRID * GRID) as usize;

/// Build `(grid [B,C,G,G], scalars [B,S])` on `device` from observations.
pub fn pack(obs: &[Obs], device: Device) -> (Tensor, Tensor) {
    let b = obs.len();
    let mut grid = Vec::with_capacity(b * GRID_LEN);
    let mut scalars = Vec::with_capacity(b * SCALARS as usize);
    for o in obs {
        debug_assert_eq!(o.grid.len(), GRID_LEN);
        grid.extend_from_slice(&o.grid);
        scalars.extend_from_slice(&o.scalars);
    }
    let grid = Tensor::from_slice(&grid)
        .reshape([b as i64, CHANNELS, GRID, GRID])
        .to_device(device);
    let scalars = Tensor::from_slice(&scalars)
        .reshape([b as i64, SCALARS])
        .to_device(device);
    (grid, scalars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slither_rl::obs::{GRID, Obs};

    /// Packing a batch preserves each obs's data in the right `[B,C,G,G]` cell —
    /// a mismatch here would silently feed the net transposed/garbled grids.
    #[test]
    fn pack_preserves_layout() {
        let mut a = Obs::zeros();
        let mut b = Obs::zeros();
        // Distinctive markers at known flat indices and scalars.
        a.grid[0] = 1.0;
        a.grid[GRID * GRID] = 2.0; // channel 1, cell (0,0)
        a.scalars = [0.1, 0.2, 0.3];
        b.grid[7] = 5.0;
        b.scalars = [0.4, 0.5, 0.6];

        let (grid, scalars) = pack(&[a.clone(), b.clone()], Device::Cpu);
        assert_eq!(grid.size(), [2, CHANNELS, GRID as i64, GRID as i64]);
        assert_eq!(scalars.size(), [2, SCALARS]);

        let flat = grid.reshape([2, -1]);
        let row0: Vec<f32> = (&flat.get(0)).try_into().unwrap();
        let row1: Vec<f32> = (&flat.get(1)).try_into().unwrap();
        assert_eq!(row0[0], 1.0);
        assert_eq!(row0[GRID * GRID], 2.0);
        assert_eq!(row1[7], 5.0);
        let s: Vec<f32> = (&scalars.reshape([-1])).try_into().unwrap();
        assert_eq!(&s, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }
}
