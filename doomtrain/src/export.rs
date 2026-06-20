use std::io::{Read, Write};
use std::path::Path;

use tch::nn::VarStore;
use tch::Kind;

const MAGIC: &[u8; 8] = b"DOOMDFP1";

/// Export every trainable tensor as a portable flat file the eventual
/// wasm/in-browser forward can load without tch: magic, count, then for each
/// tensor: name (u16 len + utf8), ndim (u8), dims (u32 each), fp32 data
/// (little-endian, row-major). Mirrors azt/azsnake's flat-export idea.
pub fn export(vs: &VarStore, path: &Path) {
    let vars = vs.variables();
    let mut names: Vec<String> = vars.keys().cloned().collect();
    names.sort();

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(names.len() as u32).to_le_bytes());

    for name in &names {
        let t = vars[name].to_kind(Kind::Float).to_device(tch::Device::Cpu);
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        buf.extend_from_slice(nb);
        let dims = t.size();
        buf.push(dims.len() as u8);
        for d in &dims {
            buf.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        let data = Vec::<f32>::try_from(t.flatten(0, -1)).unwrap();
        for x in &data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    let mut f = std::fs::File::create(path).expect("create export file");
    f.write_all(&buf).expect("write export");
}

pub struct Exported {
    pub tensors: std::collections::HashMap<String, (Vec<i64>, Vec<f32>)>,
}

/// Read back a flat export (for the round-trip check and as the reference the
/// wasm loader will mirror).
pub fn read(path: &Path) -> Exported {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .expect("open export")
        .read_to_end(&mut bytes)
        .expect("read export");

    let mut p = 0usize;
    let take = |p: &mut usize, n: usize, bytes: &[u8]| -> Vec<u8> {
        let s = bytes[*p..*p + n].to_vec();
        *p += n;
        s
    };
    assert_eq!(&take(&mut p, 8, &bytes)[..], MAGIC, "bad magic");
    let count = u32::from_le_bytes(take(&mut p, 4, &bytes).try_into().unwrap()) as usize;

    let mut tensors = std::collections::HashMap::new();
    for _ in 0..count {
        let nlen = u16::from_le_bytes(take(&mut p, 2, &bytes).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&mut p, nlen, &bytes)).unwrap();
        let ndim = take(&mut p, 1, &bytes)[0] as usize;
        let mut dims = Vec::with_capacity(ndim);
        let mut numel = 1i64;
        for _ in 0..ndim {
            let d = u32::from_le_bytes(take(&mut p, 4, &bytes).try_into().unwrap()) as i64;
            dims.push(d);
            numel *= d;
        }
        let mut data = Vec::with_capacity(numel as usize);
        for _ in 0..numel {
            data.push(f32::from_le_bytes(
                take(&mut p, 4, &bytes).try_into().unwrap(),
            ));
        }
        tensors.insert(name, (dims, data));
    }
    Exported { tensors }
}

/// Round-trip check: every VarStore tensor matches the re-read export exactly.
pub fn verify_roundtrip(vs: &VarStore, path: &Path) -> bool {
    let exported = read(path);
    let vars = vs.variables();
    for (name, t) in &vars {
        let (dims, data) = match exported.tensors.get(name) {
            Some(v) => v,
            None => {
                eprintln!("missing exported tensor {name}");
                return false;
            }
        };
        if dims != &t.size() {
            eprintln!("dim mismatch for {name}");
            return false;
        }
        let orig = Vec::<f32>::try_from(
            t.to_kind(Kind::Float)
                .to_device(tch::Device::Cpu)
                .flatten(0, -1),
        )
        .unwrap();
        let max_diff = orig
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_diff > 1e-6 {
            eprintln!("value mismatch for {name}: max_diff={max_diff}");
            return false;
        }
    }
    true
}

/// Load a previously-saved tch checkpoint (.ot) into a VarStore in place.
pub fn load_checkpoint(vs: &mut VarStore, path: &Path) {
    vs.load(path).unwrap_or_else(|e| {
        eprintln!("failed to load checkpoint {}: {e}", path.display());
        std::process::exit(1);
    });
}
