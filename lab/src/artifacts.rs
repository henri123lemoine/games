//! Artifact bytes (trained nets, solver tables) behind one lookup: files on
//! native; on wasm an in-memory store the host fills via the engine's
//! `load_artifact` before creating matches that need them.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static STORE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Registers artifact bytes under `id` — the same path-like key `read` is
/// later called with (e.g. `data/azero/chess.bin`).
pub fn put(id: &str, bytes: Vec<u8>) {
    STORE.with(|s| s.borrow_mut().insert(id.to_string(), bytes));
}

/// Bytes for `id`: the in-memory store first, then (native only) the
/// filesystem.
pub fn read(id: &str) -> Result<Vec<u8>, String> {
    if let Some(b) = STORE.with(|s| s.borrow().get(id).cloned()) {
        return Ok(b);
    }
    #[cfg(not(target_arch = "wasm32"))]
    return std::fs::read(id).map_err(|e| format!("failed to read '{id}': {e}"));
    #[cfg(target_arch = "wasm32")]
    Err(format!(
        "artifact '{id}' not loaded — the host must call load_artifact first"
    ))
}
