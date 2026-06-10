//! The lab as a library: the registry of games and bots, type-erased matches,
//! and the statistical comparison driver. Two frontends consume it — the
//! terminal client binary (`main.rs`) and the browser engine (`web/engine`,
//! compiled to wasm). See ARCHITECTURE.md and WEB.md.

pub mod artifacts;
pub mod compare;
pub mod registry;
pub mod runner;
