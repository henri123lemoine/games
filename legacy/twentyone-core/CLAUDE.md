# CLAUDE.md

## Rust Guidelines

- Toolchain: use **stable** with **Edition 2024**.
- Formatting: run **rustfmt** (set `style_edition = "2024"`); no manual formatting.
- Lints: run **clippy** and fix warnings; treat warnings as errors in CI when feasible.
- Layout: follow **Cargo** conventions—`src/lib.rs`, `src/main.rs`, optional `src/bin/`, `tests/`, `benches/`, `examples/`; use **workspaces** for multi-crate repos.
- Public API: keep the surface small and consistent; follow the **Rust API Guidelines**; respect SemVer and document MSRV.
- Errors: return `Result<T, E>`; use **`thiserror`** for library error types; **`anyhow`** is fine for binaries; avoid `unwrap`/`expect` outside tests/bin, though occasionally acceptable during development.
- Ownership: prefer borrowing (`&str`, slices) and avoid unnecessary clones; minimize `Arc/Mutex` to true sharing needs.
- Concurrency/async: don’t block in async contexts; prefer structured concurrency and graceful shutdown. Use async rust only when it is strictly necessary, or helps performance tremendously or when it does not add significant complexity.
- Docs: write `///` rustdoc for public items; include examples that compile as doctests.
- Testing: unit tests inline (`#[cfg(test)]`); integration tests in `tests/`; `cargo test` must pass locally and in CI.
- Features: keep defaults minimal; features should be additive and orthogonal.
- Logging: use a logging/tracing facade in libraries; avoid `println!` in library code, though here it's acceptable.
- Unsafe: isolate and document safety invariants; prefer safe abstractions first.
- CI gate: `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo test` must be clean before merge.
