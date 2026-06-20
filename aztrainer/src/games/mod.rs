//! The per-game plug-ins: each game supplies its `NetConfig`, its replay sample
//! codec, its self-play reward shaping, and its eval/serving commands, then
//! drives the shared `aztrainer` core. The generic algorithm lives in the
//! crate root; only game knowledge lives here.

pub mod snake;
