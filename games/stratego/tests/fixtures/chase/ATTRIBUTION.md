# Attribution

The `.txt` game logs in this directory (`continuous_chase_games/`,
`continuous_chase_games_new/`, `strategus_games/attack_chase.txt`) are vendored
verbatim from the reference implementation this crate faithfully reimplements:

https://github.com/AtaraxosAI/stratego (MIT License, Copyright (c) 2023 Gabriele Farina)

They are real recorded games (a board string + the move sequence played),
each ending in a continuous-chase-rule violation validated directly against
the reference's own CUDA environment (`tests/test_continuous_chase_new.py`
asserts `env.current_legal_action_mask` directly, for the `_new` set). They
are the ground-truth fixtures `chase_replay.rs` replays through our own
`rules::legal_mask`/`rules::apply` to certify `chase.rs`'s port of the
production `chase_state.cu` kernel.
