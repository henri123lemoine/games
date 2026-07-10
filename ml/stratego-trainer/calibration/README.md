# Calibration bridge: our checkpoints vs "Demon of Ignorance" (DoI)

Plays a trained `stratego_trainer` checkpoint against the classic open-source
Java Stratego engine [braathwaate/stratego](https://github.com/braathwaate/stratego)
("Demon of Ignorance", DoI) — an independent, non-neural (alpha-beta) reference
opponent, useful as an external sanity check that isn't just "beat our own
past checkpoints."

## TL;DR

```bash
cd ml/stratego-trainer
# one-time build (needs a JDK; verified against OpenJDK 21):
mkdir -p calibration/vendor/stratego/bin
javac -d calibration/vendor/stratego/bin -encoding UTF-8 \
  $(find calibration/vendor/stratego/src -name '*.java' | grep -v /editor/)
mkdir -p calibration/vendor/stratego/bin/com/cjmalloy/stratego/resource
cp -r calibration/vendor/stratego/src/com/cjmalloy/stratego/resource/* \
      calibration/vendor/stratego/bin/com/cjmalloy/stratego/resource/
cp calibration/vendor/stratego/src/com/cjmalloy/stratego/resource/ai.cfg \
   calibration/vendor/stratego/

# run
.venv/bin/python calibration/eval_vs_doi.py --ckpt <path.safetensors> --games 20
```

`bin/` and the copied `ai.cfg` are build output (`javac`'s target, gitignored
by convention elsewhere in this repo) — not committed; the two `cp`/`mkdir`
steps above are required every time you rebuild from a clean checkout. See
"Why the resource copy" below for why the second `ai.cfg` copy is necessary at
all (it works around a real bug in DoI, not a packaging choice of ours).

## Requirements

- A JDK. Verified against `openjdk version "21.0.1"`. No Maven/Gradle build
  file ships with the repo — it's an old Eclipse project (`.project`/
  `.classpath`); plain `javac` is the only build path, and it works (one
  package, `com.cjmalloy.stratego.player.editor`, fails to compile against a
  modern JDK — an old GUI board editor unrelated to the `-t` test hook we use;
  excluded above with `grep -v /editor/`).
- Nothing else. `eval_vs_doi.py` only touches this repo's existing
  `stratego_sim`/`stratego_nets`/`stratego_trainer` Python stack plus the
  stdlib + numpy.

## The interface: DoI's `-t` testing hook

DoI's README documents an "AI Regression Testing" mode:

> Demon Of Ignorance supports the interface protocol defined in [Stratego AI
> Evaluator](https://github.com/braathwaate/strategoevaluator)... To use
> Demon Of Ignorance with the Stratego AI Evaluator, use the `-t` option.

`strategoevaluator` is a 2012 programming-competition referee (`manager/*.cpp`)
that spawns an AI as a subprocess and talks to it over stdin/stdout in a
small text protocol (`ai_controller.cpp`, `controller.cpp`, `game.cpp`,
`stratego.cpp::Board::Print`/`MovePiece`). **We do not build or run that C++
manager.** Instead `calibration/eval_vs_doi.py` is a from-scratch Python
implementation of the *manager side* of that exact protocol (reverse-engineered
from the vendored `strategoevaluator` source, empirically verified against a
live DoI subprocess — see "Validation" below), because the manager is Linux/
SDL-flavored C++ with a Makefile that doesn't obviously cross to macOS, while
reimplementing its ~150 lines of wire-protocol logic in Python is small,
inspectable, and lets our own `stratego_sim` (not DoI, not a second copy of
the manager's own rules engine) stay the single source of truth for legality
and termination.

This is architecture **(b)** from the brief: a thin Python referee that mirrors
every move into *both* engines and asserts they agree, not architecture (a)
(hosting the whole game inside DoI's own process). Concretely, per game:

1. `stratego_sim.BatchSim(num_envs=1, ...)` is authoritative for legality,
   battle resolution's board-state bookkeeping, and termination on **both**
   seats.
2. DoI runs as a subprocess (`java -cp bin com.cjmalloy.stratego.player.StrategoDriver -t -l<level>`)
   on one seat; our net drives the other seat directly through `stratego_sim`
   (no textual protocol needed for our own side).
3. Every ply, the mover's move is decoded to absolute board coordinates and
   checked against `stratego_sim`'s own `move_legal` mask **before** it is
   applied — if DoI proposes a move our sim's legal mask forbids (or vice
   versa), the referee raises loudly (`assert_legal_or_die`) instead of
   silently diverging. This never fired across validation, but it is a real
   gate, not a decorative one — see the two-square/chase caveat below for
   exactly the kind of divergence it exists to catch.
4. The chosen move is forced into `stratego_sim` via a one-hot logit row (so
   the sim's board stays byte-identical to what actually happened) and
   mirrored into a plain Python board (`Referee.board`) the referee maintains
   itself, so it can compute DoI's required battle-outcome tokens
   (`KILLS`/`DIES`/`BOTHDIE ` + attacker/defender rank chars) without any
   further engine introspection — the referee placed every piece on both
   sides itself, so it always has full information.

### The wire protocol (reconstructed from `strategoevaluator/manager/*.cpp`)

- **Setup**: send `"<RED|BLUE> <opponent-name> 10 10"`; read exactly 4 lines
  back, each 10 rank-token characters (`Piece::tokens` = `.*Fs9876543 21B?`
  for nothing/lake/flag/spy/scout down to marshal/bomb/unknown). Row `y`
  (0..4), column `x` (0..10) of that reply places at absolute cell
  `(y_start + y) * 10 + x`, `y_start = 0` for red / `6` for blue — a direct
  row-major placement, **no reflection for either colour**
  (`Controller::Setup`, `game.cpp`).
- **Move query** (only on DoI's own turn): write 10 dummy board lines (any 10
  chars; DoI's `-t` wrapper (`AITest.java`) discards them unparsed — verified
  empirically, not assumed), then read one line: `"X Y DIRECTION [MULT]"` in
  absolute coordinates, `UP`/`DOWN`/`LEFT`/`RIGHT`, optional slide length for
  scouts.
- **Move relay**: after *every* move (DoI's own or the opponent's), send one
  line back to DoI: the same `"X Y DIRECTION [MULT]"` plus a result suffix —
  `" OK"`, `" VICTORY_FLAG"`, `" KILLS <atk> <def>"`, `" DIES <atk> <def>"`, or
  `" BOTHDIE <atk> <def>"` (`Controller::MakeMove`, `game.cpp`'s
  `red->Message(buffer); blue->Message(buffer)` — sent to **both** sides after
  every move, including back to the mover itself as its own move confirmation;
  this is why the referee sends exactly one relay line per ply regardless of
  who moved, not one line per side).
- Rank/strength resolution (`Board::MovePiece`) is the same classic Stratego
  battle table as `games/stratego/src/rules.rs::defender_wins`/`is_tie`,
  including the spy-beats-marshal and bomb-needs-a-miner special cases, with
  the piece orderings lining up 1:1 between the two codebases' enums. This
  part is *not* a caveat — it agrees.

### Why the resource copy

`AI.getBoardSetup` (`AI.java`) tries `Class.class.getResourceAsStream(...)` to
load `resource/ai.cfg` (the AI's canned setup library) off the classpath. On a
modern JDK this silently returns `null` because `Class.class` resolves against
`java.lang.Class`'s own (bootstrap) classloader, not the application
classpath — a real bug in 15-year-old code, not anything specific to this
bridge. `AI.java` falls back to a plain `new File("ai.cfg")` in the process's
**current working directory** if it exists, so the build steps above copy
`ai.cfg` there and `eval_vs_doi.py` launches the `java` subprocess with
`cwd=calibration/vendor/stratego`.

## Rules caveats — read before trusting any win-rate from this bridge

**These are real, verified differences between our sim's rules and DoI's.
A win-rate comparison via this bridge is a directional signal, not a
certified head-to-head under identical rules.**

1. **Continuous-chase ("more squares") rule mismatch — likely, unverified
   both ways.** Our sim (`games/stratego/src/chase.rs`) implements a
   from-the-reference continuous-chase restriction: once a chase counter
   reaches 2, a player cannot reproduce an earlier threatening board state
   except by reverting their own literal last move. DoI's own README only
   documents a **Two Squares** rule toggle (`-1`/`-2` command-line flags,
   `Settings.twoSquares`); it never mentions a chase/more-squares rule
   anywhere in its README, settings, or command-line flags. The straightforward
   reading is that DoI does not implement this rule at all, which means our
   sim can and will forbid some moves DoI considers perfectly legal. This is
   exactly what `assert_legal_or_die` exists to catch — it never fired in the
   6-game validation run below, but that is a small sample against a
   move-cap-truncated random policy, not a proof the two engines never
   disagree here. **If it ever fires in a real run, that is this caveat
   manifesting, not a bug in the bridge** — treat any resulting exploitable
   pattern as unrepresentative of a real Stratego game.
2. **Two-square rule variant is plausibly not bit-identical.** Our sim's
   `twosquare.rs` implements a specific 4-cell zig-zag detector with a
   "precluding direction" relaxation for scouts, ported from a specific
   reference implementation. DoI's own two-square rule (togglable via `-1`/
   `-2`) is its own independent implementation of what is nonetheless the same
   *named* classic rule — we did not diff the two algorithms move-for-move.
   Disagreements here would also surface through `assert_legal_or_die`.
3. **Flag-handedness is a hard-coded artifact of our sim, not a Stratego
   rule, and it can reject DoI's real setups.** `games/stratego/src/game.rs`
   always constructs deployment with `force_handedness=true`
   (`arrangement.rs::flag_allowed_on_right`: the flag may only land in the
   right half, column-major slot `%10 >= 5`, of the player's own home rows).
   This is a self-play symmetry-breaking convention from training data
   generation, **not a real Stratego rule**, and DoI's setups are free of it —
   DoI placed its flag on the "illegal" (for our sim) half in multiple
   validation games. **Mitigation**: when this happens, the referee mirrors
   DoI's entire reported setup left-right (`mirror_columns`) before seeding
   our sim, and thereafter mirrors every coordinate (`x -> 9-x`, `LEFT`↔
   `RIGHT`) exchanged with the DoI subprocess for the rest of that game. Since
   the board's lakes are bilaterally symmetric, this is a sound, fully
   rules-transparent transform (DoI is still playing its own real setup, up
   to a left-right relabelling it never observes) — but it means roughly half
   of DoI-seat games run "mirrored," which is worth knowing if you ever go
   digging through logs.
4. **No SURRENDER / illegal-move leniency handling.** DoI's protocol supports
   a `SURRENDER` response and a "let a human retry" leniency path
   (`Controller::MakeMove`); this bridge implements neither (DoI never sends
   `SURRENDER` in `-t` mode in practice, and our sim only ever forces
   sim-legal actions for the hero, so leniency is moot for us) — a `SURRENDER`
   or unparseable line from DoI raises loudly rather than being handled
   gracefully.
5. **DoI's own move-generation timing/level is not tuned for strength here.**
   `--doi-level` maps straight to DoI's `-l<N>` search-depth flag; the default
   (`0`, fastest) is chosen for validation speed, not maximum DoI strength.
   Raise it for any comparison meant to say something about relative playing
   strength.

None of the above affects **setup deployment counts** — DoI's classic
40-piece supply (`Piece::maxUnits` in `stratego.cpp`: 1 flag, 1 spy, 8 scouts,
5 miners, 4 sergeants, 4 lieutenants, 4 captains, 3 majors, 2 colonels,
1 general, 1 marshal, 6 bombs) is exactly `stratego_nets.spec.CLASSIC_PIECE_COUNTS`.

## Running it

```bash
.venv/bin/python calibration/eval_vs_doi.py --ckpt <path.safetensors> --games N [--seed S]
```

Loads the checkpoint the same way `stratego_trainer/eval_ckpt.py` does
(`rundir.load_checkpoint(..., prefer_ema=True)`, net shape read back from the
checkpoint's own `net_size` metadata) at `--temperature 0.25` by default.
Alternates which colour DoI plays across games (`doi_seat = i % 2`). Prints
one JSON line at the end:

```
{"ws_doi": <win share>, "games": N, "wins": w, "draws": d, "losses": l}
```

Flags: `--move-cap` (default 4000, matches the reference-parity cap in
`rules.rs::MAX_NUM_MOVES`), `--doi-level` (DoI's `-l` search depth, default
`0`), `--timeout` (per-line read timeout on the DoI subprocess, default 30s).

## Validation

`calibration/make_random_ckpt.py` writes a randomly-initialized "default"-size
checkpoint to `calibration/random_ckpt.safetensors` (not under any `runs/`
directory, and not a trained network — purely a cheap way to exercise the
real `eval_ckpt.py`-style loading path). Six full games were run against it
(`--games 6 --seed 0 --move-cap 400`, `--doi-level 0`):

```
game 0: hero_seat=1 doi_seat=0 mirror=True  -> draw
game 1: hero_seat=0 doi_seat=1 mirror=False -> draw
game 2: hero_seat=1 doi_seat=0 mirror=True  -> draw
game 3: hero_seat=0 doi_seat=1 mirror=False -> loss
game 4: hero_seat=1 doi_seat=0 mirror=True  -> draw
game 5: hero_seat=0 doi_seat=1 mirror=False -> loss
{"ws_doi": 0.3333333333333333, "games": 6, "wins": 0, "draws": 4, "losses": 2}
```

All six games reached a terminal state cleanly through the full protocol
(setup, mirrored and unmirrored, on both seats; the move-relay loop for
dozens of plies each) with no subprocess hangs, no dropped/duplicated
protocol lines, and no `assert_legal_or_die` firings. A random network losing
or drawing against even a fast/shallow alpha-beta engine, and never winning,
is the expected direction — this validates the bridge's plumbing, not the
net's strength.
