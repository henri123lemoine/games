# Calibration bridge: our checkpoints vs "Demon of Ignorance" (DoI)

Plays a trained `stratego_trainer` checkpoint against the classic open-source
Java Stratego engine [braathwaate/stratego](https://github.com/braathwaate/stratego)
("Demon of Ignorance", DoI) — an independent, non-neural (alpha-beta) reference
opponent, useful as an external sanity check that isn't just "beat our own
past checkpoints."

## TL;DR

```bash
cd ml/stratego-trainer
# one-time: fetch the pinned third-party sources (gitignored), then build
# (needs a JDK; verified against OpenJDK 21):
calibration/fetch_vendor.sh
calibration/build_doi.sh

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

## Elo ladder pipeline (doi_bridge.py, doi_vs_doi.py, run_ladder.sh, fit_elo.py, plot_elo.py)

`eval_vs_doi.py`'s lockstep referee protocol (subprocess management, coordinate
mirroring, outcome-token encoding, `assert_legal_or_die`) is factored out into
`doi_bridge.py` (`DoIProcess` + `BaseReferee` + the pure coordinate/board
helpers). `eval_vs_doi.py`'s `Referee` and `doi_vs_doi.py`'s
`DoIVsDoIReferee` both subclass `BaseReferee`; the only real difference
between them is which side(s) are driven by net inference vs a second DoI
query.

**`doi_vs_doi.py`** plays two DoI subprocesses (no net) against each other at
independent `-l` levels, refereed the same way, for measuring DoI's own
self-play strength curve across levels:

```bash
.venv/bin/python calibration/doi_vs_doi.py --level-a 0 --level-b 4 --games 24 --seed 0
# {"level_a": 0, "level_b": 4, "wins_a": .., "draws": .., "wins_b": ..}
```

**`run_ladder.sh`** orchestrates the full measurement: a DoI self-ladder
(adjacent pairs 0v4, 4v8, 8v12, 24 games each) plus every `--ckpts` checkpoint
vs DoI levels 0/4/8 (and additionally level 12 for the later half of the
`--ckpts` list, `--extra-level-frac` configurable), 20 games each. Every
match result is appended as one JSON line to `calibration/ladder_results.jsonl`.
The script is idempotent — before playing a match it checks
`ladder_results.jsonl` for an existing line with the same *identifying*
fields (which two DoI levels, or which checkpoint vs which DoI level) and
skips it if found, regardless of the recorded outcome, so a killed/resumed
run never re-plays or duplicates a match.

**`fit_elo.py`** fits Bradley-Terry strengths over `ladder_results.jsonl` by
Zermelo/minorization-maximization iteration (no external optimizer
dependency), treating a draw as half a win for each side. DoI level 0 is
fixed as the zero point of the resulting Elo scale (`elo = 400 * log10(p)`,
the standard Elo <-> Bradley-Terry correspondence). Standard errors are
computed by bootstrap resampling: each match line's win/loss games are
resampled (binomial resample around the observed win rate, same total game
count) and the whole ladder refit, `--bootstrap` (default 200) times; each
entity's stderr is the sample standard deviation of its rating across
resamples. Output `calibration/elo_estimates.json`:
`{"<entity>": {"rating": .., "stderr": .., "n_games": ..}}` for every DoI
level and checkpoint seen in the data (`doi_l<N>` for DoI levels, the raw
checkpoint path otherwise).

**`plot_elo.py`** plots `calibration/elo_progress.png`: fitted Elo (with
bootstrap-stderr error bars) vs cumulative training iteration for every
checkpoint in `elo_estimates.json`, connected by a line in x-order; DoI rungs
as labeled horizontal dotted lines; and coarse shaded reference bands for
human/engine skill tiers. x-axis mapping (see `ckpt_x` in `plot_elo.py`):
`runs/marathon1c/ckpt_100.safetensors` is the x=0 anchor,
`runs/marathon_r1/ckpt_<N>.safetensors` plots at x=N, and
`runs/marathon_r2/ckpt_<N>.safetensors` plots at x=N+2200 (marathon_r2 is a
continuation run that starts its own iteration counter back near 0, so it is
offset to stay cumulative with marathon_r1's ~2200 iterations on this shared
axis). Any checkpoint path outside that known set still plots (at its own raw
`ckpt_<N>` iteration, unmapped, with a stderr warning) rather than crashing —
this matters for sparse smoke-test data with only one or two checkpoints.

### Elo-to-Gravon caveat — read before trusting the reference bands on the plot

There is **no measured mapping** from this pipeline's internal,
DoI-level-0-anchored Bradley-Terry Elo scale to Gravon's (the reference
Stratego server) real-money-tournament rating scale — these are two
independent rating pools with no shared games between them. `plot_elo.py`
draws its human-skill-tier reference bands on the same axis as our fitted
checkpoint Elo only by making one explicit, coarse assumption
(`--gravon-anchor-doi-level 12 --gravon-anchor-rating 1790`): **we assume our
fitted DoI-level-12 Elo corresponds to approximately Gravon rating 1790**,
i.e. shifting the whole internal Elo scale by a constant offset so DoI
level 12 lands at 1790 Gravon. That specific number comes from DeepNash
(Science 2022, McAleer et al.), which reported roughly a 1780-1800 Gravon
rating and a 97%+ win rate against the DoI-family classic bots specifically —
so "DoI at some non-trivial search depth sits somewhat below DeepNash, in the
low-to-mid 1700s-1800s Gravon range" is a defensible order-of-magnitude
placement, not DoI level 12 exactly equalling DeepNash's rating. The band
boundaries themselves (beginner human 1000-1200, club amateur 1200-1500,
strong human 1500-1700, DeepNash/Ataraxos tier 1750-1850, best human
1850-2000) are read off the shape of Gravon's public rating distribution
from general knowledge, not from a cited table — treat every Gravon number on
this plot as **illustrative and approximate, not a certified conversion**.
A different, equally defensible anchor choice (e.g. anchoring DoI level 0
instead of level 12, or picking a different assumed DoI-level-to-Gravon
point) would shift every band and every checkpoint point on the plot by a
constant amount without changing the shape of the training-progress curve —
the curve's *shape* (checkpoints climbing relative to the DoI rungs) is the
trustworthy signal from this pipeline; the absolute Gravon axis position is
not.

## Smoke-test-only artifacts

`calibration/ladder_results.jsonl`, `calibration/elo_estimates.json`, and
`calibration/elo_progress.png` are **not committed** — they are cheap,
sparse, low-level (DoI level 0-8, 2-4 games/match) smoke-test output produced
while validating this pipeline, not a real measurement, and the real overnight
ladder run will overwrite them with actual results. See the invocation below
for the real run.

```bash
# real full ladder (session lead runs this overnight, NOT part of this smoke test):
cd ml/stratego-trainer
bash calibration/run_ladder.sh --ckpts \
runs/marathon1c/ckpt_100.safetensors,runs/marathon_r1/ckpt_100.safetensors,runs/marathon_r1/ckpt_200.safetensors,runs/marathon_r1/ckpt_300.safetensors,runs/marathon_r1/ckpt_400.safetensors,runs/marathon_r1/ckpt_500.safetensors,runs/marathon_r1/ckpt_600.safetensors,runs/marathon_r1/ckpt_700.safetensors,runs/marathon_r1/ckpt_800.safetensors,runs/marathon_r1/ckpt_900.safetensors,runs/marathon_r1/ckpt_1000.safetensors,runs/marathon_r1/ckpt_1100.safetensors,runs/marathon_r1/ckpt_1200.safetensors,runs/marathon_r1/ckpt_1300.safetensors,runs/marathon_r1/ckpt_1400.safetensors,runs/marathon_r1/ckpt_1500.safetensors,runs/marathon_r1/ckpt_1600.safetensors,runs/marathon_r1/ckpt_1700.safetensors,runs/marathon_r1/ckpt_1800.safetensors,runs/marathon_r1/ckpt_1900.safetensors,runs/marathon_r1/ckpt_2000.safetensors,runs/marathon_r1/ckpt_2100.safetensors,runs/marathon_r2/ckpt_100.safetensors,runs/marathon_r2/ckpt_200.safetensors,runs/marathon_r2/ckpt_300.safetensors,runs/marathon_r2/ckpt_400.safetensors,runs/marathon_r2/ckpt_500.safetensors,runs/marathon_r2/ckpt_600.safetensors,runs/marathon_r2/ckpt_700.safetensors,runs/marathon_r2/ckpt_800.safetensors,runs/marathon_r2/ckpt_900.safetensors,runs/marathon_r2/ckpt_1000.safetensors,runs/marathon_r2/ckpt_1100.safetensors,runs/marathon_r2/ckpt_1200.safetensors,runs/marathon_r2/ckpt_1300.safetensors,runs/marathon_r2/ckpt_1400.safetensors \
--games-self 24 --games-ckpt 20

.venv/bin/python calibration/fit_elo.py --bootstrap 200
.venv/bin/python calibration/plot_elo.py
```

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
