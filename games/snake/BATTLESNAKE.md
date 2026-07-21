# Canonical Battlesnake

The competitive `snake` registry entry is the official 11×11 Battlesnake
ruleset, not a turn-ordered approximation. Two to four active players choose
from the same public pre-state; the engine resolves the complete joint action,
then runs the official health, hazard, feeding/growth, collision, royale, and
food-placement order.

## Search

`battlesnake::search` implements the public Shapeshifter family of ideas:

- iterative best-node search with null-window alpha-beta probes;
- exact full Cartesian replies in duels, plus MCS and BRS+ opponent models for
  multiplayer branching;
- compact `u128` board masks, transpositions, history ordering, quiescence, and
  separated-endgame detection;
- a phase-interpolated evaluator over territory, checkerboard capacity, tail
  control, food, health, length, hazards, and mobility.

The minimizing response layer is internal to one hero's search. It never lets
an arena opponent condition its actual move on the hero's current choice.

## Neural experiments

`ml/aztrainer --bin snake` trains a shared policy/value network from all active
perspectives at each joint state. Its controlled methods are:

- `logit`: general-sum quantal-response backups over the complete joint table;
- `maximin`: entropic duel maximin backups;
- `policy`: policy-only backup ablation.

The two equilibrium methods batch root and one-ply leaf evaluations; the
policy-only negative control deliberately omits the leaf table. Training data
uses all eight dihedral transforms, and evaluation uses paired seat swaps or
rotated fields. Sequential PUCT is intentionally absent: it would reveal a
current move that is hidden in the real game.

## Research verdict

There is no single peer-reviewed, canonical-11×11 benchmark that establishes a
universal Battlesnake SOTA. Public leaderboards are live, opponents are usually
closed-source, modes differ, and stochastic multiplayer tournament results have
large variance. The strongest reproducible evidence separates into two tracks:

1. **Competition strength:** Shapeshifter is the best documented public engine.
   It won the 2022 Summer Elite tournament and, when checked on 2026-07-21, was
   first on the live Standard leaderboard and in the top three in both Duels
   and Royale. These ranks were sampled during the daily run and can move as
   new games finish.
   Its author reports that BNS beat MTD(f) decisively in self-play and that MCS
   reached roughly 60% win rate in both four- and eight-player public games.
2. **Learning research:** Albatross is the strongest published simultaneous-game
   learning result found. Fixed-depth Logit-equilibrium backup beat DUCT, EXP3,
   regret matching, SM-OOS, and Nash backup in its stochastic two- and
   four-player experiments. Its opponent-conditioned response model then beat
   its AlphaZero baseline and fixed-search opponents in aggregate. Those were
   7×7 research variants, not the canonical live 11×11 leaderboard, and used
   48–96 hours on 2–3 RTX 3090 GPUs per run across five seeds.

Consequently, “neural” is not synonymous with “strongest Battlesnake.” The
highest-confidence engineering path is a fast Shapeshifter-style search engine;
the highest-upside learning path is a joint-action policy/value model trained
with a stable equilibrium target and evaluated against that engine.

## Local controlled results (2026-07-21)

Three two-player neural methods received equal 30-minute wall-clock trials on
the same 4×64 residual architecture. Maximin beat Logit 39-15-10 over 64
seat-swapped games (Logit's score 0.273, 95% CI 0.179–0.393). Both equilibrium
methods beat the policy-only ablation 64-0-0; the policy loss of that ablation
stayed near `ln(4)`, as expected from uniform self-distillation without a joint
payoff table. The winning maximin checkpoint also beat an early checkpoint
47-17-0, but lost every game to even the 1 ms depth-4 BNS anchor. A subsequent
10-minute safety/solver continuation did not improve it: the final checkpoint
scored 0.492 (14-35-15, CI 0.374–0.612) against its parent, and the mid-run
low-loss checkpoint scored 0.445 (18-21-25, CI 0.330–0.567).

For four-player Logit training, two equal-sample probes compared stochastic
action sampling for 20 turns with sampling for the full episode. The 20-turn
variant won the two held-out balanced 2-vs-2 slices; combined from the
full-episode variant's perspective the result was 6-2-10, score 0.389 (95% CI
0.203–0.614). On identical weights, increasing Logit rationality from 8 to 16
then won 19-1-4 across two independent balanced slices, score 0.8125 (95% CI
0.618–0.921). Rationality 16 and 20 sampling turns are therefore the promoted
long-run configuration. The selected short checkpoint went 8-0 against a random
field but 0-8 against both 1 ms depth-4 and 5 ms depth-8 BNS fields. It learned
coherent play, but not search-level strength. The first immutable checkpoint
from the eight-hour continuation (about ten minutes) scored only 0.375 against
its seed (4-1-7, CI 0.165–0.646), so the seed remains the incumbent pending
later held-out gates; the long run is an experiment, not an automatic promotion.

The fixed-position 200 ms search benchmark reaches depth 10 at 2.74M nodes/s
in a full-reply duel, depth 8 at 1.75M nodes/s with four-player MCS, and depth 5
at 1.98M nodes/s with BRS+. Exact four-player replies reach only depth 1. In
64-game field probes, MCS invaded a BRS+ field at 0.266 win share while BRS+
invaded an MCS field at 0.219 (fair is 0.250); both tests remained statistically
open, but the direction plus the depth advantage supports MCS as the default.
A fixed-depth SPSA evaluator candidate finished essentially tied with the public
baseline (10-46-8, score 0.516), so the tuned weights were rejected rather than
promoted from noise.

### Why ordinary AlphaZero is wrong here

Every snake chooses from the same immutable state. A sequentialized tree that
lets player B see player A's current action solves a different game. Even
decoupled UCT can cycle and lacks general Nash-convergence guarantees. The
credible alternatives preserve the hidden concurrent choice:

- enumerate a complete joint-action normal-form game at a shallow fixed depth,
  then solve a Logit/Nash/maximin equilibrium;
- use SM-MCTS variants such as stochastic EXP3 or regret matching;
- use paranoid alpha-beta internally as an opponent model, without exposing the
  hero's hypothetical action to the actual arena opponent.

Albatross found the first option—fixed depth plus Logit equilibrium—to be the
best neural training target in stochastic Battlesnake. Entropy smoothing is
important: small value-network errors can make a Nash target jump discontinuously
between pure actions, while a Logit target changes smoothly.

### What the strongest published training recipe actually used

Albatross is much more than a policy network with ordinary self-play. Its proxy
model is trained across a distribution of rationality temperatures; a separate
response model is conditioned on estimated opponent temperatures and learns a
smooth best response. At every internal search node it constructs the full
normal-form joint-action game, solves a Logit equilibrium, and backs the expected
utilities toward the root. The paper's implementation used MobileNetV3, a replay
buffer of two million examples, batch size 2,000, cosine learning-rate decay,
and 150 iterations of stochastic fictitious play per equilibrium solve. Search
depth was three in stochastic two-player Battlesnake and one in four-player
Battlesnake. The respective runs used 3 RTX 3090s for 96 hours and 2 RTX 3090s
for 48 hours, with 14 CPU cores feeding each GPU, repeated across five seeds.

The authors have since split their simulator into `hisss`, a maintained C++
engine with Python bindings, all main ruleset modes, perspective observations,
eight symmetry transforms, and Nash/Logit solver utilities. That makes it the
best public reference stack found for reproducing or extending Albatross-style
training, although the official Go rules engine remains the authority for exact
live semantics.

That scale matters when interpreting a short local run. A 30-minute CPU probe
can falsify a bad backup rule and validate the data path, but it cannot reproduce
the paper's asymptotic strength. The local `logit` lane implements the core
full-joint equilibrium target; opponent-temperature conditioning and the paper's
depth-three duel search are follow-on complexity only worth paying for if that
target first wins the equal-budget gate.

## Optimization inventory

| Lever | Why it matters | Main cost or caveat |
|---|---|---|
| `u128` bitboards | Cheap state copies and shift-based flood fill on 121 cells | Mode/rule correctness is less forgiving than a grid implementation |
| BNS + null windows | Finds the best root move without paying for its exact minimax value | Depends heavily on good ordering and a reliable bounded evaluator |
| Iterative deepening | Always retains a complete result before the deadline and improves ordering | Re-search overhead; a transposition table is essential |
| MCS | Covers every individual opponent move in at most four joint replies | More joint-action blind spots than BRS+ or full Cartesian search |
| BRS+ | Varies one opponent at a time from strong default moves | Branching still grows with player count; assumes locality |
| Full replies in duels | Exact simultaneous maximin tree at only four replies | Too expensive in four-player positions |
| Transposition table | Reuses the many move-order transpositions and null-window results | Memory, replacement policy, and correct bound tagging |
| History/tactical ordering | Produces earlier alpha-beta cutoffs | Stale history can misorder unusual modes; it must never prune |
| Root parallel BNS | Tests independent root moves concurrently at each threshold | Shared history/TT updates need care, and shallow roots may under-utilize cores |
| Quiescence | Extends unstable head, food, and forced-mobility tactics | Can explode if the stability test is too broad |
| Tactical MCTS fallback | Shapeshifter falls back when a shallow forced loss leaves alpha-beta little useful discrimination | Adds a second engine and is only worthwhile when the trigger is precise |
| Dynamic flood fill | Models territorial reach while tails vacate over time | Eating can pin a tail, so naive static flood fill is optimistic |
| Checkerboard capacity | Detects whether a claimed region can actually fit a growing path | It is a capacity heuristic, not a proof of a Hamiltonian path |
| Separated endgame solver | Converts partitioned duels into near-terminal bounds | Must be conservative around food, tail access, and wrapped boards |
| Phase/mode-specific weights | Food/health dominate early; space/tails dominate late | Requires paired, field-based tuning rather than single lucky games |
| Genetic/SPSA weight tuning | Optimizes many correlated evaluator terms with paired common-seed games | Noisy objectives need large match budgets and held-out gates |
| Batched inference | Makes neural leaf evaluation feasible | Batch latency competes with the 500 ms request deadline |
| Eight board symmetries | Gives an eightfold data multiplier and reduces orientation bias | Direction labels and player perspective must transform exactly |
| Head-centered padded observations | Hisss's default 21×21 view gives a CNN translation-consistent ego frame while retaining the whole 11×11 board | Nearly quadruples spatial inference work versus an 11×11 absolute-board tensor |
| Replay + checkpoint population | Reduces catastrophic forgetting and self-play cycling | More storage and a more complex sampling schedule |
| Opponent conditioning | Exploits systematically weak or bounded-rational opponents | Needs within-game evidence and can overfit opponent classes |
| Regret-matching joint MCTS | Gives a principled mixed policy-improvement operator for two-player simultaneous games | Considerably dearer than a fixed-depth 4×4 payoff table; current evidence is deterministic and duel-only |
| Distributional value targets | Simultaneous AlphaZero reports more stable search values than scalar regression | More output bins and no published Battlesnake ablation yet |
| Common random numbers | Applying the same chance stream to every joint-action leaf reduces stochastic payoff-table variance | Correlates leaf estimates by design and must never depend on the action index |
| Hard safety masks | Prevents obvious wall/body suicides and accelerates early RL | Only provably fatal moves should be removed; heuristic masks hide tactics |

## Training evidence and dead ends

- The 2020 AWS Battlesnake Challenge trained PPO agents with action masking,
  reward shaping, and post-policy safety overrides. Human guidance improved the
  learners, with reward manipulation performing best online, but the work was a
  framework/baseline result rather than evidence of overall SOTA.
- A separate 2020 Asymptotic Labs PPO system briefly topped the then-global
  arena and later won a competition. It used terminal-only rewards, 208 parallel
  environments, population replay against older checkpoints, roughly 524 million
  turns over four days on one Titan RTX, and an alpha-beta safety/win override.
  The authors reported policy inference below 15 ms and performance close to,
  but not beyond, their 500 ms alpha-beta agent. This is strong historical
  evidence for hybrid deployment, not a current canonical SOTA comparison.
- AlphaSnake Zero and the 2022/2025 UVic MAS-MCTS work contain useful systems
  ideas: parallel games, batched accelerator inference, state caches, relative
  coordinates, joint-action MCTS, and policy distillation. The 2022 experiment
  used 2,000 simulations to bootstrap roughly 270,000 examples and established
  improvement over earlier network iterations, but did not beat a strong search
  baseline. The later reported arena ranks and comparisons do not approach the
  top documented search engines.
- Simultaneous AlphaZero (2025) is a newer general method for two-player
  zero-sum Markov games. It treats each tree node as a matrix game, uses a
  regret-optimal bandit-feedback solver for joint-action MCTS, trains separate
  player policies plus a distributional value head, and evaluates
  exploitability against exact best responses. Its published experiments are
  deterministic pursuit/evasion and satellite-custody tasks—not Battlesnake;
  the paper explicitly leaves stochastic transitions and more than two players
  as future work. It is therefore a promising duel-search direction, not
  stronger Battlesnake evidence than Albatross's stochastic 2/4-player tests.
- A 2026 ConvLSTM/PPO repository describes an intended 500-million-step
  curriculum and claims scaling its tournament sprint to eight A100s, but
  publishes no controlled match table or checkpoint evidence. Architecture and
  compute claims alone are not a strength result.
- Pure policy learning is a valuable ablation, not the default winner. It removes
  the expensive joint backup but also removes the main mechanism that teaches
  mixed simultaneous tactics.

## Evaluation standard

Training loss and self-play survival time are diagnostics only. Selection uses:

- common random seeds and paired seat swaps in duels;
- hero rotation through every seat in multiplayer fields;
- independent, reproducibly seeded sampling from neural mixed strategies
  (never argmax-collapse an equilibrium policy);
- direct cross-play between candidate checkpoints;
- fixed random, shallow-BNS, and deeper-BNS anchors;
- win share with draws counted as half in duels, plus raw W-D-L counts;
- checkpoint slopes and ablations, not one terminal snapshot.

The 30-minute trials in this repository are equal-wall-clock CPU architecture
probes. They are intentionally not presented as a reproduction of Albatross's
multi-GPU training budget.

## Primary sources

- [Official rules and simultaneous resolution](https://docs.battlesnake.com/rules)
- [Production rules source](https://github.com/BattlesnakeOfficial/rules)
- [Live Battlesnake leaderboards](https://play.battlesnake.com/leaderboards)
- [2022 Summer Elite result](https://docs.battlesnake.com/blog/2022/08/08/summer-league-results-are-in)
- [Shapeshifter technical postmortem](https://notpeerreviewed.com/blog/battlesnake/)
- [Shapeshifter source](https://github.com/JonathanArns/shapeshifter)
- [Fuzzified/Best-Node Search thesis](https://dspace.lu.lv/dspace/bitstream/handle/7/2266/LUR-770-Datorzinatne.pdf?sequence=1)
- [MTD(f) paper](https://arxiv.org/abs/1404.1511)
- [Best Reply Search paper](https://doi.org/10.1109/TCIAIG.2011.2107323)
- [BRS+ paper](https://doi.org/10.1007/978-3-319-09165-5_11)
- [Albatross paper](https://arxiv.org/abs/2402.03136) and [source/models](https://github.com/ymahlau/albatross)
- [Hisss simulator and equilibrium utilities](https://github.com/ymahlau/hisss)
- [Battlesnake Challenge paper](https://arxiv.org/abs/2007.10504) and [AWS source](https://github.com/awslabs/sagemaker-battlesnake-ai)
- [Asymptotic Labs PPO and alpha-beta postmortem](https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428)
- [2022 multi-agent simultaneous MCTS report](https://uvicai.ca/assets/images/MCTS-and-RL-for-a-Four-Player-Simultaneous-Move-Game.pdf)
- [Multi-agent simultaneous MCTS report](https://web.uvic.ca/~nmehta/data_mining_spring2025/alphasnake_zero_final_report.pdf)
- [AlphaSnake Zero source](https://github.com/Fool-Yang/AlphaSnake-Zero)
- [Simultaneous AlphaZero](https://arxiv.org/abs/2512.12486)
- [2026 ConvLSTM/PPO repository](https://github.com/giovannettif/convLSTM-Battlesnake)
