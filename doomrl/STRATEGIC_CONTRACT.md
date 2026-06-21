# Strategic 1v1 Doom — OBS / ACTION / REWARD contract

Authoritative parity contract. The same OBS layout, ACTION decode, and the
per-seat raw-state float bridge MUST be mirrored across three languages; if one
drifts, train != deploy and the bot plays garbage:

- Rust trainer: `doomtrain/src/env.rs` (`observation`, `decode_action`,
  `shaped_reward`, `NUM_ACTIONS`, `OBS_DIM`), FFI struct `doomtrain/src/ffi.rs`,
  C struct `doomrl/doomrl.h` (`doomrl_player_state_t`).
- C WASM bridge: `doomrl/doomrl_web.c` (`web_player_state` float layout +
  `web_set_action` decode -> doomrl_action_t).
- Browser forward: `web/app/public/doom-ai/forward.js` (`observation`,
  `decodeAction`, the `S` state-float index map, `PLAYER_STATE_FLOATS`).

The match runs in **AltDeath** (`deathmatch = 2`): items respawn 30 s after
pickup and weapons do NOT stay in place. That is what makes item timing a skill.
Set in `doomrl/doomrl.c` (`dm_setup_match`), the WASM `web_init`, and the
`dm_driver.c` CLI (`-altdeath`).

---

## Key items (NUM_KEY_ITEMS = 3, fixed order)

`[0]=rocket_launcher (2003, MT_MISC27)`,
`[1]=megaarmor (2019, MT_MISC1)`,
`[2]=soulsphere (2013, MT_MISC12)`.

The C side records each key item's fixed map spawn position once (scanning live
mobjs + the engine respawn queue at level load), reports `available` by scanning
live mobjs of the matching MT_ type, and `respawn_secs` from the engine's
item-respawn queue (P_RespawnSpecials, 30 s timer in deathmatch==2).

---

## Per-seat raw state float bridge — `PLAYER_STATE_FLOATS = 39`

`web_player_state(seat, out)` writes these floats, in this exact order. The JS
`S` index map and `observation()` read from here; the Rust `observation()` reads
the equivalent `PlayerState` fields.

```
 0 alive
 1 x
 2 y
 3 z
 4 angle_deg
 5 momx
 6 momy
 7 health
 8 armor
 9 armortype          (0 none / 1 green / 2 blue)
10 ready_weapon       (wp_* 0..8)
11 ammo_clip
12 ammo_shell
13 ammo_cell
14 ammo_misl
15 frags
16 deaths
17 opponent_visible
18 opp_bearing_deg
19 opp_dist
20 opp_rel_vx
21 opp_rel_vy
22 opp_health
23 opp_mem_valid
24 opp_mem_ticks
25 opp_mem_last_bearing
26 opp_mem_last_dist
27..30 item0 rocket     [available, respawn_secs, bearing_deg, dist]
31..34 item1 megaarmor  [available, respawn_secs, bearing_deg, dist]
35..38 item2 soulsphere [available, respawn_secs, bearing_deg, dist]
```
**PLAYER_STATE_FLOATS = 39** (27 base + 3 items × 4).

---

## OBS vector — `OBS_DIM = 40`

index : meaning : normalization
```
 0 health                       /100
 1 armor                        /200      (blue megaarmor caps at 200)
 2 armortype == green (0/1)     raw
 3 armortype == blue  (0/1)     raw
 4 ammo clip                    /200
 5 ammo shell                   /50
 6 ammo cell                    /300
 7 ammo misl (rockets)          /50
 8 ready_weapon == shotgun(3)   0/1
 9 ready_weapon == chaingun(4)  0/1
10 ready_weapon == rocket(5)    0/1
11 angle sin
12 angle cos
13 x                            /1024 (ARENA_HALF)
14 y                            /1024
15 momx                         /16
16 momy                         /16
17 opponent_visible             raw
18 opp bearing sin
19 opp bearing cos
20 opp dist                     min(dist/512, 8)
21 opp rel vx                   /16
22 opp rel vy                   /16
23 opp health                   /100
24 opp memory valid             raw
25 opp memory ticks_since_seen  min(t/35, 20)
26 opp memory bearing sin
27 opp memory bearing cos
-- key items (4 channels each: available, respawn_norm, bearing_sin*invdist, bearing_cos*invdist) --
28 item0 rocket    available     0/1
29 item0           respawn_norm  respawn_secs/30
30 item0           bearing_sin * invdist
31 item0           bearing_cos * invdist
32 item1 megaarmor available
33 item1           respawn_norm
34 item1           bearing_sin * invdist
35 item1           bearing_cos * invdist
36 item2 soul      available
37 item2           respawn_norm
38 item2           bearing_sin * invdist
39 item2           bearing_cos * invdist
```
`invdist = min(512 / max(dist,1), 1)` so each item's bearing vector shrinks with
distance, encoding heading + proximity in 2 channels. bearings in radians.

**OBS_DIM = 40.**

---

## ACTION space — `NUM_ACTIONS = 486`

`9 (turn) × 3 (forward) × 3 (strafe) × 2 (fire) × 3 (weapon) = 486`.

Decode is mixed-radix, least-significant first:
```
weapon_sel = idx % 3 ; idx /= 3
fire       = idx % 2 ; idx /= 2
strafe_i   = idx % 3 ; idx /= 3
forward_i  = idx % 3 ; idx /= 3
turn_i     = idx                 (0..8)

TURNS   = [-1300,-700,-300,-120,0,120,300,700,1300]
MOVES   = [-40, 0, 50]    (forward)
STRAFE  = [-40, 0, 40]    (side)
WEAPONS = [0, 3, 5]       (Doom weapon CHANGE slot; 0 = keep)
```
`weapon` field semantics (doomgeneric `doomrl_action_t.weapon`): 0 = no change,
else BT_CHANGE to slot `weapon` (1..8). Slot 3 = shotgun, slot 5 = rocket
launcher. So WEAPONS index -> {keep, shotgun, rocket}. Chaingun (slot 4) is the
spawn default; the agent switches up to shotgun or rocket as it grabs them.

Action struct fields set by decode_action:
```
forward = MOVES[forward_i]
side    = STRAFE[strafe_i]
turn    = TURNS[turn_i]
fire    = fire
use_    = 0
weapon  = WEAPONS[weapon_sel]   (0/3/5)
```

---

## REWARD — `shaped_reward`

Frag dominant; all shaping kept well below +5.

```
+5.0   * frag (kill opponent)                 DOMINANT
-2.0   on death by opponent
-3.0   on self/environment death (suicide)
+0.01  * damage dealt to opponent (~+1 over a full kill, dense approach signal)
+0.0005 * min(moved,30)                        anti-camp
-- item control (one-time on pickup transition, from self-economy deltas) --
+0.5   grab rocket launcher   (ammo_misl rose AND we now ready/own the rocket weapon)
+0.5   grab megaarmor         (armortype became blue this tic)
+0.3   grab soulsphere        (health spiked > 50 this tic)
-- standing control (per tic, tiny) --
+0.002 if we hold the rocket launcher (ready_weapon == rocket OR own misl ammo>0 path)
+0.002 if armortype == blue (armor-stacked)
```
Pickup detection uses only self-state deltas already in `PlayerState`, no extra
FFI plumbing. Cumulative item shaping over an episode stays bounded under one
frag, so fragging stays the objective.

---

## Constants
- ARENA_HALF = 1024 (dumbbell long axis ~[-1024,1024] in x).
- Item respawn = 30 × 35 = 1050 tics (TICRATE = 35).
- NUM_KEY_ITEMS = 3, order [rocket(2003), megaarmor(2019), soulsphere(2013)].
