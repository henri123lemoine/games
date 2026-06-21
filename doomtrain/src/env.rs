use crate::ffi::{Action, Engine, PlayerState};

// Strategic 1v1 OBS/ACTION/reward — see doomrl/STRATEGIC_CONTRACT.md. The same
// layout is mirrored in doomrl_web.c (web_player_state) and forward.js.
pub const OBS_DIM: usize = 40;
pub const MEAS_DIM: usize = 5;

const ARENA_HALF: f32 = 1024.0;

pub struct DoomEnv {
    engine: Engine,
}

impl DoomEnv {
    pub fn new(iwad: &str, arena: Option<&str>) -> DoomEnv {
        let mut args: Vec<&str> = vec!["doomrl", "-iwad", iwad];
        if let Some(wad) = arena {
            std::env::set_var("DOOMRL_ALLOW_FILE", "1");
            args.push("-file");
            args.push(wad);
        }
        args.extend_from_slice(&[
            "-warp",
            "1",
            "1",
            "-skill",
            "3",
            "-altdeath",
            "-solo-net",
            "-nomonsters",
            "-nomusic",
            "-nosfx",
            "-nodraw",
        ]);
        DoomEnv {
            engine: Engine::new(&args),
        }
    }

    pub fn reset(&self) {
        self.engine.reset();
    }

    /// Teleport the two players to `dist` units apart (curriculum spawn-near).
    pub fn spawn_near(&self, dist: f32) {
        self.engine.spawn_near(dist);
    }

    pub fn step(&self, a0: Action, a1: Action) {
        self.engine.step(&a0, &a1);
    }

    pub fn player_state(&self, seat: i32) -> PlayerState {
        self.engine.player_state(seat)
    }

    pub fn num_players(&self) -> i32 {
        self.engine.num_players()
    }
}

// Doom weapon slot identities (BT_CHANGE slot numbers) used in obs + action.
const WP_SHOTGUN: i32 = 3;
const WP_CHAINGUN: i32 = 4;
const WP_ROCKET: i32 = 5;

pub fn observation(st: &PlayerState) -> [f32; OBS_DIM] {
    let ang = st.angle_deg.to_radians();
    let opp_bear = st.opp_bearing_deg.to_radians();
    let mem_bear = st.opp_memory.last_bearing_deg.to_radians();

    let mut o = [0.0f32; OBS_DIM];
    o[0] = (st.health as f32) / 100.0;
    o[1] = (st.armor as f32) / 200.0;
    o[2] = (st.armortype == 1) as i32 as f32;
    o[3] = (st.armortype == 2) as i32 as f32;
    o[4] = (st.ammo[0] as f32) / 200.0;
    o[5] = (st.ammo[1] as f32) / 50.0;
    o[6] = (st.ammo[2] as f32) / 300.0;
    o[7] = (st.ammo[3] as f32) / 50.0;
    o[8] = (st.ready_weapon == WP_SHOTGUN) as i32 as f32;
    o[9] = (st.ready_weapon == WP_CHAINGUN) as i32 as f32;
    o[10] = (st.ready_weapon == WP_ROCKET) as i32 as f32;
    o[11] = ang.sin();
    o[12] = ang.cos();
    o[13] = st.x / ARENA_HALF;
    o[14] = st.y / ARENA_HALF;
    o[15] = st.momx / 16.0;
    o[16] = st.momy / 16.0;
    o[17] = st.opponent_visible as f32;
    o[18] = opp_bear.sin();
    o[19] = opp_bear.cos();
    o[20] = (st.opp_dist / 512.0).min(8.0);
    o[21] = st.opp_rel_vx / 16.0;
    o[22] = st.opp_rel_vy / 16.0;
    o[23] = (st.opp_health as f32) / 100.0;
    o[24] = st.opp_memory.valid as f32;
    o[25] = (st.opp_memory.ticks_since_seen as f32 / 35.0).min(20.0);
    o[26] = mem_bear.sin();
    o[27] = mem_bear.cos();
    // 3 key items x 4 channels: [available, respawn_norm, bear_sin*invdist, bear_cos*invdist]
    for k in 0..crate::ffi::NUM_KEY_ITEMS {
        let it = &st.key_items[k];
        let base = 28 + k * 4;
        let bear = it.bearing_deg.to_radians();
        let invdist = (512.0 / it.dist.max(1.0)).min(1.0);
        o[base] = it.available as f32;
        o[base + 1] = (it.respawn_secs / 30.0).clamp(0.0, 1.0);
        o[base + 2] = bear.sin() * invdist;
        o[base + 3] = bear.cos() * invdist;
    }
    o
}

/// PPO reward shaping. Frag dominant (+5); item-control shaping kept well below
/// it so fragging stays the objective. See doomrl/STRATEGIC_CONTRACT.md.
pub fn shaped_reward(
    prev: &PlayerState,
    cur: &PlayerState,
    opp_prev: &PlayerState,
    opp_cur: &PlayerState,
) -> f32 {
    let mut r = 0.0f32;

    // frag (kill of opponent) — the dominant prize.
    let frag_delta = (cur.frags - prev.frags).max(0) as f32;
    r += 5.0 * frag_delta;

    // death: alive->dead transition this tic.
    let died = prev.alive != 0 && cur.alive == 0;
    if died {
        let opp_scored = (opp_cur.frags - opp_prev.frags).max(0) > 0;
        r -= if opp_scored { 2.0 } else { 3.0 };
    }

    if cur.alive != 0 {
        // dense damage-dealt credit toward the kill (~+1 over a full kill).
        let dmg = (opp_prev.health - opp_cur.health).max(0) as f32;
        r += 0.01 * dmg;

        // anti-camp movement nudge.
        let dx = cur.x - prev.x;
        let dy = cur.y - prev.y;
        let moved = (dx * dx + dy * dy).sqrt();
        r += 0.0005 * moved.min(30.0);

        // --- item control: one-time pickup bonuses from self-economy deltas ---
        // rocket launcher: rocket ammo rose AND we now own/ready the rocket weapon.
        let got_rockets = cur.ammo[3] > prev.ammo[3];
        let has_rocket = cur.ready_weapon == WP_ROCKET || cur.ammo[3] > 0;
        if got_rockets && has_rocket && prev.ammo[3] == 0 {
            r += 0.5;
        }
        // megaarmor: armortype became blue this tic.
        if cur.armortype == 2 && prev.armortype != 2 {
            r += 0.5;
        }
        // soulsphere: health spiked > 50 in one tic.
        if cur.health - prev.health > 50 {
            r += 0.3;
        }

        // --- standing control: tiny per-tic holds ---
        if cur.ready_weapon == WP_ROCKET || cur.ammo[3] > 0 {
            r += 0.002;
        }
        if cur.armortype == 2 {
            r += 0.002;
        }
    }

    r
}

pub fn measurements(st: &PlayerState) -> [f32; MEAS_DIM] {
    let opp_damage = if st.opponent_visible != 0 {
        (100 - st.opp_health).max(0) as f32 / 100.0
    } else {
        0.0
    };
    let aim_align = if st.opponent_visible != 0 {
        st.opp_bearing_deg.to_radians().cos().max(0.0)
    } else {
        0.0
    };
    [
        (st.health as f32) / 100.0,
        (st.ammo[0] as f32) / 50.0,
        st.frags as f32,
        opp_damage,
        aim_align,
    ]
}

// Action space: 9 turn x 3 forward x 3 strafe x 2 fire x 3 weapon = 486.
const TURNS: [i16; 9] = [-1300, -700, -300, -120, 0, 120, 300, 700, 1300];
const MOVES: [i8; 3] = [-40, 0, 50];
const STRAFE: [i8; 3] = [-40, 0, 40];
// Doom BT_CHANGE weapon slot: 0 = keep current, 3 = shotgun, 5 = rocket.
const WEAPONS: [u8; 3] = [0, 3, 5];

pub const NUM_ACTIONS: usize =
    TURNS.len() * MOVES.len() * STRAFE.len() * 2 * WEAPONS.len();

pub fn decode_action(idx: usize) -> Action {
    let weapon_sel = idx % WEAPONS.len();
    let mut rest = idx / WEAPONS.len();
    let fire = rest % 2;
    rest /= 2;
    let strafe_i = rest % STRAFE.len();
    rest /= STRAFE.len();
    let forward_i = rest % MOVES.len();
    let turn_i = rest / MOVES.len();
    Action {
        forward: MOVES[forward_i],
        side: STRAFE[strafe_i],
        turn: TURNS[turn_i],
        fire: fire as u8,
        use_: 0,
        weapon: WEAPONS[weapon_sel],
    }
}

/// Snap a continuous Action (e.g. the scripted hunter's) to the nearest discrete
/// action index, so a mixed rollout stays in the same action space.
pub fn encode_action(a: &Action) -> usize {
    let nearest_i16 = |val: i32, arr: &[i16]| -> usize {
        arr.iter()
            .enumerate()
            .min_by_key(|(_, &v)| (v as i32 - val).abs())
            .map(|(i, _)| i)
            .unwrap()
    };
    let nearest_i8 = |val: i32, arr: &[i8]| -> usize {
        arr.iter()
            .enumerate()
            .min_by_key(|(_, &v)| (v as i32 - val).abs())
            .map(|(i, _)| i)
            .unwrap()
    };
    let turn_i = nearest_i16(a.turn as i32, &TURNS);
    let forward_i = nearest_i8(a.forward as i32, &MOVES);
    let strafe_i = nearest_i8(a.side as i32, &STRAFE);
    let fire = (a.fire != 0) as usize;
    let weapon_sel = WEAPONS
        .iter()
        .position(|&w| w == a.weapon)
        .unwrap_or(0);
    (((turn_i * MOVES.len() + forward_i) * STRAFE.len() + strafe_i) * 2 + fire)
        * WEAPONS.len()
        + weapon_sel
}

/// A weakened scripted opponent for the curriculum: the hunter with aim NOISE
/// and REACTION DELAY so a learning policy can out-duel it. `skill` in [0,1].
pub struct BeatableBot {
    pub aim_noise_deg: f32,
    pub react_tics: u32,
    pub fire_prob: f32,
    counter: u32,
    last_action: Action,
    rng: u64,
}

impl BeatableBot {
    pub fn new(aim_noise_deg: f32, react_tics: u32, seed: u64) -> BeatableBot {
        BeatableBot {
            aim_noise_deg,
            react_tics,
            fire_prob: 1.0,
            counter: 0,
            last_action: Action::default(),
            rng: seed | 1,
        }
    }

    pub fn for_skill(skill: f32, seed: u64) -> BeatableBot {
        let s = skill.clamp(0.0, 1.0);
        let noise = 25.0 * (1.0 - s);
        let react = (1.0 + 5.0 * (1.0 - s)).round() as u32;
        let fire_prob = 0.15 + 0.85 * s;
        let mut b = BeatableBot::new(noise, react.max(1), seed);
        b.fire_prob = fire_prob;
        b
    }

    fn rand_unit(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        ((self.rng >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    }

    pub fn act(&mut self, st: &PlayerState) -> Action {
        if self.counter.is_multiple_of(self.react_tics.max(1)) {
            let mut s = *st;
            let jitter = self.rand_unit() * self.aim_noise_deg;
            s.opp_bearing_deg += jitter;
            s.opp_memory.last_bearing_deg += jitter;
            let mut a = scripted_hunter(&s);
            if a.fire != 0 && (self.rand_unit() * 0.5 + 0.5) > self.fire_prob {
                a.fire = 0;
            }
            self.last_action = a;
        }
        self.counter = self.counter.wrapping_add(1);
        self.last_action
    }
}

/// A fixed scripted opponent for evaluation and BC. Hunts the visible opponent,
/// closes to firing range, shoots when aligned, and additionally seeks the
/// nearest AVAILABLE key item when no opponent is visible — so it exercises the
/// strafe and item-bearing parts of the new action/obs space (a competent
/// strategic demonstrator for the warmstart, not just a turret).
pub fn scripted_hunter(st: &PlayerState) -> Action {
    let mut a = Action::default();
    if st.alive == 0 {
        return a;
    }
    let (bearing, dist, visible) = if st.opponent_visible != 0 {
        (st.opp_bearing_deg, st.opp_dist, true)
    } else if st.opp_memory.valid != 0 && st.opp_memory.ticks_since_seen < 70 {
        (st.opp_memory.last_bearing_deg, st.opp_memory.last_dist, false)
    } else {
        // no opponent: steer toward the nearest available key item (item economy).
        let mut best: Option<(f32, f32)> = None;
        for k in 0..crate::ffi::NUM_KEY_ITEMS {
            let it = &st.key_items[k];
            if it.available != 0 {
                match best {
                    Some((_, bd)) if bd <= it.dist => {}
                    _ => best = Some((it.bearing_deg, it.dist)),
                }
            }
        }
        match best {
            Some((b, d)) => (b, d, false),
            None => (0.0, 9999.0, false),
        }
    };

    a.turn = (bearing * 80.0).clamp(-1300.0, 1300.0) as i16;
    a.forward = if dist > 256.0 { 50 } else { 0 };
    // strafe a little when in a firefight to dodge (use bearing sign to weave).
    if visible && dist < 512.0 {
        a.side = if bearing >= 0.0 { 40 } else { -40 };
    }
    // switch to the rocket launcher once we own rockets (better duel weapon).
    if st.ammo[3] > 0 && st.ready_weapon != WP_ROCKET as i32 {
        a.weapon = WP_ROCKET as u8;
    } else if st.ready_weapon == 1 || st.ready_weapon == 2 {
        // off the fist/pistol: grab the chaingun/shotgun line via shotgun slot.
        a.weapon = WP_SHOTGUN as u8;
    }
    if visible && bearing.abs() < 20.0 {
        a.fire = 1;
    }
    a
}
