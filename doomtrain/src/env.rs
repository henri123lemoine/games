use crate::ffi::{Action, Engine, PlayerState};

pub const OBS_DIM: usize = 18;
pub const MEAS_DIM: usize = 5;

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
            "-deathmatch",
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

pub fn observation(st: &PlayerState) -> [f32; OBS_DIM] {
    let ang = st.angle_deg.to_radians();
    let opp_bear = st.opp_bearing_deg.to_radians();
    let mem_bear = st.opp_memory.last_bearing_deg.to_radians();
    [
        (st.health as f32) / 100.0,
        (st.armor as f32) / 100.0,
        (st.ammo[0] as f32) / 50.0,
        ang.sin(),
        ang.cos(),
        st.momx / 16.0,
        st.momy / 16.0,
        st.opponent_visible as f32,
        opp_bear.sin(),
        opp_bear.cos(),
        (st.opp_dist / 512.0).min(8.0),
        st.opp_rel_vx / 16.0,
        st.opp_rel_vy / 16.0,
        (st.opp_health as f32) / 100.0,
        st.opp_memory.valid as f32,
        (st.opp_memory.ticks_since_seen as f32 / 35.0).min(20.0),
        mem_bear.sin(),
        mem_bear.cos(),
    ]
}

/// Arnold-style light reward shaping for PPO, computed from the seat's previous
/// and current state plus the opponent's. +frag, -death, -suicide,
/// +small·distance-moved (anti-camp), +small·damage-dealt-to-opponent (dense).
pub fn shaped_reward(
    prev: &PlayerState,
    cur: &PlayerState,
    opp_prev: &PlayerState,
    opp_cur: &PlayerState,
) -> f32 {
    let mut r = 0.0f32;

    // frag (kill of opponent) — the dominant prize. Made large so finishing a
    // kill clearly beats the "chip damage and move" local optimum the policy
    // settled into when frag ≈ total diffuse damage reward.
    let frag_delta = (cur.frags - prev.frags).max(0) as f32;
    r += 5.0 * frag_delta;

    // death: alive→dead transition this tic.
    let died = prev.alive != 0 && cur.alive == 0;
    if died {
        // suicide vs killed-by-opponent: if the opponent didn't just score, it's
        // an environment/self death — penalize harder (anti-suicide).
        let opp_scored = (opp_cur.frags - opp_prev.frags).max(0) > 0;
        r -= if opp_scored { 2.0 } else { 3.0 };
    }

    // damage dealt this tic — dense credit toward the kill. Kept modest so its
    // cumulative total (~1 over a full kill) does not rival the +5 frag; it
    // shapes the approach, the frag is the payoff.
    if cur.alive != 0 {
        let dmg = (opp_prev.health - opp_cur.health).max(0) as f32;
        r += 0.01 * dmg;
    }

    // anti-camp: a tiny movement nudge, small enough not to compete with combat.
    if cur.alive != 0 {
        let dx = cur.x - prev.x;
        let dy = cur.y - prev.y;
        let moved = (dx * dx + dy * dy).sqrt();
        r += 0.0005 * moved.min(30.0);
    }

    r
}

pub fn measurements(st: &PlayerState) -> [f32; MEAS_DIM] {
    // opp_damage is a dense proxy for "hurting the enemy": it rises as we shoot
    // the opponent down from 100 hp and drops when they respawn, giving gradient
    // toward aiming/firing long before the sparse frag event. Still a game state
    // variable (opponent health), not hand-designed reward shaping.
    let opp_damage = if st.opponent_visible != 0 {
        (100 - st.opp_health).max(0) as f32 / 100.0
    } else {
        0.0
    };
    // aim_align: how well we're facing a visible enemy (1 = dead-on, 0 = 90deg+
    // off or invisible). A dense, instantaneous signal that rewards the
    // turn-to-face maneuver — the prerequisite for landing damage that DFP could
    // not credit-assign from opp_damage alone.
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

// Finer turn granularity (incl. small angles) so the policy can actually track a
// target — the coarse 5-level turn could not. 9 turns x 3 forward x 2 fire = 54.
const TURNS: [i16; 9] = [-1300, -700, -300, -120, 0, 120, 300, 700, 1300];
const MOVES: [i8; 3] = [-40, 0, 50];

pub const NUM_ACTIONS: usize = TURNS.len() * MOVES.len() * 2;

pub fn decode_action(idx: usize) -> Action {
    let fire = idx % 2;
    let m = (idx / 2) % MOVES.len();
    let t = (idx / 2) / MOVES.len();
    Action {
        forward: MOVES[m],
        side: 0,
        turn: TURNS[t],
        fire: fire as u8,
        use_: 0,
        weapon: 0,
    }
}

/// Snap a continuous Action (e.g. the scripted hunter's) to the nearest discrete
/// action index, so a mixed rollout stays in the same action space.
pub fn encode_action(a: &Action) -> usize {
    let nearest = |val: i32, arr: &[i16]| -> usize {
        arr.iter()
            .enumerate()
            .min_by_key(|(_, &v)| (v as i32 - val).abs())
            .map(|(i, _)| i)
            .unwrap()
    };
    let t = nearest(a.turn as i32, &TURNS);
    let move_arr: [i16; 3] = [MOVES[0] as i16, MOVES[1] as i16, MOVES[2] as i16];
    let mi = nearest(a.forward as i32, &move_arr);
    let fire = (a.fire != 0) as usize;
    (t * MOVES.len() + mi) * 2 + fire
}

/// A weakened scripted opponent for the curriculum: the hunter with aim NOISE
/// (random bearing jitter) and REACTION DELAY (acts on a stale observation,
/// refreshed every `react_tics`), so a learning policy can actually out-duel it
/// and frag-share can climb off 0. `skill` in [0,1] scales toward the perfect
/// hunter (1 = no noise/delay).
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

    /// skill 0 → a near-passive target: large aim noise, slow reactions, and it
    /// rarely fires (fire_prob low) so it doesn't kill the learner — a beatable
    /// dummy the policy can frag during exploration to BOOTSTRAP the first kills,
    /// which PPO then reinforces. skill 1 → the perfect, always-firing hunter.
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
            // gate firing by fire_prob so a low-skill bot rarely shoots.
            if a.fire != 0 && (self.rand_unit() * 0.5 + 0.5) > self.fire_prob {
                a.fire = 0;
            }
            self.last_action = a;
        }
        self.counter = self.counter.wrapping_add(1);
        self.last_action
    }
}

/// A fixed scripted opponent for evaluation: turn toward the visible opponent,
/// close to firing range, shoot when roughly aligned. Uses only the LOS-gated
/// observation (no ground-truth peeking), so it is a fair fixed benchmark.
pub fn scripted_hunter(st: &PlayerState) -> Action {
    let mut a = Action::default();
    if st.alive == 0 {
        return a;
    }
    let (bearing, dist, visible) = if st.opponent_visible != 0 {
        (st.opp_bearing_deg, st.opp_dist, true)
    } else if st.opp_memory.valid != 0 && st.opp_memory.ticks_since_seen < 70 {
        (
            st.opp_memory.last_bearing_deg,
            st.opp_memory.last_dist,
            false,
        )
    } else {
        (0.0, 9999.0, false)
    };

    a.turn = (bearing * 80.0).clamp(-1300.0, 1300.0) as i16;
    a.forward = if dist > 256.0 { 50 } else { 0 };
    if visible && bearing.abs() < 20.0 {
        a.fire = 1;
    }
    a
}
