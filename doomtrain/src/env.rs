use crate::ffi::{Action, Engine, PlayerState};

pub const OBS_DIM: usize = 18;
pub const MEAS_DIM: usize = 4;

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
    [
        (st.health as f32) / 100.0,
        (st.ammo[0] as f32) / 50.0,
        st.frags as f32,
        opp_damage,
    ]
}

const TURNS: [i16; 5] = [-1200, -400, 0, 400, 1200];
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
