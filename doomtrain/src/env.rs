use crate::ffi::{Action, Engine, PlayerState};

pub const OBS_DIM: usize = 18;
pub const MEAS_DIM: usize = 3;

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
    [
        (st.health as f32) / 100.0,
        (st.ammo[0] as f32) / 50.0,
        st.frags as f32,
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
