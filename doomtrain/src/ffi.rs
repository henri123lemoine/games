use std::ffi::CString;
use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TargetMemory {
    pub type_: c_int,
    pub valid: c_int,
    pub ticks_since_seen: c_int,
    pub last_bearing_deg: f32,
    pub last_dist: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlayerState {
    pub seat: c_int,
    pub alive: c_int,

    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub angle_deg: f32,
    pub momx: f32,
    pub momy: f32,

    pub health: c_int,
    pub armor: c_int,
    pub armortype: c_int,
    pub ready_weapon: c_int,
    pub ammo: [c_int; 4],
    pub frags: c_int,
    pub deaths: c_int,

    pub opponent_visible: c_int,
    pub opp_bearing_deg: f32,
    pub opp_dist: f32,
    pub opp_rel_vx: f32,
    pub opp_rel_vy: f32,
    pub opp_health: c_int,

    pub opp_memory: TargetMemory,

    pub reward: f32,
}

impl Default for PlayerState {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Action {
    pub forward: i8,
    pub side: i8,
    pub turn: i16,
    pub fire: u8,
    pub use_: u8,
    pub weapon: u8,
}

extern "C" {
    fn doomrl_dm_init(argc: c_int, argv: *mut *mut c_char);
    fn doomrl_dm_step(a0: *const Action, a1: *const Action);
    fn doomrl_get_player_state(seat: c_int, out: *mut PlayerState);
    fn doomrl_reset();
    fn doomrl_num_players() -> c_int;
}

pub struct Engine {
    _argv_keepalive: Vec<CString>,
}

impl Engine {
    pub fn new(args: &[&str]) -> Engine {
        let cstrings: Vec<CString> = args.iter().map(|s| CString::new(*s).unwrap()).collect();
        let mut ptrs: Vec<*mut c_char> =
            cstrings.iter().map(|c| c.as_ptr() as *mut c_char).collect();
        unsafe {
            doomrl_dm_init(ptrs.len() as c_int, ptrs.as_mut_ptr());
        }
        Engine {
            _argv_keepalive: cstrings,
        }
    }

    pub fn step(&self, a0: &Action, a1: &Action) {
        unsafe { doomrl_dm_step(a0, a1) }
    }

    pub fn player_state(&self, seat: i32) -> PlayerState {
        let mut st = PlayerState::default();
        unsafe { doomrl_get_player_state(seat, &mut st) }
        st
    }

    pub fn reset(&self) {
        unsafe { doomrl_reset() }
    }

    pub fn num_players(&self) -> i32 {
        unsafe { doomrl_num_players() }
    }
}
