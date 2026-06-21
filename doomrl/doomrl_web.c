#include "doomrl.h"

#include <emscripten.h>
#include <string.h>

#include "doomgeneric.h"

extern pixel_t *DG_ScreenBuffer;

// Pending per-tic actions for both seats, set from JS before each step.
static doomrl_action_t s_web_a0;
static doomrl_action_t s_web_a1;

EMSCRIPTEN_KEEPALIVE
void web_init(void)
{
    // Render ON (no -nodraw): D_Display draws players[consoleplayer]'s POV into
    // DG_ScreenBuffer each tic. consoleplayer is seat 0 (the human).
    char *argv[] = {
        "doomrl",
        "-iwad", "doom1.wad",
        "-file", "dumbbell.wad",
        "-warp", "1", "1",
        "-skill", "3",
        "-altdeath", "-solo-net", "-nomonsters",
        "-nomusic", "-nosfx",
    };
    int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    doomrl_dm_init(argc, argv);
}

EMSCRIPTEN_KEEPALIVE
void web_set_action(int seat, int forward, int side, int turn, int fire, int use, int weapon)
{
    doomrl_action_t a;
    a.forward = (signed char)forward;
    a.side = (signed char)side;
    a.turn = (short)turn;
    a.fire = (unsigned char)(fire != 0);
    a.use = (unsigned char)(use != 0);
    a.weapon = (unsigned char)weapon;
    if (seat == 0)
        s_web_a0 = a;
    else
        s_web_a1 = a;
}

EMSCRIPTEN_KEEPALIVE
void web_step(void)
{
    doomrl_dm_step(&s_web_a0, &s_web_a1);
}

EMSCRIPTEN_KEEPALIVE
void web_spawn_near(float dist)
{
    doomrl_dm_spawn_near(dist);
}

EMSCRIPTEN_KEEPALIVE
void web_reset(void)
{
    doomrl_reset();
}

EMSCRIPTEN_KEEPALIVE
pixel_t *web_screenbuffer(void)
{
    return DG_ScreenBuffer;
}

EMSCRIPTEN_KEEPALIVE
int web_screen_w(void) { return DOOMGENERIC_RESX; }

EMSCRIPTEN_KEEPALIVE
int web_screen_h(void) { return DOOMGENERIC_RESY; }

// Flat per-seat state for JS: write PLAYER_STATE_FLOATS (39) floats into `out`
// for `seat`, in the exact order of the strategic OBS/state contract
// (doomrl/STRATEGIC_CONTRACT.md). The JS observation encoder reads the SAME
// numbers the trainer's observation() consumes — this is the parity bridge.
EMSCRIPTEN_KEEPALIVE
void web_player_state(int seat, float *out)
{
    doomrl_player_state_t st;
    doomrl_get_player_state(seat, &st);
    int i = 0;
    out[i++] = (float)st.alive;             /* 0  */
    out[i++] = st.x;                        /* 1  */
    out[i++] = st.y;                        /* 2  */
    out[i++] = st.z;                        /* 3  */
    out[i++] = st.angle_deg;                /* 4  */
    out[i++] = st.momx;                     /* 5  */
    out[i++] = st.momy;                     /* 6  */
    out[i++] = (float)st.health;            /* 7  */
    out[i++] = (float)st.armor;             /* 8  */
    out[i++] = (float)st.armortype;         /* 9  */
    out[i++] = (float)st.ready_weapon;      /* 10 */
    out[i++] = (float)st.ammo[0];           /* 11 clip  */
    out[i++] = (float)st.ammo[1];           /* 12 shell */
    out[i++] = (float)st.ammo[2];           /* 13 cell  */
    out[i++] = (float)st.ammo[3];           /* 14 misl  */
    out[i++] = (float)st.frags;             /* 15 */
    out[i++] = (float)st.deaths;            /* 16 */
    out[i++] = (float)st.opponent_visible;  /* 17 */
    out[i++] = st.opp_bearing_deg;          /* 18 */
    out[i++] = st.opp_dist;                 /* 19 */
    out[i++] = st.opp_rel_vx;               /* 20 */
    out[i++] = st.opp_rel_vy;               /* 21 */
    out[i++] = (float)st.opp_health;        /* 22 */
    out[i++] = (float)st.opp_memory.valid;          /* 23 */
    out[i++] = (float)st.opp_memory.ticks_since_seen;/* 24 */
    out[i++] = st.opp_memory.last_bearing_deg;      /* 25 */
    out[i++] = st.opp_memory.last_dist;             /* 26 */
    for (int k = 0; k < DOOMRL_NUM_KEY_ITEMS; k++)  /* 27..38: 3 items x 4 */
    {
        out[i++] = (float)st.key_items[k].available;
        out[i++] = st.key_items[k].respawn_secs;
        out[i++] = st.key_items[k].bearing_deg;
        out[i++] = st.key_items[k].dist;
    }
}
