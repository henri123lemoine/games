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
        "-file", "flatarena.wad",
        "-warp", "1", "1",
        "-skill", "3",
        "-deathmatch", "-solo-net", "-nomonsters",
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

// Flat per-seat state for JS: write 32 floats into `out` for `seat`. Layout is
// the canonical doomrl_player_state_t fields the JS observation encoder needs,
// in a fixed order (see web/doom-ai/forward.js). Keeps the obs encoder in JS
// reading the SAME numbers the trainer's observation() consumed.
EMSCRIPTEN_KEEPALIVE
void web_player_state(int seat, float *out)
{
    doomrl_player_state_t st;
    doomrl_get_player_state(seat, &st);
    int i = 0;
    out[i++] = (float)st.alive;
    out[i++] = st.x;
    out[i++] = st.y;
    out[i++] = st.z;
    out[i++] = st.angle_deg;
    out[i++] = st.momx;
    out[i++] = st.momy;
    out[i++] = (float)st.health;
    out[i++] = (float)st.armor;
    out[i++] = (float)st.ready_weapon;
    out[i++] = (float)st.ammo[0];
    out[i++] = (float)st.frags;
    out[i++] = (float)st.deaths;
    out[i++] = (float)st.opponent_visible;
    out[i++] = st.opp_bearing_deg;
    out[i++] = st.opp_dist;
    out[i++] = st.opp_rel_vx;
    out[i++] = st.opp_rel_vy;
    out[i++] = (float)st.opp_health;
    out[i++] = (float)st.opp_memory.valid;
    out[i++] = (float)st.opp_memory.ticks_since_seen;
    out[i++] = st.opp_memory.last_bearing_deg;
    out[i++] = st.opp_memory.last_dist;
}
