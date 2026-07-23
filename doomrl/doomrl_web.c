#include "doomrl.h"

#include <emscripten.h>
#include <stdio.h>
#include <string.h>

#include "doomgeneric.h"
#include "i_swap.h"
#include "i_video.h"
#include "m_menu.h"
#include "v_video.h"
#include "w_wad.h"
#include "z_zone.h"

extern pixel_t *DG_ScreenBuffer;
extern void D_Display(void);

// Pending per-tic actions for all vanilla-Doom seats, set from JS before step.
static doomrl_action_t s_web_actions[DOOMRL_MAX_PLAYERS];

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
int web_set_player_count(int num_players)
{
    memset(s_web_actions, 0, sizeof(s_web_actions));
    return doomrl_dm_set_player_count(num_players);
}

EMSCRIPTEN_KEEPALIVE
int web_num_players(void)
{
    return doomrl_num_players();
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
    if (seat >= 0 && seat < DOOMRL_MAX_PLAYERS)
        s_web_actions[seat] = a;
}

EMSCRIPTEN_KEEPALIVE
void web_step(void)
{
    doomrl_dm_step_all(s_web_actions);
}

EMSCRIPTEN_KEEPALIVE
void web_draw_pause(int selection)
{
    char *continue_text = "CONTINUE";
    char *home_text = "HOME";
    int continue_y = 92;
    int home_y = 112;
    int continue_x;
    int home_x;
    int cursor_x;
    patch_t *pause_patch;
    patch_t *skull_patch;

    /* Redraw the frozen game first so moving the skull never leaves trails. */
    D_Display();

    pause_patch = W_CacheLumpName("M_PAUSE", PU_CACHE);
    skull_patch = W_CacheLumpName("M_SKULL1", PU_CACHE);
    V_DrawPatchDirect((SCREENWIDTH - SHORT(pause_patch->width)) / 2, 64, pause_patch);

    continue_x = (SCREENWIDTH - M_StringWidth(continue_text)) / 2;
    home_x = (SCREENWIDTH - M_StringWidth(home_text)) / 2;
    M_WriteText(continue_x, continue_y, continue_text);
    M_WriteText(home_x, home_y, home_text);

    cursor_x = (continue_x < home_x ? continue_x : home_x) - 32;
    V_DrawPatchDirect(cursor_x, (selection ? home_y : continue_y) - 5, skull_patch);
    I_FinishUpdate();
}

EMSCRIPTEN_KEEPALIVE
void web_draw_setup(int bot_count, int difficulty, int selection)
{
    static char *difficulty_names[] = { "CASUAL", "STANDARD", "RELENTLESS" };
    char opponents_text[32];
    char difficulty_text[32];
    char *fight_text = "FIGHT";
    int row_y[] = { 88, 108, 132 };
    int row_x[3];
    int cursor_x;
    patch_t *logo_patch;
    patch_t *skull_patch;

    if (bot_count < 1) bot_count = 1;
    if (bot_count > 3) bot_count = 3;
    if (difficulty < 0) difficulty = 0;
    if (difficulty > 2) difficulty = 2;
    if (selection < 0) selection = 0;
    if (selection > 2) selection = 2;

    D_Display();
    logo_patch = W_CacheLumpName("M_DOOM", PU_CACHE);
    skull_patch = W_CacheLumpName("M_SKULL1", PU_CACHE);
    V_DrawPatchDirect(94, 2, logo_patch);

    M_WriteText((SCREENWIDTH - M_StringWidth("DEATHMATCH")) / 2, 66, "DEATHMATCH");
    snprintf(opponents_text, sizeof(opponents_text), "AI OPPONENTS  %d", bot_count);
    snprintf(difficulty_text, sizeof(difficulty_text), "SKILL  %s", difficulty_names[difficulty]);

    row_x[0] = (SCREENWIDTH - M_StringWidth(opponents_text)) / 2;
    row_x[1] = (SCREENWIDTH - M_StringWidth(difficulty_text)) / 2;
    row_x[2] = (SCREENWIDTH - M_StringWidth(fight_text)) / 2;
    M_WriteText(row_x[0], row_y[0], opponents_text);
    M_WriteText(row_x[1], row_y[1], difficulty_text);
    M_WriteText(row_x[2], row_y[2], fight_text);

    cursor_x = row_x[0];
    if (row_x[1] < cursor_x) cursor_x = row_x[1];
    if (row_x[2] < cursor_x) cursor_x = row_x[2];
    V_DrawPatchDirect(cursor_x - 32, row_y[selection] - 5, skull_patch);
    I_FinishUpdate();
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
