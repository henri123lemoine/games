#include "doomrl.h"

#include <string.h>
#include <math.h>

#include "doomgeneric.h"
#include "doomtype.h"
#include "doomdef.h"
#include "d_event.h"
#include "d_player.h"
#include "d_ticcmd.h"
#include "p_mobj.h"
#include "p_tick.h"
#include "p_local.h"
#include "tables.h"
#include "m_fixed.h"
#include "i_sound.h"
#include "i_timer.h"
#include "doomstat.h"
#include "d_main.h"
#include "d_loop.h"
#include "g_game.h"
#include "info.h"
#include "r_main.h"

extern thinker_t thinkercap;
extern boolean netdemo;

static uint32_t s_clock_ms = 0;
static const doomrl_action_t *s_pending_action = NULL;

static struct {
    const mobj_t *mo;
    int           start_tic;
    int           type;
    int           valid;
    int           ticks_since_seen;
    float         last_bearing_deg;
    float         last_dist;
} s_target;

void DG_Init(void) {}

void DG_DrawFrame(void) {}

void DG_SleepMs(uint32_t ms)
{
    s_clock_ms += ms ? ms : 1;
}

uint32_t DG_GetTicksMs(void)
{
    s_clock_ms += (1000u / TICRATE);
    return s_clock_ms;
}

int DG_GetKey(int *pressed, unsigned char *key)
{
    (void)pressed;
    (void)key;
    return 0;
}

void DG_SetWindowTitle(const char *title) { (void)title; }

static int clamp8(int v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return v;
}

static void action_to_ticcmd(const doomrl_action_t *a, ticcmd_t *cmd)
{
    cmd->forwardmove = (signed char)clamp8(a->forward);
    cmd->sidemove = (signed char)clamp8(a->side);
    cmd->angleturn = (short)a->turn;
    cmd->buttons = 0;
    if (a->fire)
        cmd->buttons |= BT_ATTACK;
    if (a->use)
        cmd->buttons |= BT_USE;
    if (a->weapon > 0)
    {
        cmd->buttons |= BT_CHANGE;
        cmd->buttons |= ((a->weapon - 1) << BT_WEAPONSHIFT) & BT_WEAPONMASK;
    }
}

void DGRL_OverrideTiccmd(ticcmd_t *cmd)
{
    if (s_pending_action == NULL)
        return;

    action_to_ticcmd(s_pending_action, cmd);
}

void doomrl_init(int argc, char **argv)
{
    extern boolean singletics;
    singletics = true;
    s_clock_ms = 0;
    s_pending_action = NULL;
    memset(&s_target, 0, sizeof(s_target));
    s_target.start_tic = -1;
    doomgeneric_Create(argc, argv);
}

void doomrl_step(const doomrl_action_t *action)
{
    s_pending_action = action;
    doomgeneric_Tick();
    s_pending_action = NULL;
}

int doomrl_tic(void)
{
    return gametic;
}

static float fx2f(fixed_t v)
{
    return (float)v / (float)FRACUNIT;
}

static float ang2deg(angle_t a)
{
    return (float)((double)a / (double)ANG_MAX * 360.0);
}

static float wrap180(float deg)
{
    while (deg > 180.0f) deg -= 360.0f;
    while (deg < -180.0f) deg += 360.0f;
    return deg;
}

void doomrl_get_state(doomrl_state_t *out)
{
    memset(out, 0, sizeof(*out));

    out->tic = gametic;
    out->gamestate = gamestate;
    out->level_done = (gamestate != GS_LEVEL);
    out->total_kills = totalkills;
    out->total_items = totalitems;
    out->total_secrets = totalsecret;

    player_t *pl = &players[consoleplayer];

    out->health = pl->health;
    out->armor = pl->armorpoints;
    out->armortype = pl->armortype;
    out->ready_weapon = pl->readyweapon;
    out->ammo[0] = pl->ammo[0];
    out->ammo[1] = pl->ammo[1];
    out->ammo[2] = pl->ammo[2];
    out->ammo[3] = pl->ammo[3];
    out->killcount = pl->killcount;
    out->itemcount = pl->itemcount;
    out->secretcount = pl->secretcount;

    mobj_t *mo = pl->mo;
    if (mo == NULL || gamestate != GS_LEVEL)
    {
        out->alive = 0;
        return;
    }

    out->alive = (pl->playerstate == PST_LIVE && pl->health > 0) ? 1 : 0;
    out->x = fx2f(mo->x);
    out->y = fx2f(mo->y);
    out->z = fx2f(mo->z);
    out->angle_deg = ang2deg(mo->angle);
    out->momx = fx2f(mo->momx);
    out->momy = fx2f(mo->momy);

    if (s_target.start_tic != levelstarttic)
    {
        memset(&s_target, 0, sizeof(s_target));
        s_target.start_tic = levelstarttic;
    }

    const mobj_t *nearest = NULL;
    float nearest_dist = 0.0f;
    float nearest_bearing = 0.0f;

    int saved_validcount = validcount;

    int n = 0;
    thinker_t *th = thinkercap.next;
    while (th != &thinkercap && n < DOOMRL_MAX_ENEMIES)
    {
        if (th->function.acp1 == (actionf_p1)P_MobjThinker)
        {
            mobj_t *m = (mobj_t *)th;
            int is_monster = (m->flags & MF_COUNTKILL) && !(m->flags & MF_CORPSE);
            if (is_monster && m->health > 0 && m != mo && P_CheckSight(mo, m))
            {
                doomrl_enemy_t *e = &out->enemies[n];
                e->type = m->type;
                e->x = fx2f(m->x);
                e->y = fx2f(m->y);
                e->z = fx2f(m->z);
                e->angle_deg = ang2deg(m->angle);
                e->health = m->health;
                float dx = e->x - out->x;
                float dy = e->y - out->y;
                e->dist = sqrtf(dx * dx + dy * dy);
                float abs_bearing = atan2f(dy, dx) * 180.0f / (float)M_PI;
                e->bearing_deg = wrap180(abs_bearing - out->angle_deg);
                e->rel_vx = fx2f(m->momx) - out->momx;
                e->rel_vy = fx2f(m->momy) - out->momy;
                e->awake = (m->target == mo) ? 1 : 0;

                if (nearest == NULL || e->dist < nearest_dist)
                {
                    nearest = m;
                    nearest_dist = e->dist;
                    nearest_bearing = e->bearing_deg;
                }
                n++;
            }
        }
        th = th->next;
    }

    validcount = saved_validcount;

    out->num_visible_enemies = n;

    if (nearest != NULL)
    {
        s_target.mo = nearest;
        s_target.type = nearest->type;
        s_target.valid = 1;
        s_target.ticks_since_seen = 0;
        s_target.last_bearing_deg = nearest_bearing;
        s_target.last_dist = nearest_dist;
    }
    else if (s_target.valid)
    {
        s_target.ticks_since_seen++;
    }

    out->target.type = s_target.type;
    out->target.valid = s_target.valid;
    out->target.ticks_since_seen = s_target.ticks_since_seen;
    out->target.last_bearing_deg = s_target.last_bearing_deg;
    out->target.last_dist = s_target.last_dist;
}

static int s_dm_players = 0;

static struct {
    const doomrl_action_t *pending[DOOMRL_MAX_PLAYERS];

    int   prev_frags[DOOMRL_MAX_PLAYERS];
    int   prev_deaths[DOOMRL_MAX_PLAYERS];
    int   prev_health[DOOMRL_MAX_PLAYERS];
    int   prev_armor[DOOMRL_MAX_PLAYERS];
    int   was_dead[DOOMRL_MAX_PLAYERS];
    int   deaths[DOOMRL_MAX_PLAYERS];
    float reward[DOOMRL_MAX_PLAYERS];

    int   opp_start_tic[DOOMRL_MAX_PLAYERS];
    int   opp_valid[DOOMRL_MAX_PLAYERS];
    int   opp_ticks_since_seen[DOOMRL_MAX_PLAYERS];
    float opp_last_bearing[DOOMRL_MAX_PLAYERS];
    float opp_last_dist[DOOMRL_MAX_PLAYERS];
} s_dm;

void DGRL_OverrideSet(ticcmd_t *cmds, boolean *ingame)
{
    if (s_dm_players < 2)
        return;

    for (int i = 0; i < s_dm_players; i++)
    {
        ingame[i] = true;
        memset(&cmds[i], 0, sizeof(ticcmd_t));
        if (players[i].playerstate == PST_DEAD)
            cmds[i].buttons = BT_USE;
        else if (s_dm.pending[i])
            action_to_ticcmd(s_dm.pending[i], &cmds[i]);
        cmds[i].consistancy = 0;
    }
}

static void dm_setup_match(int first_time)
{
    netgame = true;
    netdemo = true;
    deathmatch = 1;
    consoleplayer = 0;
    displayplayer = 0;
    for (int i = 0; i < MAXPLAYERS; i++)
        playeringame[i] = (i < s_dm_players);

    if (first_time)
    {
        G_InitNew(startskill, startepisode, startmap);
    }
    else
    {
        for (int i = 0; i < s_dm_players; i++)
            players[i].playerstate = PST_REBORN;
        gameaction = ga_loadlevel;
        for (int t = 0; t < 16 && gameaction != ga_nothing; t++)
        {
            for (int i = 0; i < DOOMRL_MAX_PLAYERS; i++)
                s_dm.pending[i] = NULL;
            doomgeneric_Tick();
        }
    }

    memset(&s_dm, 0, sizeof(s_dm));
    memset(&s_dm.pending, 0, sizeof(s_dm.pending));
    for (int i = 0; i < s_dm_players; i++)
    {
        s_dm.prev_health[i] = players[i].health;
        s_dm.prev_armor[i] = players[i].armorpoints;
        s_dm.opp_start_tic[i] = levelstarttic;
    }
}

void doomrl_dm_init(int argc, char **argv)
{
    extern boolean singletics;
    singletics = true;
    s_clock_ms = 0;
    s_pending_action = NULL;
    s_dm_players = DOOMRL_MAX_PLAYERS;
    memset(&s_dm, 0, sizeof(s_dm));

    doomgeneric_Create(argc, argv);
    dm_setup_match(1);
}

void doomrl_reset(void)
{
    if (s_dm_players < 2)
        return;
    dm_setup_match(0);
}

int doomrl_num_players(void)
{
    return s_dm_players;
}

void doomrl_dm_spawn_near(float dist)
{
    if (s_dm_players < 2)
        return;
    mobj_t *m0 = players[0].mo;
    mobj_t *m1 = players[1].mo;
    if (m0 == NULL || m1 == NULL)
        return;
    if (players[0].playerstate != PST_LIVE || players[1].playerstate != PST_LIVE)
        return;

    fixed_t d = (fixed_t)(dist * (float)FRACUNIT);
    // place seat 1 TOWARD the map center so the pair can't be pushed into the
    // void, then clamp the target x inside the arena (+/- 960 units).
    fixed_t dir = (m0->x > 0) ? -d : d;
    fixed_t tx = m0->x + dir;
    const fixed_t bound = (1024 - 64) * FRACUNIT;
    if (tx > bound) tx = bound;
    if (tx < -bound) tx = -bound;
    if (P_TeleportMove(m1, tx, m0->y))
    {
        m0->angle = (m0->x > 0) ? ANG180 : 0;
        m1->angle = (m0->x > 0) ? 0 : ANG180;
    }
}

static int player_kills_of_opponent(int seat)
{
    int opp = seat ^ 1;
    return players[seat].frags[opp];
}

void doomrl_dm_step(const doomrl_action_t *a0, const doomrl_action_t *a1)
{
    s_dm.pending[0] = a0;
    s_dm.pending[1] = a1;

    for (int i = 0; i < s_dm_players; i++)
        s_dm.reward[i] = 0.0f;

    int prev_dead[DOOMRL_MAX_PLAYERS];
    for (int i = 0; i < s_dm_players; i++)
        prev_dead[i] = s_dm.was_dead[i];

    doomgeneric_Tick();

    for (int i = 0; i < s_dm_players; i++)
    {
        player_t *pl = &players[i];

        int kills = player_kills_of_opponent(i);
        s_dm.reward[i] += (float)(kills - s_dm.prev_frags[i]);
        s_dm.prev_frags[i] = kills;

        int dead = (pl->playerstate == PST_DEAD) || (pl->health <= 0);
        if (dead && !s_dm.was_dead[i])
        {
            s_dm.deaths[i]++;
            s_dm.reward[i] -= 1.0f;
        }
        s_dm.was_dead[i] = dead;

        if (!dead && !prev_dead[i])
        {
            int dh = pl->health - s_dm.prev_health[i];
            int da = pl->armorpoints - s_dm.prev_armor[i];
            s_dm.reward[i] += 0.01f * (float)dh + 0.005f * (float)da;
        }
        s_dm.prev_health[i] = pl->health;
        s_dm.prev_armor[i] = pl->armorpoints;
    }

    s_dm.pending[0] = NULL;
    s_dm.pending[1] = NULL;
}

void doomrl_get_player_state(int seat, doomrl_player_state_t *out)
{
    memset(out, 0, sizeof(*out));
    if (seat < 0 || seat >= s_dm_players)
        return;

    out->seat = seat;
    out->reward = s_dm.reward[seat];
    out->deaths = s_dm.deaths[seat];
    out->frags = player_kills_of_opponent(seat);

    player_t *pl = &players[seat];
    out->health = pl->health;
    out->armor = pl->armorpoints;
    out->armortype = pl->armortype;
    out->ready_weapon = pl->readyweapon;
    for (int k = 0; k < 4; k++)
        out->ammo[k] = pl->ammo[k];

    mobj_t *mo = pl->mo;
    if (mo == NULL || gamestate != GS_LEVEL)
    {
        out->alive = 0;
        return;
    }

    out->alive = (pl->playerstate == PST_LIVE && pl->health > 0) ? 1 : 0;
    out->x = fx2f(mo->x);
    out->y = fx2f(mo->y);
    out->z = fx2f(mo->z);
    out->angle_deg = ang2deg(mo->angle);
    out->momx = fx2f(mo->momx);
    out->momy = fx2f(mo->momy);

    if (s_dm.opp_start_tic[seat] != levelstarttic)
    {
        s_dm.opp_start_tic[seat] = levelstarttic;
        s_dm.opp_valid[seat] = 0;
        s_dm.opp_ticks_since_seen[seat] = 0;
    }

    int opp = seat ^ 1;
    player_t *op = &players[opp];
    mobj_t *omo = op->mo;

    int visible = 0;
    if (omo != NULL && op->playerstate == PST_LIVE && op->health > 0)
    {
        int saved_validcount = validcount;
        visible = P_CheckSight(mo, omo) ? 1 : 0;
        validcount = saved_validcount;
    }

    if (visible)
    {
        float dx = fx2f(omo->x) - out->x;
        float dy = fx2f(omo->y) - out->y;
        float dist = sqrtf(dx * dx + dy * dy);
        float abs_bearing = atan2f(dy, dx) * 180.0f / (float)M_PI;
        float bearing = wrap180(abs_bearing - out->angle_deg);

        out->opponent_visible = 1;
        out->opp_bearing_deg = bearing;
        out->opp_dist = dist;
        out->opp_rel_vx = fx2f(omo->momx) - out->momx;
        out->opp_rel_vy = fx2f(omo->momy) - out->momy;
        out->opp_health = op->health;

        s_dm.opp_valid[seat] = 1;
        s_dm.opp_ticks_since_seen[seat] = 0;
        s_dm.opp_last_bearing[seat] = bearing;
        s_dm.opp_last_dist[seat] = dist;
    }
    else if (s_dm.opp_valid[seat])
    {
        s_dm.opp_ticks_since_seen[seat]++;
    }

    out->opp_memory.valid = s_dm.opp_valid[seat];
    out->opp_memory.ticks_since_seen = s_dm.opp_ticks_since_seen[seat];
    out->opp_memory.last_bearing_deg = s_dm.opp_last_bearing[seat];
    out->opp_memory.last_dist = s_dm.opp_last_dist[seat];
}
