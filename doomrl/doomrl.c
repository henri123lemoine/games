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
#include "info.h"
#include "r_main.h"

extern thinker_t thinkercap;

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

void DGRL_OverrideTiccmd(ticcmd_t *cmd)
{
    if (s_pending_action == NULL)
        return;

    cmd->forwardmove = (signed char)clamp8(s_pending_action->forward);
    cmd->sidemove = (signed char)clamp8(s_pending_action->side);
    cmd->angleturn = (short)s_pending_action->turn;
    cmd->buttons = 0;
    if (s_pending_action->fire)
        cmd->buttons |= BT_ATTACK;
    if (s_pending_action->use)
        cmd->buttons |= BT_USE;
    if (s_pending_action->weapon > 0)
    {
        cmd->buttons |= BT_CHANGE;
        cmd->buttons |= ((s_pending_action->weapon - 1) << BT_WEAPONSHIFT) & BT_WEAPONMASK;
    }
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
