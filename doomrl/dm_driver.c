#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "doomrl.h"
#include "doomtype.h"
#include "d_player.h"
#include "p_mobj.h"
#include "p_local.h"
#include "m_fixed.h"
#include "tables.h"
#include "doomstat.h"

extern player_t players[];

static void place_for_duel(void)
{
    mobj_t *m0 = players[0].mo;
    mobj_t *m1 = players[1].mo;
    if (m0 == NULL || m1 == NULL)
        return;
    fixed_t dir = (m0->x > 0) ? -(256 * FRACUNIT) : (256 * FRACUNIT);
    fixed_t tx = m0->x + dir;
    const fixed_t bound = (1024 - 64) * FRACUNIT;
    if (tx > bound) tx = bound;
    if (tx < -bound) tx = -bound;
    P_TeleportMove(m1, tx, m0->y);
    m0->angle = (m0->x > 0) ? ANG180 : 0;
    m1->angle = (m0->x > 0) ? 0 : ANG180;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static float ang_to_deg(unsigned int a)
{
    return (float)((double)a / 4294967296.0 * 360.0);
}

static float wrap180(float d)
{
    while (d > 180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

static void hunter_action(int seat, const doomrl_player_state_t *st, doomrl_action_t *a)
{
    memset(a, 0, sizeof(*a));

    mobj_t *me = players[seat].mo;
    mobj_t *op = players[seat ^ 1].mo;
    if (me == NULL || op == NULL || players[seat].playerstate != PST_LIVE)
        return;

    float dx = (float)(op->x - me->x) / 65536.0f;
    float dy = (float)(op->y - me->y) / 65536.0f;
    float want = atan2f(dy, dx) * 180.0f / (float)M_PI;
    float bearing = wrap180(want - ang_to_deg(me->angle));

    a->turn = (short)(bearing * 80.0f);
    if (a->turn > 1300) a->turn = 1300;
    if (a->turn < -1300) a->turn = -1300;

    if (st->opp_dist > 256.0f)
        a->forward = 50;
    else
        a->forward = 0;

    if (st->opponent_visible && bearing > -20.0f && bearing < 20.0f)
        a->fire = 1;
}

int main(int argc, char **argv)
{
    int episodes = 3;
    int ep_tics = 6000;
    int print_every = 1500;
    int duel = 0;
    static char ep[2] = "1";
    static char mp[2] = "1";
    const char *arena = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (!strncmp(argv[i], "--episodes=", 11)) episodes = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--tics=", 7)) ep_tics = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--print-every=", 14)) print_every = atoi(argv[i] + 14);
        else if (!strcmp(argv[i], "--duel")) duel = 1;
        else if (!strncmp(argv[i], "--map=", 6)) mp[0] = argv[i][6];
        else if (!strncmp(argv[i], "--episode=", 10)) ep[0] = argv[i][10];
        else if (!strncmp(argv[i], "--arena=", 8)) arena = argv[i] + 8;
    }

    char *base_argv[] = {
        "doomrl",
        "-iwad", "../web/app/public/doom/doom1.wad",
        "-warp", ep, mp,
        "-skill", "3",
        "-altdeath", "-solo-net", "-nomonsters",
        "-nomusic", "-nosfx", "-nodraw",
    };
    char *arena_argv[] = {
        "doomrl",
        "-iwad", "../web/app/public/doom/doom1.wad",
        "-file", arena ? (char *)arena : "",
        "-warp", ep, mp,
        "-skill", "3",
        "-altdeath", "-solo-net", "-nomonsters",
        "-nomusic", "-nosfx", "-nodraw",
    };
    char **doom_argv = arena ? arena_argv : base_argv;
    int doom_argc = arena ? (int)(sizeof(arena_argv) / sizeof(arena_argv[0]))
                          : (int)(sizeof(base_argv) / sizeof(base_argv[0]));
    if (arena)
        setenv("DOOMRL_ALLOW_FILE", "1", 1);

    doomrl_dm_init(doom_argc, doom_argv);
    printf("init: players=%d  (deathmatch=%d netgame=%d)\n", doomrl_num_players(), deathmatch, netgame);

    double t0 = now_seconds();
    int grand_tics = 0, grand_frags = 0, grand_deaths = 0;
    long saw_total = 0;

    for (int ep = 0; ep < episodes; ep++)
    {
        if (ep > 0)
            doomrl_reset();

        if (duel)
        {
            doomrl_action_t z;
            memset(&z, 0, sizeof(z));
            doomrl_dm_step(&z, &z);
            place_for_duel();
        }

        doomrl_player_state_t s0, s1;
        doomrl_get_player_state(0, &s0);
        doomrl_get_player_state(1, &s1);
        float sep = sqrtf((s0.x - s1.x) * (s0.x - s1.x) + (s0.y - s1.y) * (s0.y - s1.y));
        printf("ep%d start: seat0 spawn=(%.0f,%.0f) seat1 spawn=(%.0f,%.0f) separation=%.0f\n",
               ep, s0.x, s0.y, s1.x, s1.y, sep);

        double rew0 = 0.0, rew1 = 0.0;
        int prev_deaths = s0.deaths + s1.deaths;

        for (int t = 0; t < ep_tics; t++)
        {
            doomrl_get_player_state(0, &s0);
            doomrl_get_player_state(1, &s1);

            if (duel && (s0.deaths + s1.deaths) != prev_deaths)
            {
                place_for_duel();
                prev_deaths = s0.deaths + s1.deaths;
            }

            doomrl_action_t a0, a1;
            hunter_action(0, &s0, &a0);
            hunter_action(1, &s1, &a1);

            doomrl_dm_step(&a0, &a1);
            grand_tics++;

            doomrl_get_player_state(0, &s0);
            doomrl_get_player_state(1, &s1);
            rew0 += s0.reward;
            rew1 += s1.reward;
            if (s0.opponent_visible) saw_total++;
            if (s1.opponent_visible) saw_total++;

            if (print_every > 0 && (t % print_every) == 0)
            {
                printf("  t=%4d seat0[hp=%3d frags=%d deaths=%d vis=%d bear=%6.1f dist=%5.0f rew=%+.2f] "
                       "seat1[hp=%3d frags=%d deaths=%d vis=%d]\n",
                       t, s0.health, s0.frags, s0.deaths, s0.opponent_visible, s0.opp_bearing_deg, s0.opp_dist, s0.reward,
                       s1.health, s1.frags, s1.deaths, s1.opponent_visible);
                printf("        items rocket[av=%d resp=%4.1f bear=%6.1f dist=%5.0f] "
                       "armor[av=%d resp=%4.1f] soul[av=%d resp=%4.1f]  "
                       "seat0[atype=%d rw=%d misl=%d]\n",
                       s0.key_items[0].available, s0.key_items[0].respawn_secs,
                       s0.key_items[0].bearing_deg, s0.key_items[0].dist,
                       s0.key_items[1].available, s0.key_items[1].respawn_secs,
                       s0.key_items[2].available, s0.key_items[2].respawn_secs,
                       s0.armortype, s0.ready_weapon, s0.ammo[3]);
            }
        }

        doomrl_get_player_state(0, &s0);
        doomrl_get_player_state(1, &s1);
        printf("ep%d end: seat0[frags=%d deaths=%d cum_reward=%+.2f] seat1[frags=%d deaths=%d cum_reward=%+.2f]\n",
               ep, s0.frags, s0.deaths, rew0, s1.frags, s1.deaths, rew1);
        grand_frags += s0.frags + s1.frags;
        grand_deaths += s0.deaths + s1.deaths;
    }

    double elapsed = now_seconds() - t0;
    printf("\nTOTAL: %d episodes, %d tics, frags=%d deaths=%d, opp-visible player-tics=%ld\n",
           episodes, grand_tics, grand_frags, grand_deaths, saw_total);
    printf("perf: %.0f tics/sec | %.2f frags/sec | %.2f deaths/sec | reset loops cleanly\n",
           grand_tics / elapsed, grand_frags / elapsed, grand_deaths / elapsed);

    return 0;
}
