#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "doomrl.h"

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static unsigned int rng_state = 1u;
static unsigned int xorshift(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char **argv)
{
    int total_tics = 2100;
    int print_every = 35;
    int benchmark = 0;
    unsigned int seed = 1u;

    for (int i = 1; i < argc; i++)
    {
        if (!strncmp(argv[i], "--tics=", 7))
            total_tics = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--print-every=", 14))
            print_every = atoi(argv[i] + 14);
        else if (!strncmp(argv[i], "--seed=", 7))
            seed = (unsigned int)strtoul(argv[i] + 7, NULL, 10);
        else if (!strcmp(argv[i], "--bench"))
            benchmark = 1;
    }
    rng_state = seed ? seed : 1u;

    char *doom_argv[] = {
        "doomrl",
        "-iwad", "../web/app/public/doom/doom1.wad",
        "-warp", "1", "1",
        "-skill", "3",
        "-nomusic", "-nosound", "-nodraw",
    };
    int doom_argc = (int)(sizeof(doom_argv) / sizeof(doom_argv[0]));

    doomrl_init(doom_argc, doom_argv);

    doomrl_state_t st;
    doomrl_get_state(&st);
    printf("init: tic=%d gamestate=%d pos=(%.1f,%.1f,%.1f) ang=%.1f hp=%d arm=%d weap=%d totalkills=%d alive=%d\n",
           st.tic, st.gamestate, st.x, st.y, st.z, st.angle_deg,
           st.health, st.armor, st.ready_weapon, st.total_kills, st.alive);
    printf("       enemies=%d ammo=[%d %d %d %d]\n",
           st.num_enemies, st.ammo[0], st.ammo[1], st.ammo[2], st.ammo[3]);

    double t0 = now_seconds();

    for (int t = 0; t < total_tics; t++)
    {
        doomrl_action_t a;
        memset(&a, 0, sizeof(a));

        a.forward = 50;
        if ((t / 35) % 4 == 1)
            a.turn = 600;
        else if ((t / 35) % 4 == 3)
            a.turn = -600;
        if ((xorshift() & 7) == 0)
            a.fire = 1;

        doomrl_step(&a);

        if (!benchmark && print_every > 0 && (t % print_every) == 0)
        {
            doomrl_get_state(&st);
            printf("t=%4d tic=%4d pos=(%.1f,%.1f) ang=%6.1f hp=%3d arm=%3d kills=%d/%d items=%d enemies=%d alive=%d gs=%d\n",
                   t, st.tic, st.x, st.y, st.angle_deg, st.health, st.armor,
                   st.killcount, st.total_kills, st.itemcount, st.num_enemies, st.alive, st.gamestate);

            int show = st.num_enemies < 3 ? st.num_enemies : 3;
            for (int e = 0; e < show; e++)
            {
                doomrl_enemy_t *en = &st.enemies[e];
                printf("        enemy[%d] type=%d pos=(%.1f,%.1f) hp=%d dist=%.0f awake=%d\n",
                       e, en->type, en->x, en->y, en->health, en->dist, en->awake);
            }
        }
    }

    double t1 = now_seconds();
    double elapsed = t1 - t0;

    doomrl_get_state(&st);
    printf("final: tic=%d pos=(%.1f,%.1f) ang=%.1f hp=%d kills=%d/%d items=%d secrets=%d alive=%d gs=%d\n",
           st.tic, st.x, st.y, st.angle_deg, st.health,
           st.killcount, st.total_kills, st.itemcount, st.secretcount, st.alive, st.gamestate);
    printf("perf: %d tics in %.4f s = %.0f tics/sec (%.2fx realtime @35Hz)\n",
           total_tics, elapsed, total_tics / elapsed, (total_tics / elapsed) / 35.0);

    return 0;
}
