#ifndef DOOMRL_H
#define DOOMRL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DOOMRL_MAX_ENEMIES 64

/* Strategic 1v1: the three contested timed power items, fixed order:
 * 0 = rocket launcher (2003), 1 = blue megaarmor (2019), 2 = soulsphere (2013). */
#define DOOMRL_NUM_KEY_ITEMS 3

typedef struct {
    int   available;       /* 1 if the pickup mobj is on the map right now */
    float respawn_secs;    /* seconds until it respawns (0 when available)  */
    float bearing_deg;     /* egocentric bearing to its fixed map position  */
    float dist;            /* planar distance to its fixed map position     */
} doomrl_key_item_t;

typedef struct {
    int   type;
    float x;
    float y;
    float z;
    float angle_deg;
    int   health;
    float dist;
    int   awake;
    float bearing_deg;
    float rel_vx;
    float rel_vy;
} doomrl_enemy_t;

typedef struct {
    int   type;
    int   valid;
    int   ticks_since_seen;
    float last_bearing_deg;
    float last_dist;
} doomrl_target_memory_t;

typedef struct {
    int   tic;
    int   gamestate;
    int   level_done;

    float x;
    float y;
    float z;
    float angle_deg;
    float momx;
    float momy;

    int   health;
    int   armor;
    int   armortype;
    int   ready_weapon;
    int   ammo[4];
    int   killcount;
    int   itemcount;
    int   secretcount;
    int   total_kills;
    int   total_items;
    int   total_secrets;
    int   alive;

    int   num_visible_enemies;
    doomrl_enemy_t enemies[DOOMRL_MAX_ENEMIES];

    doomrl_target_memory_t target;
} doomrl_state_t;

typedef struct {
    int8_t  forward;
    int8_t  side;
    int16_t turn;
    uint8_t fire;
    uint8_t use;
    uint8_t weapon;
} doomrl_action_t;

void doomrl_init(int argc, char **argv);
void doomrl_step(const doomrl_action_t *action);
void doomrl_get_state(doomrl_state_t *out);
int  doomrl_tic(void);

#define DOOMRL_MAX_PLAYERS 4

typedef struct {
    int   seat;
    int   alive;

    float x;
    float y;
    float z;
    float angle_deg;
    float momx;
    float momy;

    int   health;
    int   armor;
    int   armortype;
    int   ready_weapon;
    int   ammo[4];
    int   frags;
    int   deaths;

    int   opponent_visible;
    float opp_bearing_deg;
    float opp_dist;
    float opp_rel_vx;
    float opp_rel_vy;
    int   opp_health;

    doomrl_target_memory_t opp_memory;

    doomrl_key_item_t key_items[DOOMRL_NUM_KEY_ITEMS];

    float reward;
} doomrl_player_state_t;

void doomrl_dm_init(int argc, char **argv);
void doomrl_dm_init_players(int argc, char **argv, int num_players);
void doomrl_dm_step(const doomrl_action_t *a0, const doomrl_action_t *a1);
void doomrl_dm_step_all(const doomrl_action_t actions[DOOMRL_MAX_PLAYERS]);
void doomrl_get_player_state(int seat, doomrl_player_state_t *out);
void doomrl_reset(void);
int  doomrl_dm_set_player_count(int num_players);
void doomrl_dm_spawn_near(float dist);
int  doomrl_num_players(void);

#ifdef __cplusplus
}
#endif

#endif
