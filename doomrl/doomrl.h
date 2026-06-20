#ifndef DOOMRL_H
#define DOOMRL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DOOMRL_MAX_ENEMIES 64

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

#define DOOMRL_MAX_PLAYERS 2

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

    float reward;
} doomrl_player_state_t;

void doomrl_dm_init(int argc, char **argv);
void doomrl_dm_step(const doomrl_action_t *a0, const doomrl_action_t *a1);
void doomrl_get_player_state(int seat, doomrl_player_state_t *out);
void doomrl_reset(void);
void doomrl_dm_spawn_near(float dist);
int  doomrl_num_players(void);

#ifdef __cplusplus
}
#endif

#endif
