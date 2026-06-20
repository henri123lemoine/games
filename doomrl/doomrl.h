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
} doomrl_enemy_t;

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

    int   num_enemies;
    doomrl_enemy_t enemies[DOOMRL_MAX_ENEMIES];
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

#ifdef __cplusplus
}
#endif

#endif
