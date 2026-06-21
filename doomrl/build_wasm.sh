#!/usr/bin/env bash
# Build the doomrl 1v1 deathmatch substrate to WebAssembly: renders seat 0's POV
# into DG_ScreenBuffer (JS blits to canvas), both seats' ticcmds driven from JS
# (human seat 0, RL bot seat 1). No SDL — uses doomgeneric's generic i_video +
# our null sound; JS pumps web_step() per tic.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DG="$HERE/vendor/doomgeneric/doomgeneric"
OUT="${1:-$HERE/build_wasm}"
mkdir -p "$OUT"

if ! command -v emcc >/dev/null 2>&1; then
  echo "emcc not on PATH — source the emsdk env first:" >&2
  echo "  source /private/tmp/emsdk/emsdk_env.sh" >&2
  exit 1
fi

IWAD="$HERE/../web/app/public/doom/doom1.wad"
ARENA="$HERE/assets/dumbbell.wad"
[ -f "$IWAD" ] || { echo "missing $IWAD" >&2; exit 1; }
[ -f "$ARENA" ] || { echo "missing $ARENA — run tools/make_arena_wad.py assets/dumbbell.wad" >&2; exit 1; }

CFLAGS="-O2 -DNORMALUNIX -D_DEFAULT_SOURCE -DDOOMRL_ALLOW_FILE_DEFAULT \
  -Wno-deprecated-non-prototype -Wno-format -Wno-parentheses -I$DG -I$HERE"

CORE=(
  dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main
  d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom
  i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox m_cheat
  m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy
  p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup
  p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw r_main
  r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound
  tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc
  i_input i_video doomgeneric
)

OBJS=()
for f in "${CORE[@]}"; do
  obj="$OUT/$f.o"
  emcc $CFLAGS -c "$DG/$f.c" -o "$obj"
  OBJS+=("$obj")
done
for f in doomrl doomrl_sound_null doomrl_web; do
  obj="$OUT/$f.o"
  emcc $CFLAGS -c "$HERE/$f.c" -o "$obj"
  OBJS+=("$obj")
done

EXPORTS='["_web_init","_web_set_action","_web_step","_web_spawn_near","_web_reset","_web_screenbuffer","_web_screen_w","_web_screen_h","_web_player_state","_malloc","_free"]'

emcc $CFLAGS "${OBJS[@]}" -lm \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 -s EXPORT_NAME=DoomRL -s EXPORT_ES6=1 \
  -s "EXPORTED_FUNCTIONS=$EXPORTS" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAPF32','HEAP32']" \
  -s INITIAL_MEMORY=67108864 \
  -s ENVIRONMENT=web \
  --preload-file "$IWAD@doom1.wad" \
  --preload-file "$ARENA@dumbbell.wad" \
  -o "$OUT/doomrl.js"

echo "built: $OUT/doomrl.js $OUT/doomrl.wasm $OUT/doomrl.data"
