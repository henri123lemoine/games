#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DG="$HERE/vendor/doomgeneric/doomgeneric"
OUT="$HERE/build"
mkdir -p "$OUT"

CC="${CC:-clang}"
CFLAGS="${CFLAGS:--O2 -Wall -DNORMALUNIX -D_DEFAULT_SOURCE}"
CFLAGS="$CFLAGS -Wno-deprecated-non-prototype -Wno-format -Wno-parentheses"

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
  $CC $CFLAGS -I"$DG" -c "$DG/$f.c" -o "$obj"
  OBJS+=("$obj")
done

for f in doomrl doomrl_sound_null; do
  obj="$OUT/$f.o"
  $CC $CFLAGS -I"$DG" -c "$HERE/$f.c" -o "$obj"
  OBJS+=("$obj")
done

$CC $CFLAGS -I"$DG" -c "$HERE/driver.c" -o "$OUT/driver.o"
$CC $CFLAGS -I"$DG" -c "$HERE/dm_driver.c" -o "$OUT/dm_driver.o"

$CC $CFLAGS "${OBJS[@]}" "$OUT/driver.o" -lm -o "$OUT/doomrl_driver"
echo "built: $OUT/doomrl_driver"

$CC $CFLAGS "${OBJS[@]}" "$OUT/dm_driver.o" -lm -o "$OUT/doomrl_dm"
echo "built: $OUT/doomrl_dm"
