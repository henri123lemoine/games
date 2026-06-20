#!/usr/bin/env bash
# Run the doomtrain binary with libtorch on the dylib path. tch's
# download-libtorch drops the dylibs in the build output dir but does not embed
# an rpath (same as azt), so point DYLD at it here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-release}"

cargo build --profile "$PROFILE" --manifest-path "$HERE/Cargo.toml" >&2

LIBTORCH_LIB="$(dirname "$(find "$HERE/target" -name libtorch_cpu.dylib | head -1)")"
if [ -z "$LIBTORCH_LIB" ]; then
  echo "could not locate libtorch_cpu.dylib under target/" >&2
  exit 1
fi

BIN_DIR="release"
[ "$PROFILE" = "dev" ] && BIN_DIR="debug"

exec env DYLD_LIBRARY_PATH="$LIBTORCH_LIB:${DYLD_LIBRARY_PATH:-}" \
  "$HERE/target/$BIN_DIR/doomtrain" "$@"
