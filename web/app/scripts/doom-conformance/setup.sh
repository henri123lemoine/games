#!/usr/bin/env bash
# Populate the harness sandbox from the shipping DOOM build. The engine binaries,
# the WAD, and the extracted demo lumps are large and/or derivable, so they are
# not committed — this script reconstructs them next to the committed loader
# pages (doom-*.html) and configs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC="$HERE/../../public/doom"
SANDBOX="$HERE/sandbox"
mkdir -p "$SANDBOX"

cp "$PUBLIC/websockets-doom.js" "$SANDBOX/"
cp "$PUBLIC/websockets-doom.wasm" "$SANDBOX/"
cp "$PUBLIC/doom1.wad" "$SANDBOX/"

# Two config sets so the A/B isolates exactly the change:
#  - default-baseline.cfg: mouse ON, no extra config  -> reproduces both bugs
#    (the baseline/GL loader pages use this and no -extraconfig)
#  - default.cfg + websockets-doom.cfg: the shipping fix (mouse OFF, software
#    renderer) -> the soft loader pages use these via -config/-extraconfig
cp "$HERE/default-baseline.cfg" "$SANDBOX/default-baseline.cfg"
cp "$PUBLIC/default.cfg" "$SANDBOX/default.cfg"
cp "$PUBLIC/websockets-doom.cfg" "$SANDBOX/websockets-doom.cfg"

# Extract the WAD's built-in deterministic demos to .lmp files for -playdemo.
python3 - "$SANDBOX/doom1.wad" "$SANDBOX" <<'PY'
import struct, sys
wad = open(sys.argv[1], 'rb').read()
_, numlumps, dirofs = struct.unpack('<4sii', wad[:12])
for i in range(numlumps):
    off = dirofs + i * 16
    pos, size = struct.unpack('<ii', wad[off:off + 8])
    name = wad[off + 8:off + 16].split(b'\x00')[0].decode('ascii', 'replace')
    if name.upper().startswith('DEMO'):
        open(f"{sys.argv[2]}/{name.lower()}.lmp", 'wb').write(wad[pos:pos + size])
        print("extracted", name.lower() + ".lmp", size, "bytes")
PY

echo "sandbox ready at $SANDBOX"
