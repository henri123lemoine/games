#!/usr/bin/env python3
"""Generate a minimal flat single-room deathmatch arena as a PWAD.

One convex rectangular sector at a single floor height with four 1-sided walls
and four deathmatch starts (thing type 11) in the corners. Vanilla Doom needs
prebuilt nodes; for one convex subsector the BSP is trivial (a single subsector,
the root node index is the subsector with the 0x8000 leaf bit). The IWAD
supplies the flats/textures named here (FLOOR4_8 / CEIL3_5 / STARTAN3 exist in
the shareware doom1.wad).

Replaces E1M1 when loaded after the IWAD (-merge / -file), so boot with
`-warp 1 1` against this PWAD as an extra file.
"""

import struct
import sys

HALF = 1024
FLOOR_Z = 0
CEIL_Z = 128
LIGHT = 200
FLOOR_TEX = b"FLOOR4_8"
CEIL_TEX = b"CEIL3_5\x00"
WALL_TEX = b"STARTAN3"


def pad8(name: bytes) -> bytes:
    return name[:8].ljust(8, b"\x00")


def build_map_lumps():
    # corners (CCW so the sector is on the front/right of each 1-sided line)
    verts = [(-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)]

    things = []
    # four deathmatch starts (type 11), inset from the corners, facing inward
    inset = HALF - 256
    starts = [
        (-inset, -inset, 45),
        (inset, -inset, 135),
        (inset, inset, 225),
        (-inset, inset, 315),
    ]
    for x, y, ang in starts:
        things.append(struct.pack("<hhHHH", x, y, ang, 11, 0x07))
    # also a single-player start (type 1) so the map is valid standalone
    things.append(struct.pack("<hhHHH", 0, 0, 90, 1, 0x07))
    things_lump = b"".join(things)

    vertexes_lump = b"".join(struct.pack("<hh", x, y) for x, y in verts)

    # one sector
    sectors_lump = struct.pack(
        "<hh8s8shhh", FLOOR_Z, CEIL_Z, pad8(FLOOR_TEX), pad8(CEIL_TEX), LIGHT, 0, 0
    )

    # four sidedefs, all referencing sector 0 with the wall texture as midtex
    sidedef = struct.pack("<hh8s8s8sh", 0, 0, pad8(b"-"), pad8(b"-"), pad8(WALL_TEX), 0)
    sidedefs_lump = sidedef * 4

    # four linedefs forming the loop; 1-sided (back = 0xFFFF), impassable+blocking
    linedefs = []
    for i in range(4):
        v1 = i
        v2 = (i + 1) % 4
        flags = 0x0001  # ML_BLOCKING (impassable, 1-sided)
        linedefs.append(struct.pack("<HHHHHHH", v1, v2, flags, 0, 0, i, 0xFFFF))
    linedefs_lump = b"".join(linedefs)

    # one seg per linedef. seg: v1, v2, angle, linedef, side(0=front), offset
    segs = []
    angles = {0: 0, 1: 0x4000, 2: -0x8000 & 0xFFFF, 3: -0x4000 & 0xFFFF}
    for i in range(4):
        v1 = i
        v2 = (i + 1) % 4
        segs.append(struct.pack("<HHHHHH", v1, v2, angles[i], i, 0, 0))
    segs_lump = b"".join(segs)

    # one subsector covering all four segs
    ssectors_lump = struct.pack("<HH", 4, 0)

    # NODES: with a single subsector there is no split node. The reference
    # vanilla behaviour is an empty NODES lump; R_PointInSubsector treats the
    # root (numnodes-1 with no nodes) as subsector 0 via the 0x8000 leaf bit.
    # Provide one degenerate node whose both children point at subsector 0 so
    # P_LoadNodes has a root to start from.
    # node: x,y,dx,dy, bbox[2][4], child[2]
    nodes_lump = struct.pack(
        "<hhhh" + "hhhh" * 2 + "HH",
        -HALF, 0, 0, 1,
        HALF, -HALF, -HALF, HALF,
        HALF, -HALF, -HALF, HALF,
        0x8000, 0x8000,
    )

    # REJECT: 1 sector -> ceil(1*1/8)=1 byte, all visible (0)
    reject_lump = b"\x00"

    blockmap_lump = build_blockmap(linedefs, verts)

    return {
        "THINGS": things_lump,
        "LINEDEFS": linedefs_lump,
        "SIDEDEFS": sidedefs_lump,
        "VERTEXES": vertexes_lump,
        "SEGS": segs_lump,
        "SSECTORS": ssectors_lump,
        "NODES": nodes_lump,
        "SECTORS": sectors_lump,
        "REJECT": reject_lump,
        "BLOCKMAP": blockmap_lump,
    }


def build_blockmap(linedefs, verts):
    # Doom links mobjs/lines into 128-unit blocks indexed from (bmaporgx,
    # bmaporgy). The map must span the whole room or P_PathTraverse /
    # P_BlockThingsIterator miss mobjs in unlisted blocks.
    BLOCK = 128
    origin_x = -HALF - BLOCK
    origin_y = -HALF - BLOCK
    span = 2 * HALF + 2 * BLOCK
    cols = (span + BLOCK - 1) // BLOCK
    rows = cols
    header = struct.pack("<hhHH", origin_x, origin_y, cols, rows)

    n_blocks = cols * rows
    head_words = 4
    # Put every wall linedef in every block's list (cheap and correct for a
    # tiny single room; the four walls bound the whole arena). Empty would also
    # work for interior blocks, but listing all keeps wall collision exact.
    line_indices = list(range(len(linedefs)))
    blocklist = [0x0000] + line_indices + [0xFFFF]
    block_words = len(blocklist)

    offsets = []
    first = head_words + n_blocks
    for b in range(n_blocks):
        offsets.append(first + b * block_words)

    offsets_bytes = struct.pack("<%dH" % n_blocks, *offsets)
    body = struct.pack("<%dH" % block_words, *blocklist) * n_blocks
    return header + offsets_bytes + body


def write_wad(path, lumps_in_order):
    # PWAD with a single map: marker E1M1 then the 10 map lumps
    directory = []
    data = bytearray()

    def add_lump(name, payload):
        offset = 12 + len(data)
        data.extend(payload)
        directory.append((offset, len(payload), pad8(name)))

    add_lump(b"E1M1", b"")
    for name, payload in lumps_in_order:
        add_lump(name.encode(), payload)

    dir_offset = 12 + len(data)
    header = struct.pack("<4sii", b"PWAD", len(directory), dir_offset)

    dir_bytes = bytearray()
    for offset, size, name in directory:
        dir_bytes.extend(struct.pack("<ii8s", offset, size, name))

    with open(path, "wb") as f:
        f.write(header)
        f.write(data)
        f.write(dir_bytes)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "flatarena.wad"
    lumps = build_map_lumps()
    order = [
        "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
        "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP",
    ]
    write_wad(out, [(n, lumps[n]) for n in order])
    print(f"wrote {out}: 1 sector arena, 4 DM starts, floor_z={FLOOR_Z}")


if __name__ == "__main__":
    main()
