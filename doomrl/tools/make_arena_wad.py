#!/usr/bin/env python3
"""Generate the strategic 1v1 "dumbbell" deathmatch arena as a PWAD.

Two end pockets joined by a central hall (the dumbbell long axis is x). Contested
power items sit in the open: the ROCKET LAUNCHER (2003) on a slightly SUNKEN
central altar, the BLUE MEGAARMOR (2019) on a RAISED ledge in the opposite
corner, and a SOULSPHERE (2013) in a pocket. Two SOLID COVER BLOCKS break line of
sight across the hall; a CHOKE narrows one hall mouth. Starter weapons (chaingun
2002 / shotgun 2001) sit near the spawns. Four inward-facing DM starts (type 11).
Shareware E1 things only (no SSG).

Unlike the old flat box this is a multi-sector, non-convex map with height
variation and interior voids, so the BSP is built by tools/nodebuild.py (a real
node builder) — the hand-faked single-subsector hack cannot represent it.

Replaces E1M1 when loaded after the IWAD (-file). Run with `-altdeath` so the
items respawn on 30 s timers.
"""

import sys

import nodebuild

# ---- textures / flats present in shareware doom1.wad ----
WALL = b"STARTAN3"
STEP = b"STEP1"        # step / riser texture (lower/upper)
FLOOR = b"FLOOR4_8"
FLOOR_ALT = b"FLAT5_4"
CEIL = b"CEIL3_5"
LIGHT = 200

ARENA_HALF = 1024
CEIL_Z = 192

# floor heights
Z_MAIN = 0
Z_ALTAR = -24      # rocket launcher sunken altar
Z_LEDGE = 40       # megaarmor raised ledge


class MapBuilder:
    def __init__(self):
        self.verts = []
        self.vmap = {}
        self.linedefs = []
        self.sidedefs = []
        self.sectors = []
        self.things = []

    def v(self, x, y):
        x, y = int(x), int(y)
        k = (x, y)
        i = self.vmap.get(k)
        if i is None:
            i = len(self.verts)
            self.verts.append((x, y))
            self.vmap[k] = i
        return i

    def sector(self, floor, ceil, ftex=FLOOR, ctex=CEIL, light=LIGHT, special=0, tag=0):
        self.sectors.append(dict(floor=floor, ceil=ceil, floortex=ftex, ceiltex=ctex,
                                 light=light, special=special, tag=tag))
        return len(self.sectors) - 1

    def sidedef(self, sector, upper=b"-", lower=b"-", middle=b"-", xoff=0, yoff=0):
        self.sidedefs.append(dict(xoff=xoff, yoff=yoff, upper=upper, lower=lower,
                                  middle=middle, sector=sector))
        return len(self.sidedefs) - 1

    def line(self, x1, y1, x2, y2, front_sd, back_sd=None, flags=None, special=0, tag=0):
        if flags is None:
            flags = 0x0001 if back_sd is None else 0x0004  # blocking / two-sided
        self.linedefs.append(dict(
            v1=self.v(x1, y1), v2=self.v(x2, y2), flags=flags, special=special, tag=tag,
            front=front_sd, back=back_sd))

    def one_sided_loop(self, pts, sector, tex=WALL):
        """A closed loop of one-sided walls bounding `sector` (front faces in)."""
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            sd = self.sidedef(sector, middle=tex)
            self.line(x1, y1, x2, y2, sd)

    def two_sided_loop(self, pts, inner_sector, outer_sector,
                       lower=STEP, upper=STEP):
        """A closed loop separating an inner sub-sector (e.g. altar/ledge) from
        the surrounding play sector. Front sidedef -> inner, back -> outer. The
        step riser shows on lower (sunken) or upper (raised)."""
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            front = self.sidedef(inner_sector, lower=lower, upper=upper)
            back = self.sidedef(outer_sector, lower=lower, upper=upper)
            self.line(x1, y1, x2, y2, front, back, flags=0x0004)

    def void_block(self, cx, cy, hw, hh, sector, tex=WALL):
        """A solid pillar / cover block: an inner square of one-sided walls wound
        CLOCKWISE so the front sidedefs face OUTWARD into `sector` (the block's
        interior is void / solid)."""
        pts = [(cx - hw, cy - hh), (cx - hw, cy + hh),
               (cx + hw, cy + hh), (cx + hw, cy - hh)]  # CW
        self.one_sided_loop(pts, sector, tex)

    def thing(self, x, y, angle, type_, flags=0x07):
        self.things.append(dict(x=int(x), y=int(y), angle=int(angle), type=int(type_), flags=flags))


def build():
    m = MapBuilder()
    main = m.sector(Z_MAIN, CEIL_Z, ftex=FLOOR, ctex=CEIL)

    # --- Outer boundary: a dumbbell. Two end pockets (wide) joined by a hall
    # (narrow in y). One end is offset/asymmetric on purpose. Traced CCW so the
    # one-sided front faces inward. Coordinates in map units. ---
    H = ARENA_HALF
    POCKET_HY = 640          # pocket half-height (y)
    HALL_HY = 320            # hall half-height (narrower -> the hall)
    PX = 1024                # pocket outer x extent
    HALLX = 360              # hall starts at +/- this x
    CHOKE_HY = 200           # the choke narrows the LEFT hall mouth further

    # Left pocket is taller/offset (asymmetric): shift its top up.
    # Outline vertices, CCW starting bottom-left.
    outline = [
        (-PX, -POCKET_HY),               # bottom-left pocket corner
        (-HALLX - 220, -POCKET_HY),
        (-HALLX, -CHOKE_HY),             # choke: hall mouth pinched on the left
        (HALLX, -HALL_HY),
        (PX, -POCKET_HY),                # bottom-right
        (PX, POCKET_HY),                 # top-right
        (HALLX, HALL_HY),
        (-HALLX, CHOKE_HY),              # choke top
        (-HALLX - 220, POCKET_HY + 96),  # left pocket taller (asymmetric)
        (-PX, POCKET_HY + 96),
    ]
    m.one_sided_loop(outline, main, WALL)

    # --- Sunken central altar holding the ROCKET LAUNCHER (contested, open). ---
    altar = m.sector(Z_ALTAR, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=170)
    ahw, ahh = 128, 128
    altar_pts = [(-ahw, -ahh), (-ahw, ahh), (ahw, ahh), (ahw, -ahh)]  # CCW (inner)
    m.two_sided_loop(altar_pts, altar, main, lower=STEP, upper=STEP)
    m.thing(0, 0, 0, 2003)  # rocket launcher on the altar

    # --- Raised ledge in the FAR (right-top) corner holding the MEGAARMOR. ---
    ledge = m.sector(Z_LEDGE, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=210)
    lx, ly = PX - 224, POCKET_HY - 224
    lhw, lhh = 128, 128
    ledge_pts = [(lx - lhw, ly - lhh), (lx - lhw, ly + lhh),
                 (lx + lhw, ly + lhh), (lx + lhw, ly - lhh)]  # CCW
    m.two_sided_loop(ledge_pts, ledge, main, lower=STEP, upper=STEP)
    m.thing(lx, ly, 0, 2019)  # blue megaarmor on the ledge

    # --- Soulsphere in the LEFT pocket (the third timed objective). ---
    m.thing(-PX + 256, POCKET_HY - 160, 0, 2013)

    # --- Two SOLID COVER BLOCKS in the hall that break line of sight. ---
    m.void_block(-150, 150, 70, 70, main, WALL)
    m.void_block(150, -150, 70, 70, main, WALL)

    # --- Starter weapons near the spawns. ---
    m.thing(-PX + 200, -POCKET_HY + 200, 0, 2002)   # chaingun, left-bottom spawn
    m.thing(PX - 200, -POCKET_HY + 200, 0, 2001)    # shotgun, right-bottom spawn
    m.thing(-PX + 200, POCKET_HY - 96, 0, 2001)     # shotgun, left-top
    m.thing(PX - 320, -POCKET_HY + 200, 0, 2002)    # chaingun, right-bottom-2

    # --- Four inward-facing DM starts (type 11), in the pocket corners. ---
    starts = [
        (-PX + 200, -POCKET_HY + 200, 0),     # left-bottom, face +x (inward)
        (PX - 200, -POCKET_HY + 200, 180),    # right-bottom, face -x
        (PX - 200, POCKET_HY - 200, 180),     # right-top, face -x
        (-PX + 200, POCKET_HY - 100, 0),      # left-top, face +x
    ]
    for x, y, ang in starts:
        m.thing(x, y, ang, 11)
    # A single-player start (type 1) so the map is valid standalone.
    m.thing(0, -HALL_HY + 64, 90, 1)

    return m


def write_wad(path, lumps_in_order):
    import struct
    directory = []
    data = bytearray()

    def pad8(name):
        if isinstance(name, str):
            name = name.encode()
        return name[:8].ljust(8, b"\x00")

    def add_lump(name, payload):
        offset = 12 + len(data)
        data.extend(payload)
        directory.append((offset, len(payload), pad8(name)))

    add_lump("E1M1", b"")
    for name, payload in lumps_in_order:
        add_lump(name, payload)

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
    out = sys.argv[1] if len(sys.argv) > 1 else "dumbbell.wad"
    m = build()
    lumps, stats = nodebuild.pack_map_lumps(
        m.verts, m.linedefs, m.sidedefs, m.sectors, m.things)
    write_wad(out, lumps)
    n_dm = sum(1 for t in m.things if t["type"] == 11)
    print(f"wrote {out}: sectors={len(m.sectors)} linedefs={len(m.linedefs)} "
          f"things={len(m.things)} dm_starts={n_dm}")
    print(f"  bsp: nodes={stats['nodes']} ssectors={stats['ssectors']} "
          f"segs={stats['segs']} verts={stats['verts']}")


if __name__ == "__main__":
    main()
