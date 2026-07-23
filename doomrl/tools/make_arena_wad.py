#!/usr/bin/env python3
"""Generate the strategic "Foundry" free-for-all arena as a PWAD.

An octagonal shell surrounds two fortified gatehouses, each with a center door
and north/south flanks. Contested power items occupy distinct routes: a ROCKET
LAUNCHER (2003) in the sunken central reactor, BLUE MEGAARMOR (2019) on a
three-tier climbable dais, and a SOULSPHERE (2013) in the lower west shrine.
Machinery and reactor shielding break the long sightlines; distributed weapons,
health, and armor keep every lane useful. Eight inset DM starts support up to
four active players. Shareware E1 things only (no SSG).

The map is multi-sector and non-convex with height variation and interior voids,
so its BSP is built by tools/nodebuild.py (a real node builder) — the old
hand-faked single-subsector approach cannot represent it.

Replaces E1M1 when loaded after the IWAD (-file). Run with `-altdeath` so the
items respawn on 30 s timers.
"""

import math
import sys

import nodebuild

# ---- textures / flats present in shareware doom1.wad ----
WALL = b"STARTAN3"
STEP = b"STEP1"        # step / riser texture (lower/upper)
FLOOR = b"FLOOR4_8"
FLOOR_ALT = b"FLAT5_4"
CEIL = b"CEIL3_5"
LIGHT = 200

ARENA_HALF = 1280
CEIL_Z = 192

# floor heights
Z_MAIN = 0


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

    @staticmethod
    def _clockwise(pts):
        area2 = sum(
            pts[i][0] * pts[(i + 1) % len(pts)][1]
            - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts))
        )
        if area2 == 0:
            raise ValueError("degenerate polygon loop")
        return area2 < 0

    def one_sided_loop(self, pts, sector, tex=WALL, sector_inside=True):
        """Add one-sided walls with `sector` on Doom's front (right) side.

        An outer boundary therefore winds clockwise; a solid void inside a
        sector winds counter-clockwise. Normalize here so a caller cannot
        accidentally create invisible back-facing walls and HOM artifacts.
        """
        pts = list(pts)
        if self._clockwise(pts) != sector_inside:
            pts.reverse()
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
        pts = list(pts)
        if not self._clockwise(pts):
            pts.reverse()
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            front = self.sidedef(inner_sector, lower=lower, upper=upper)
            back = self.sidedef(outer_sector, lower=lower, upper=upper)
            self.line(x1, y1, x2, y2, front, back, flags=0x0004)

    def void_block(self, cx, cy, hw, hh, sector, tex=WALL):
        """A solid pillar / cover block whose front sides face into `sector`."""
        pts = [(cx - hw, cy - hh), (cx - hw, cy + hh),
               (cx + hw, cy + hh), (cx + hw, cy - hh)]
        self.one_sided_loop(pts, sector, tex, sector_inside=False)

    def thing(self, x, y, angle, type_, flags=0x07):
        self.things.append(dict(x=int(x), y=int(y), angle=int(angle), type=int(type_), flags=flags))


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            edge_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < edge_x:
                inside = not inside
        j = i
    return inside


def _distance_to_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _validate_starts(starts, outline, clearance=72):
    for x, y, _angle in starts:
        if not _point_in_polygon((x, y), outline):
            raise ValueError(f"deathmatch start ({x}, {y}) is outside the arena")
        wall_distance = min(
            _distance_to_segment((x, y), outline[i], outline[(i + 1) % len(outline)])
            for i in range(len(outline))
        )
        if wall_distance < clearance:
            raise ValueError(
                f"deathmatch start ({x}, {y}) is only {wall_distance:.1f} units from an outer wall"
            )


def build():
    m = MapBuilder()
    main = m.sector(Z_MAIN, CEIL_Z, ftex=FLOOR, ctex=CEIL)

    # --- The Foundry: an octagonal shell containing three routes through each
    # gatehouse, a central reactor pit, two side shrines, and a stepped armor
    # dais. Clockwise normalization keeps every outer wall front-facing. ---
    outline = [
        (-1120, 768), (1120, 768), (1280, 608), (1280, -608),
        (1120, -768), (-1120, -768), (-1280, -608), (-1280, 608),
    ]
    m.one_sided_loop(outline, main, WALL)

    # West/east gatehouses split the arena into bases and a central foundry.
    # Each has a wide center door plus narrower north/south flanking routes.
    for x in (-640, 640):
        m.void_block(x, 430, 72, 190, main, WALL)
        m.void_block(x, -430, 72, 190, main, WALL)

    # Offset reactor shielding and side-route machinery break long sightlines
    # without sealing any lane.
    for cx, cy, hw, hh in [
        (-330, 300, 105, 90), (330, -300, 105, 90),
        (-930, 230, 90, 110), (930, -230, 90, 110),
        (0, 585, 170, 62), (0, -585, 170, 62),
    ]:
        m.void_block(cx, cy, hw, hh, main, WALL)

    # Central rocket reactor: deliberately sunken, but only one 16-unit step so
    # it is traversable in either direction under vanilla Doom movement rules.
    reactor = m.sector(-16, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=168)
    reactor_pts = [(-176, -144), (-176, 144), (176, 144), (176, -144)]
    m.two_sided_loop(reactor_pts, reactor, main, lower=STEP, upper=STEP)
    m.thing(0, 0, 0, 2003)

    # Three 16-unit tiers lead to the megaarmor. The former single 40-unit
    # ledge exceeded Doom's 24-unit step height and was impossible to mount.
    mx, my = 980, 430
    tier1 = m.sector(16, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=196)
    tier2 = m.sector(32, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=204)
    tier3 = m.sector(48, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=220)
    m.two_sided_loop([(760, 254), (760, 606), (1200, 606), (1200, 254)], tier1, main)
    m.two_sided_loop([(808, 302), (808, 558), (1152, 558), (1152, 302)], tier2, tier1)
    m.two_sided_loop([(856, 350), (856, 510), (1104, 510), (1104, 350)], tier3, tier2)
    m.thing(mx, my, 0, 2019)

    # A shallow lower shrine makes the soulsphere contest a separate route.
    sx, sy = -930, -500
    shrine = m.sector(-16, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=184)
    m.two_sided_loop([(-1090, -620), (-1090, -380), (-770, -380), (-770, -620)],
                     shrine, main)
    m.thing(sx, sy, 0, 2013)

    # A reachable southern supply plinth plus distributed weapons/health make
    # every route useful instead of forcing all play through the rocket pit.
    supply = m.sector(16, CEIL_Z, ftex=FLOOR_ALT, ctex=CEIL, light=202)
    m.two_sided_loop([(730, -610), (730, -390), (1010, -390), (1010, -610)],
                     supply, main)
    m.thing(870, -500, 0, 2018)  # green armor
    for x, y, thing_type in [
        (-980, 0, 2002), (980, 0, 2001),
        (-360, 520, 2001), (360, -520, 2002),
        (0, 420, 2012), (0, -420, 2012),
        (-860, 470, 2011), (860, -330, 2011),
    ]:
        m.thing(x, y, 0, thing_type)

    # Eight well-inset starts support four-player FFA and avoid the old invalid
    # right-top start, which sat beyond the sloped boundary and caused spawn HOMs.
    starts = [
        (-1100, -450, 0), (-1100, 450, 0),
        (1100, -450, 180), (1100, 680, 225),
        (-360, -660, 90), (360, -660, 90),
        (-360, 660, 270), (360, 660, 270),
    ]
    _validate_starts(starts, outline)
    for x, y, ang in starts:
        m.thing(x, y, ang, 11)
    m.thing(0, -300, 90, 1)

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
