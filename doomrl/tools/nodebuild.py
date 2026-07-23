#!/usr/bin/env python3
"""Self-contained vanilla-Doom BSP node builder.

Takes a map's VERTEXES / LINEDEFS / SIDEDEFS / SECTORS and produces the derived
lumps the engine needs: SEGS, SSECTORS, NODES, BLOCKMAP, REJECT. Pure Python, no
external nodebuilder binary — deterministic and fully under our control, which is
required the moment the map stops being one convex sector (cover blocks, a sunken
altar, a raised ledge all force real BSP partitioning).

Output is the classic (pre-ZDBSP) vanilla binary node format that doomgeneric's
P_LoadNodes / P_LoadSegs / P_LoadSubsectors read directly.

Geometry model:
  vertices : list[(x,y)] ints
  linedefs : list[dict(v1,v2,flags,special,tag,front,back)]  (front/back = sidedef idx or None)
  sidedefs : list[dict(xoff,yoff,upper,lower,middle,sector)]
  sectors  : list[dict(floor,ceil,floortex,ceiltex,light,special,tag)]
"""

import struct

NF_SUBSECTOR = 0x8000


def _pad8(name):
    if isinstance(name, str):
        name = name.encode()
    return name[:8].ljust(8, b"\x00")


class Seg:
    """A directed wall segment along one side of a linedef."""

    __slots__ = ("v1", "v2", "linedef", "side", "offset", "sector", "x1", "y1", "x2", "y2")

    def __init__(self, v1, v2, linedef, side, offset, sector, verts):
        self.v1 = v1
        self.v2 = v2
        self.linedef = linedef
        self.side = side
        self.offset = offset
        self.sector = sector
        self.x1, self.y1 = verts[v1]
        self.x2, self.y2 = verts[v2]


def _seg_angle_binam(x1, y1, x2, y2):
    """BAM angle of the seg direction (Doom's 16-bit angle, top word)."""
    import math

    a = math.atan2(y2 - y1, x2 - x1)
    if a < 0:
        a += 2 * math.pi
    return int(round(a / (2 * math.pi) * 65536.0)) & 0xFFFF


def build_segs(verts, linedefs, sidedefs):
    """One seg per sided side of each linedef (front always; back if two-sided)."""
    segs = []
    for li, ld in enumerate(linedefs):
        v1, v2 = ld["v1"], ld["v2"]
        if ld["front"] is not None:
            sec = sidedefs[ld["front"]]["sector"]
            segs.append(Seg(v1, v2, li, 0, 0, sec, verts))
        if ld["back"] is not None:
            sec = sidedefs[ld["back"]]["sector"]
            segs.append(Seg(v2, v1, li, 1, 0, sec, verts))
    return segs


def _side_of(px, py, dx, dy, x, y):
    """Cross product sign of point (x,y) relative to the directed line
    (px,py)->(px+dx,py+dy). >0 left, <0 right, 0 on the line."""
    return dx * (y - py) - dy * (x - px)


def _pick_splitter(segs):
    """Pick the seg whose supporting line best partitions the set. A valid
    splitter must make real progress: it must put at least one seg on each side
    (or split some). Boundary edges that leave one side empty are rejected so the
    recursion always shrinks. Among valid candidates, minimize |left-right| +
    8*splits and lightly prefer axis-aligned lines."""
    EPS = 1e-6
    best = None
    best_cost = None
    for i, s in enumerate(segs):
        dx = s.x2 - s.x1
        dy = s.y2 - s.y1
        if dx == 0 and dy == 0:
            continue
        # The candidate itself is collinear and same-facing, so it belongs to
        # the front/right child. Count it when checking that the split makes
        # progress; otherwise a valid two-sided boundary can be rejected just
        # because its opposite-facing twin is the only remaining seg.
        left = splits = 0
        right = 1
        for t in segs:
            if t is s:
                continue
            a = _side_of(s.x1, s.y1, dx, dy, t.x1, t.y1)
            b = _side_of(s.x1, s.y1, dx, dy, t.x2, t.y2)
            la = a > EPS
            ra = a < -EPS
            lb = b > EPS
            rb = b < -EPS
            if (la or lb) and (ra or rb):
                splits += 1
            elif la or lb:
                left += 1
            elif ra or rb:
                right += 1
            else:
                # Doom's front side is the RIGHT side of a directed seg. Two
                # opposite-facing segs on a two-sided line belong to opposite
                # BSP children even though their geometry is collinear.
                tdx = t.x2 - t.x1
                tdy = t.y2 - t.y1
                if dx * tdx + dy * tdy >= 0:
                    right += 1
                else:
                    left += 1
        # reject zero-progress splitters: one side empty and nothing split.
        if splits == 0 and (left == 0 or right == 0):
            continue
        axis_bonus = 0 if (dx == 0 or dy == 0) else 1
        cost = abs(left - right) + 8 * splits + axis_bonus
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = i
    return best


def _intersect(s, px, py, dx, dy, verts, vert_pool):
    """Split seg s by the line (px,py,dx,dy); return (new_vertex_index, t)."""
    sx, sy = s.x1, s.y1
    sdx, sdy = s.x2 - s.x1, s.y2 - s.y1
    denom = dx * sdy - dy * sdx
    # caller guarantees a real crossing, denom != 0
    t = (dx * (sy - py) - dy * (sx - px)) / (-denom)
    ix = sx + sdx * t
    iy = sy + sdy * t
    ixr, iyr = int(round(ix)), int(round(iy))
    key = (ixr, iyr)
    vi = vert_pool.get(key)
    if vi is None:
        vi = len(verts)
        verts.append((ixr, iyr))
        vert_pool[key] = vi
    return vi, t


def _bbox(segs):
    xs = [c for s in segs for c in (s.x1, s.x2)]
    ys = [c for s in segs for c in (s.y1, s.y2)]
    # NODES bbox order: top, bottom, left, right
    return [max(ys), min(ys), min(xs), max(xs)]


class _Builder:
    def __init__(self, verts, linedefs, sidedefs):
        self.verts = list(verts)
        self.linedefs = linedefs
        self.sidedefs = sidedefs
        self.vert_pool = {(x, y): i for i, (x, y) in enumerate(self.verts)}
        self.nodes = []
        self.ssectors = []
        self.out_segs = []

    def _emit_subsector(self, segs):
        if not segs:
            raise ValueError("cannot emit an empty BSP subsector")
        sectors = sorted({s.sector for s in segs})
        if len(sectors) != 1:
            raise ValueError(f"BSP subsector mixes sectors: {sectors}")
        first = len(self.out_segs)
        self.out_segs.extend(segs)
        idx = len(self.ssectors)
        self.ssectors.append((len(segs), first))
        return idx | NF_SUBSECTOR

    def _convex(self, segs):
        """Leaf test: the region is convex iff, for every seg's supporting line,
        ALL other endpoints fall on a single closed half-plane (all >=0 or all
        <=0). If any seg line has endpoints straddling it the region is concave
        and still needs partitioning. Orientation-agnostic, so it does not depend
        on input winding direction."""
        if len({s.sector for s in segs}) != 1:
            return False

        EPS = 1e-6
        for s in segs:
            dx = s.x2 - s.x1
            dy = s.y2 - s.y1
            if dx == 0 and dy == 0:
                continue
            saw_pos = saw_neg = False
            for t in segs:
                if t is s:
                    continue
                for (x, y) in ((t.x1, t.y1), (t.x2, t.y2)):
                    v = _side_of(s.x1, s.y1, dx, dy, x, y)
                    if v > EPS:
                        saw_pos = True
                    elif v < -EPS:
                        saw_neg = True
            if saw_pos and saw_neg:
                return False
        return True

    def build(self, segs):
        if self._convex(segs):
            return self._emit_subsector(segs)

        pi = _pick_splitter(segs)
        if pi is None:
            sectors = sorted({s.sector for s in segs})
            if len(sectors) != 1:
                details = [
                    (s.linedef, s.side, s.sector, s.x1, s.y1, s.x2, s.y2)
                    for s in segs
                ]
                raise ValueError(
                    f"could not separate BSP leaf sectors: {sectors}; segs={details}"
                )
            return self._emit_subsector(segs)
        part = segs[pi]
        px, py = part.x1, part.y1
        dx, dy = part.x2 - part.x1, part.y2 - part.y1

        left, right = [], []
        for s in segs:
            a = _side_of(px, py, dx, dy, s.x1, s.y1)
            b = _side_of(px, py, dx, dy, s.x2, s.y2)
            if a == 0 and b == 0:
                self._assign_collinear(part, s, left, right)
            elif a >= 0 and b >= 0:
                left.append(s)
            elif a <= 0 and b <= 0:
                right.append(s)
            else:
                # real split
                vi, _t = _intersect(s, px, py, dx, dy, self.verts, self.vert_pool)
                s_near, s_far = self._split_seg(s, vi)
                # s_near keeps s.v1..vi, s_far vi..s.v2; classify by midpoint side
                for piece in (s_near, s_far):
                    m = _side_of(px, py, dx, dy,
                                 (piece.x1 + piece.x2) / 2.0,
                                 (piece.y1 + piece.y2) / 2.0)
                    (left if m >= 0 else right).append(piece)

        if not left or not right:
            # degenerate partition; emit as leaf to avoid infinite recursion.
            return self._emit_subsector(segs)

        rbb = _bbox(right)
        lbb = _bbox(left)
        rchild = self.build(right)
        lchild = self.build(left)

        node = (
            int(round(px)), int(round(py)), int(round(dx)), int(round(dy)),
            rbb, lbb, rchild, lchild,
        )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _assign_collinear(self, part, s, left, right):
        # Doom's front side is right. Same-facing collinear segs go right;
        # opposite-facing sides of a two-sided line go left.
        pdx, pdy = part.x2 - part.x1, part.y2 - part.y1
        sdx, sdy = s.x2 - s.x1, s.y2 - s.y1
        if pdx * sdx + pdy * sdy >= 0:
            right.append(s)
        else:
            left.append(s)

    def _split_seg(self, s, vi):
        ix, iy = self.verts[vi]
        near = Seg(s.v1, vi, s.linedef, s.side, s.offset, s.sector, self.verts)
        # offset of far piece = original offset + distance along seg to split pt
        import math
        d = int(round(math.hypot(ix - s.x1, iy - s.y1)))
        far = Seg(vi, s.v2, s.linedef, s.side, s.offset + d, s.sector, self.verts)
        return near, far


def build_nodes(verts, linedefs, sidedefs):
    segs = build_segs(verts, linedefs, sidedefs)
    b = _Builder(verts, linedefs, sidedefs)
    root = b.build(segs)
    if not b.nodes:
        # single convex subsector: provide a degenerate root node both children
        # pointing at subsector 0, matching vanilla single-ssector handling.
        bb = _bbox(b.out_segs) if b.out_segs else [0, 0, 0, 0]
        b.nodes.append((0, 0, 0, 1, bb, bb, NF_SUBSECTOR, NF_SUBSECTOR))
        root = 0
    return b


def serialize(b, verts, linedefs):
    # VERTEXES
    vertexes = b"".join(struct.pack("<hh", x, y) for x, y in b.verts)

    # SEGS
    segs_b = bytearray()
    for s in b.out_segs:
        ang = _seg_angle_binam(s.x1, s.y1, s.x2, s.y2)
        segs_b += struct.pack("<HHHHHH", s.v1 & 0xFFFF, s.v2 & 0xFFFF, ang,
                              s.linedef & 0xFFFF, s.side & 0xFFFF, s.offset & 0xFFFF)

    # SSECTORS
    ssectors_b = b"".join(struct.pack("<HH", c, f) for (c, f) in b.ssectors)

    # NODES
    nodes_b = bytearray()
    for (x, y, dx, dy, rbb, lbb, rc, lc) in b.nodes:
        nodes_b += struct.pack("<hhhh", x, y, dx, dy)
        nodes_b += struct.pack("<hhhh", *[int(v) for v in rbb])
        nodes_b += struct.pack("<hhhh", *[int(v) for v in lbb])
        nodes_b += struct.pack("<HH", rc & 0xFFFF, lc & 0xFFFF)

    return vertexes, bytes(segs_b), ssectors_b, bytes(nodes_b)


def build_blockmap(verts, linedefs):
    """Block-link every linedef into each 128-unit block its bbox touches."""
    BLOCK = 128
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    origin_x = (minx & ~7) - BLOCK
    origin_y = (miny & ~7) - BLOCK
    cols = (maxx - origin_x) // BLOCK + 2
    rows = (maxy - origin_y) // BLOCK + 2

    n_blocks = cols * rows
    block_lines = [[] for _ in range(n_blocks)]
    for li, ld in enumerate(linedefs):
        x1, y1 = verts[ld["v1"]]
        x2, y2 = verts[ld["v2"]]
        bx1 = (min(x1, x2) - origin_x) // BLOCK
        bx2 = (max(x1, x2) - origin_x) // BLOCK
        by1 = (min(y1, y2) - origin_y) // BLOCK
        by2 = (max(y1, y2) - origin_y) // BLOCK
        for by in range(by1, by2 + 1):
            for bx in range(bx1, bx2 + 1):
                if 0 <= bx < cols and 0 <= by < rows:
                    block_lines[by * cols + bx].append(li)

    header = struct.pack("<hhHH", origin_x, origin_y, cols, rows)
    head_words = 4 + n_blocks
    offsets = []
    body = bytearray()
    cursor = head_words
    for b in range(n_blocks):
        offsets.append(cursor)
        words = [0] + block_lines[b] + [0xFFFF]
        body += struct.pack("<%dH" % len(words), *words)
        cursor += len(words)
    offsets_b = struct.pack("<%dH" % n_blocks, *offsets)
    return header + offsets_b + bytes(body)


def build_reject(n_sectors):
    """Conservative all-visible REJECT (every sector sees every sector)."""
    bits = n_sectors * n_sectors
    nbytes = (bits + 7) // 8
    return b"\x00" * nbytes


def validate_map_geometry(verts, linedefs, sidedefs):
    """Reject geometry that vanilla Doom would render as a HOM or bad BSP."""

    def orient(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    def intersects(a, b, c, d):
        oa, ob = orient(a, b, c), orient(a, b, d)
        oc, od = orient(c, d, a), orient(c, d, b)
        if oa == 0 and on_segment(a, b, c):
            return True
        if ob == 0 and on_segment(a, b, d):
            return True
        if oc == 0 and on_segment(c, d, a):
            return True
        if od == 0 and on_segment(c, d, b):
            return True
        return (oa > 0) != (ob > 0) and (oc > 0) != (od > 0)

    for i, ld in enumerate(linedefs):
        if ld["v1"] == ld["v2"] or verts[ld["v1"]] == verts[ld["v2"]]:
            raise ValueError(f"linedef {i} has zero length")
        if ld["front"] is None:
            raise ValueError(f"linedef {i} has no front sidedef")
        if ld["back"] is None:
            middle = sidedefs[ld["front"]]["middle"]
            if middle in (b"-", "-", b"", ""):
                raise ValueError(f"one-sided linedef {i} has no middle texture")

    for i, first in enumerate(linedefs):
        a, b = verts[first["v1"]], verts[first["v2"]]
        for j in range(i + 1, len(linedefs)):
            second = linedefs[j]
            # Adjacent map edges may meet at their shared vertex.
            if {first["v1"], first["v2"]} & {second["v1"], second["v2"]}:
                continue
            c, d = verts[second["v1"]], verts[second["v2"]]
            if intersects(a, b, c, d):
                raise ValueError(f"linedefs {i} and {j} intersect without a shared vertex")


def pack_map_lumps(verts, linedefs, sidedefs, sectors, things):
    """Return the 10 vanilla map lumps as an ordered list of (name, bytes)."""
    validate_map_geometry(verts, linedefs, sidedefs)
    things_b = b"".join(
        struct.pack("<hhHHH", t["x"], t["y"], t["angle"], t["type"], t["flags"])
        for t in things
    )
    linedefs_b = b"".join(
        struct.pack("<HHHHHHH",
                    ld["v1"] & 0xFFFF, ld["v2"] & 0xFFFF, ld["flags"] & 0xFFFF,
                    ld["special"] & 0xFFFF, ld["tag"] & 0xFFFF,
                    (ld["front"] if ld["front"] is not None else 0xFFFF) & 0xFFFF,
                    (ld["back"] if ld["back"] is not None else 0xFFFF) & 0xFFFF)
        for ld in linedefs
    )
    sidedefs_b = b"".join(
        struct.pack("<hh8s8s8sh",
                    sd["xoff"], sd["yoff"],
                    _pad8(sd["upper"]), _pad8(sd["lower"]), _pad8(sd["middle"]),
                    sd["sector"])
        for sd in sidedefs
    )
    sectors_b = b"".join(
        struct.pack("<hh8s8shhh",
                    sc["floor"], sc["ceil"],
                    _pad8(sc["floortex"]), _pad8(sc["ceiltex"]),
                    sc["light"], sc["special"], sc["tag"])
        for sc in sectors
    )

    b = build_nodes(verts, linedefs, sidedefs)
    vertexes_b, segs_b, ssectors_b, nodes_b = serialize(b, verts, linedefs)
    blockmap_b = build_blockmap(b.verts, linedefs)
    reject_b = build_reject(len(sectors))

    return [
        ("THINGS", things_b),
        ("LINEDEFS", linedefs_b),
        ("SIDEDEFS", sidedefs_b),
        ("VERTEXES", vertexes_b),
        ("SEGS", segs_b),
        ("SSECTORS", ssectors_b),
        ("NODES", nodes_b),
        ("SECTORS", sectors_b),
        ("REJECT", reject_b),
        ("BLOCKMAP", blockmap_b),
    ], {"nodes": len(b.nodes), "ssectors": len(b.ssectors), "segs": len(b.out_segs),
        "verts": len(b.verts)}
