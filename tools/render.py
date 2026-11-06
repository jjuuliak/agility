"""Render a course JSON (course.schema.json v1.0) to a PNG course drawing.

Usage:
  python render.py course.json [-o out.png] [--path] [--blind]

  --path   draw the dog path (the same model as the digitizing tool, incl. hand-set points)
  --blind  hide course id and source in the title

Obstacle footprints come from the FCI regulations (in force 1.7.2025), not from
the file. All coordinates are metres, origin top-left, y downwards.
"""
import argparse
import sys
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

INK = "#1f1f1f"
FILL = "#e9e6df"
CONTACT = "#e8a35c"
GRID_MINOR = "#eeeeee"
GRID_MAJOR = "#d6d6d6"
PATH = "#3b6fb6"

# Obstacle geometry in plan view (seen from above), from the SAGI/FCI obstacle rules.
# Sloped parts are projected onto the ground: plan length = sqrt(length^2 - height^2).
# Mid-range values are used where the rules give a range.
A_FRAME = {"ramp": 2.70, "apex": 1.70, "width_top": 0.90, "width_bottom": 1.15, "zone": 1.06}  # ramp 2.65-2.75 m; width >= 0.90, may widen to 1.15 at the bottom
DOG_WALK = {"plank": 3.70, "height": 1.25, "width": 0.30, "zone": 0.90}                       # planks 3.60-3.80 m; height 1.20-1.30 m
SEESAW = {"plank": 3.70, "width": 0.30, "zone": 0.90}                                          # drawn level
AF_HALF = math.sqrt(A_FRAME["ramp"] ** 2 - A_FRAME["apex"] ** 2)        # 2.10 m
DW_RAMP = math.sqrt(DOG_WALK["plank"] ** 2 - DOG_WALK["height"] ** 2)   # 3.48 m

JUMP = {"bar": 1.25, "wing": 0.50, "wing_depth": 0.12}   # bar 1.20-1.30 m, wings 0.40-0.60 m (hurdle and spread)
SPREAD_DEPTH = 0.50                                         # L size
WALL = {"width": 1.25, "thickness": 0.20, "tower": 0.30}    # 1.20-1.30 m; ~0.20 m thick at the bottom; towers 0.20-0.40 m (round 0.30-0.40)
TYRE = {"opening": 0.55, "ring": 0.10, "feet": 1.30, "frame": 0.85}  # opening 0.50-0.60 m; ring 0.08-0.12 m; feet 1.25-1.35 m along the direction of travel; frame uprights just outside the ring (estimated)
JUMP_WIDTH = JUMP["bar"] + 2 * JUMP["wing"]                 # 2.25 m

# depth = along heading (direction of travel), width = across it (widest point),
# half = distance from centre to the entry/exit point used for the dog path and labels.
DIMS = {
    "hurdle":    {"depth": JUMP["wing_depth"], "width": JUMP_WIDTH, "half": 0.30},
    "spread":    {"depth": SPREAD_DEPTH, "width": JUMP_WIDTH, "half": SPREAD_DEPTH / 2 + 0.20},
    "wall":      {"depth": WALL["tower"], "width": WALL["width"] + 2 * WALL["tower"], "half": 0.35},
    "tyre":      {"depth": TYRE["feet"], "width": TYRE["frame"], "half": 0.40},
    "long_jump": {"depth": 0.90, "width": 1.30, "half": 0.45},  # 0.9 m: middle of the SAGI range (XS 0.35-0.45 ... L 1.2-1.5)
    "weave":     {"depth": 7.04, "width": 0.20, "half": 3.52},  # 12 poles, 60 cm gap + 4 cm pole = 0.64 m centre to centre
    "a_frame":   {"depth": 2 * AF_HALF, "width": A_FRAME["width_bottom"], "half": AF_HALF},                              # 4.20 m
    "dog_walk":  {"depth": DOG_WALK["plank"] + 2 * DW_RAMP, "width": DOG_WALK["width"], "half": DOG_WALK["plank"] / 2 + DW_RAMP},  # 10.67 m
    "seesaw":    {"depth": SEESAW["plank"], "width": SEESAW["width"], "half": SEESAW["plank"] / 2},
}
TUNNEL_WIDTH = 0.60
# Contact zones are measured along the ramp in the rules; drawn as their plan-view projection.
CONTACT_ZONE = {
    "a_frame": A_FRAME["zone"] * AF_HALF / A_FRAME["ramp"],       # 1.06 m on the ramp -> 0.82 m in plan
    "dog_walk": DOG_WALK["zone"] * DW_RAMP / DOG_WALK["plank"],   # 0.90 m -> 0.85 m
    "seesaw": SEESAW["zone"],
}


def width_at(t, x):
    """Width at distance x from the centre along the heading (the A-frame widens towards the ground)."""
    if t != "a_frame":
        return DIMS[t]["width"]
    return A_FRAME["width_top"] + (A_FRAME["width_bottom"] - A_FRAME["width_top"]) * min(1.0, abs(x) / AF_HALF)


# ---------- geometry helpers ----------

def unit(heading_deg):
    """Direction of travel. Heading is clockwise from +x in y-down coordinates."""
    a = math.radians(heading_deg)
    return math.cos(a), math.sin(a)


def add(p, v, k=1.0):
    return (p[0] + k * v[0], p[1] + k * v[1])


def rect(c, u, depth, width):
    """Corners of a rectangle centred at c, `depth` along u and `width` across."""
    n = (-u[1], u[0])
    pts = []
    for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pts.append(add(add(c, u, su * depth / 2), n, sn * width / 2))
    return pts


def catmull_rom(points, samples=12):
    """Smooth polyline through all points (tunnel centre line)."""
    if len(points) < 3:
        return points
    pts = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                       + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                for k in (0, 1)))
    out.append(points[-1])
    return out


def offset_outline(line, half_width):
    """Polygon of a thick line (tunnel tube)."""
    left, right = [], []
    for i, p in enumerate(line):
        a = line[max(i - 1, 0)]
        b = line[min(i + 1, len(line) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        n = (-dy / length, dx / length)
        left.append(add(p, n, half_width))
        right.append(add(p, n, -half_width))
    return left + right[::-1]


def tangent_at(line, end):
    """Unit vector pointing out of the tunnel at its start ('a') or end ('b')."""
    if end == "a":
        p, q = line[0], line[1]
    else:
        p, q = line[-1], line[-2]
    dx, dy = p[0] - q[0], p[1] - q[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


# ---------- drawing ----------

def draw_placed(ax, o):
    t = o["type"]
    c = (o["x"], o["y"])
    u = unit(o["heading"])
    n = (-u[1], u[0])
    d = DIMS[t]

    def poly(pts, fill=FILL, lw=1.2, z=3):
        ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=INK, linewidth=lw, zorder=z))

    def line(p, q, lw=2.2, color=INK, z=4):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=lw, solid_capstyle="butt", zorder=z)

    def wings(at):  # wing panels in line with the bar, outside both ends
        for s in (-1, 1):
            poly(rect(add(at, n, s * (JUMP["bar"] + JUMP["wing"]) / 2), u, JUMP["wing_depth"], JUMP["wing"]), fill=FILL, lw=1.0)

    if t == "hurdle":
        line(add(c, n, -JUMP["bar"] / 2), add(c, n, JUMP["bar"] / 2))
        wings(c)
    elif t == "spread":
        for s in (-1, 1):
            b = add(c, u, s * SPREAD_DEPTH / 2)
            line(add(b, n, -JUMP["bar"] / 2), add(b, n, JUMP["bar"] / 2))
            wings(b)
    elif t == "wall":
        poly(rect(c, u, WALL["thickness"], WALL["width"]), fill="#b9b2a5")
        for s in (-1, 1):  # towers, part of the obstacle
            poly(rect(add(c, n, s * (WALL["width"] + WALL["tower"]) / 2), u, WALL["tower"], WALL["tower"]), fill=INK)
    elif t == "tyre":
        # seen from above: a ring (drawn as a circle so it is recognisable) on a frame with two feet along the travel direction
        for s in (-1, 1):
            f = add(c, n, s * TYRE["frame"] / 2)
            line(add(f, u, -TYRE["feet"] / 2), add(f, u, TYRE["feet"] / 2), lw=1.6)
        line(add(c, n, -TYRE["frame"] / 2), add(c, n, TYRE["frame"] / 2), lw=1.0)
        R = TYRE["opening"] / 2 + TYRE["ring"]
        ax.add_patch(Circle(c, R, facecolor="white", edgecolor=INK, linewidth=2.4, zorder=4))
        ax.add_patch(Circle(c, TYRE["opening"] / 2, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=5))
    elif t == "long_jump":
        for k in range(3):  # 3 boards, 0.9 m
            b = add(c, u, -0.375 + k * 0.375)
            poly(rect(b, u, 0.15, 1.2 - 0.05 * k), fill=FILL, lw=1.0)
    elif t == "weave":
        for k in range(12):
            p = add(c, u, -3.52 + k * 0.64)
            ax.add_patch(Circle(p, 0.09, facecolor=INK, edgecolor=INK, zorder=4))
    elif t in ("a_frame", "dog_walk", "seesaw"):
        def loc(x, y):
            return add(add(c, u, x), n, y)

        def band(x0, x1):
            return [loc(x0, -width_at(t, x0) / 2), loc(x1, -width_at(t, x1) / 2),
                    loc(x1, width_at(t, x1) / 2), loc(x0, width_at(t, x0) / 2)]

        h = d["depth"] / 2
        if t == "a_frame":
            wb, wt = width_at(t, h) / 2, A_FRAME["width_top"] / 2
            poly([loc(-h, -wb), loc(0, -wt), loc(h, -wb), loc(h, wb), loc(0, wt), loc(-h, wb)])
        else:
            poly(band(-h, h))
        z = CONTACT_ZONE[t]
        for s in (-1, 1):
            poly(band(s * h, s * (h - z)), fill=CONTACT, lw=0.8, z=4)
        if t == "a_frame":
            line(add(c, n, -A_FRAME["width_top"] / 2), add(c, n, A_FRAME["width_top"] / 2), lw=1.2)
        elif t == "dog_walk":
            for s in (-1, 1):
                j = add(c, u, s * DOG_WALK["plank"] / 2)
                line(add(j, n, -0.15), add(j, n, 0.15), lw=1.0)
        else:  # seesaw pivot
            ax.add_patch(Polygon([add(c, n, 0.4), add(add(c, n, 0.15), u, -0.25), add(add(c, n, 0.15), u, 0.25)],
                                 closed=True, facecolor=INK, edgecolor=INK, zorder=5))


def draw_tunnel(ax, o):
    raw = [(p["x"], p["y"]) for p in o["path"]]
    smooth = catmull_rom(raw)
    ax.add_patch(Polygon(offset_outline(smooth, TUNNEL_WIDTH / 2), closed=True,
                         facecolor=FILL, edgecolor=INK, linewidth=1.2, zorder=2))
    for end in ("a", "b"):  # mark the openings
        p = smooth[0] if end == "a" else smooth[-1]
        tx, ty = tangent_at(smooth, end)
        nx, ny = -ty, tx
        ax.plot([p[0] - nx * 0.33, p[0] + nx * 0.33], [p[1] - ny * 0.33, p[1] + ny * 0.33],
                color=INK, linewidth=2.4, zorder=3)
    return smooth


def entry_exit(o, direction, tunnel_lines):
    """Entry and exit point plus the 'outward at entry' unit vector."""
    if o["type"] == "tunnel":
        line = tunnel_lines[o["id"]]
        a, b = line[0], line[-1]
        ta, tb = tangent_at(line, "a"), tangent_at(line, "b")
        return (a, b, ta) if direction == "forward" else (b, a, tb)
    c = (o["x"], o["y"])
    u = unit(o["heading"])
    half = DIMS[o["type"]]["half"]
    entry, exit_ = add(c, u, -half), add(c, u, half)
    back = (-u[0], -u[1])
    if direction == "reverse":
        entry, exit_ = exit_, entry
        back = u
    return entry, exit_, back


def exit_dir(o, direction, tunnel_lines):
    """Unit vector of the dog's travel direction when leaving the obstacle."""
    if o["type"] == "tunnel":
        line = tunnel_lines[o["id"]]
        return tangent_at(line, "b") if direction == "forward" else tangent_at(line, "a")
    u = unit(o["heading"])
    return u if direction == "forward" else (-u[0], -u[1])


def bezier(p0, c0, c1, p1, samples=24):
    out = []
    for s in range(samples + 1):
        t = s / samples
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0[0] + b * c0[0] + c * c1[0] + d * p1[0], a * p0[1] + b * c0[1] + c * c1[1] + d * p1[1]))
    return out


# How far from straight-on (degrees) the dog may cross each jump. Estimates; adjust as needed.
JUMP_MAX_ANGLE = {"hurdle": 60, "spread": 10, "wall": 10, "tyre": 10, "long_jump": 10}
# SAGI: spread, tyre, long jump, dog walk and A-frame must be placed so that the dog can reach a
# straight approach line from the previous obstacle. The drawn path ends in a straight run-in.
STRAIGHT_APPROACH = {"spread", "tyre", "long_jump", "dog_walk", "a_frame", "wall"}  # wall: preferably straight
RUN_IN = 2.0           # metres of straight approach (shortened on short gaps)
WRAP_CLEARANCE = 0.6   # metres beyond the wing end when the dog turns around a jump
WEAVE_MAX_ANGLE = 75   # the dog may come into the first gap at an angle (degrees from the pole line)
HANDLE = 0.25          # curve roundness: handle length as a share of the gap
HANDLE_MAX = 4.5       # metres


def _norm(v):
    l = math.hypot(v[0], v[1])
    return (v[0] / l, v[1] / l) if l > 1e-9 else None


def _clamp_dir(base, want, max_deg):
    """Rotate `base` toward `want`, by at most max_deg."""
    ang = math.atan2(base[0] * want[1] - base[1] * want[0], base[0] * want[0] + base[1] * want[1])
    lim = math.radians(max_deg)
    ang = max(-lim, min(lim, ang))
    ca, sa = math.cos(ang), math.sin(ang)
    return (base[0] * ca - base[1] * sa, base[0] * sa + base[1] * ca)


def _seg_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _crosses(line, seg):
    return any(_seg_intersect(a, b, seg[0], seg[1]) for a, b in zip(line, line[1:]))


def _bar(o):
    """The jump's full width (bar plus wings) as a segment across its heading."""
    n = (-unit(o["heading"])[1], unit(o["heading"])[0])
    w = DIMS[o["type"]]["width"] / 2
    c = (o["x"], o["y"])
    return add(c, n, -w), add(c, n, w)


def _curve(points, tangents):
    """Smooth path through points with the given unit tangents (cubic pieces)."""
    out = []
    for (p0, t0), (p1, t1) in zip(zip(points, tangents), zip(points[1:], tangents[1:])):
        k = min(max(HANDLE * math.dist(p0, p1), 0.6), HANDLE_MAX)
        piece = bezier(p0, add(p0, t0, k), add(p1, t1, -k), p1)
        out.extend(piece if not out else piece[1:])
    return out


def dog_path(course, tunnel_lines=None):
    """Approximate dog path, see dogpath.py. Returns (through, gaps)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import dogpath
    if tunnel_lines is None:
        tunnel_lines = {o["id"]: catmull_rom([(p["x"], p["y"]) for p in o["path"]])
                        for o in course["obstacles"] if o["type"] == "tunnel"}
    return dogpath.dog_path(course, DIMS, tunnel_lines, entry_exit)


def polyline_length(line):
    return sum(math.dist(p, q) for p, q in zip(line, line[1:]))


LABEL_R = 0.48  # number circle radius, metres


def seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy or 1.0
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def line_dist(p, line):
    return min(seg_dist(p, a, b) for a, b in zip(line, line[1:])) if len(line) > 1 else math.dist(p, line[0])


def footprint_dist(p, o, tunnel_lines):
    """Distance from point p to the obstacle's outline (0 inside)."""
    if o["type"] == "tunnel":
        return max(0.0, line_dist(p, tunnel_lines[o["id"]]) - TUNNEL_WIDTH / 2)
    d, u = DIMS[o["type"]], unit(o["heading"])
    v = (p[0] - o["x"], p[1] - o["y"])
    a = abs(v[0] * u[0] + v[1] * u[1]) - max(d["depth"], 0.3) / 2
    b = abs(-v[0] * u[1] + v[1] * u[0]) - d["width"] / 2
    return math.hypot(max(a, 0.0), max(b, 0.0))


def label_positions(course, steps, tunnel_lines, path_lines):
    """Place each number as close to its obstacle as possible.

    Candidates on rings around the obstacle are scored: nearer is better, the
    approach side is preferred, and positions that overlap an obstacle, the dog
    path, another number or the ring edge are penalised.
    """
    W, H = course["field"]["width"], course["field"]["height"]
    placed = []
    for o, entry, _, back in steps:
        long_ = o["type"] in ("tunnel", "weave", "dog_walk", "a_frame", "seesaw", "long_jump")
        anchor = entry if long_ else ((o["x"], o["y"]) if o["type"] != "tunnel" else entry)
        best, best_cost = None, float("inf")
        for r in (0.75, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5):
            for k in range(24):
                ang = 2 * math.pi * k / 24
                c = (anchor[0] + r * math.cos(ang), anchor[1] + r * math.sin(ang))
                own = footprint_dist(c, o, tunnel_lines)
                cost = r + (0.0 if (c[0] - anchor[0]) * back[0] + (c[1] - anchor[1]) * back[1] >= -0.05 else 5.0)  # numbers belong on the take-off side
                if own < LABEL_R + 0.12:
                    cost += 100
                for q in course["obstacles"]:
                    if q is not o and footprint_dist(c, q, tunnel_lines) < LABEL_R + 0.12:
                        cost += 100
                if any(math.dist(c, q) < 2 * LABEL_R + 0.08 for q in placed):
                    cost += 100
                if path_lines and min(line_dist(c, l) for l in path_lines) < LABEL_R + 0.08:
                    cost += 8
                if not (LABEL_R <= c[0] <= W - LABEL_R and LABEL_R <= c[1] <= H - LABEL_R):
                    cost += 50
                if cost < best_cost:
                    best, best_cost = c, cost
        placed.append(best)
    return placed


def render(course, out, show_path=False, blind=False):
    W, H = course["field"]["width"], course["field"]["height"]
    scale = 0.28  # inches per metre
    fig, ax = plt.subplots(figsize=(W * scale + 0.8, H * scale + 1.1), dpi=200)
    ax.set_xlim(-1, W + 1)
    ax.set_ylim(H + 1, -1)  # y downwards
    ax.set_aspect("equal")
    ax.axis("off")

    # grid: 1 m minor, 5 m major with labels
    for x in range(0, int(W) + 1):
        ax.plot([x, x], [0, H], color=GRID_MAJOR if x % 5 == 0 else GRID_MINOR, linewidth=0.6, zorder=0)
        if x % 5 == 0:
            ax.text(x, -0.35, f"{x}", ha="center", va="bottom", fontsize=6, color="#888888")
    for y in range(0, int(H) + 1):
        ax.plot([0, W], [y, y], color=GRID_MAJOR if y % 5 == 0 else GRID_MINOR, linewidth=0.6, zorder=0)
        if y % 5 == 0:
            ax.text(-0.35, y, f"{y}", ha="right", va="center", fontsize=6, color="#888888")
    ax.add_patch(Polygon([(0, 0), (W, 0), (W, H), (0, H)], closed=True, fill=False,
                         edgecolor=INK, linewidth=1.6, zorder=1))

    obstacles = {o["id"]: o for o in course["obstacles"]}
    tunnel_lines = {}
    for o in course["obstacles"]:
        if o["type"] == "tunnel":
            tunnel_lines[o["id"]] = draw_tunnel(ax, o)
        else:
            draw_placed(ax, o)

    steps = []
    for step in course["sequence"]:
        o = obstacles[step["obstacle"]]
        steps.append((o, *entry_exit(o, step.get("direction", "forward"), tunnel_lines)))

    if show_path:
        through, gaps = dog_path(course, tunnel_lines)
        for line in through + gaps:
            ax.plot([p[0] for p in line], [p[1] for p in line], color=PATH, linewidth=1.0,
                    solid_capstyle="round", solid_joinstyle="round", zorder=6, alpha=0.9)
        for g in gaps:  # small arrowhead halfway along each gap
            m = len(g) // 2
            (x0, y0), (x1, y1) = g[m - 1], g[m + 1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=PATH, lw=0, mutation_scale=7))

    through, gaps = dog_path(course, tunnel_lines)
    positions = label_positions(course, steps, tunnel_lines, through + gaps)
    for i, pos in enumerate(positions, start=1):
        ax.add_patch(Circle(pos, LABEL_R, facecolor="white", edgecolor=INK, linewidth=1.1, zorder=7))
        ax.text(pos[0], pos[1], str(i), ha="center", va="center", fontsize=7, fontweight="bold",
                color=INK, zorder=8)

    m = course["meta"]
    kind = {"agility": "Agility", "jumping": "Hyppy"}[m["discipline"]]
    title = f"{kind} {m['class']}  ·  {len(course['sequence'])} estettä  ·  {W:g} × {H:g} m"
    if not blind:
        title = f"{m['course_id']} ({m['source']})  ·  " + title
    ax.set_title(title, fontsize=9, loc="left", color=INK, pad=14)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("course")
    ap.add_argument("-o", "--out")
    ap.add_argument("--path", action="store_true")
    ap.add_argument("--blind", action="store_true")
    a = ap.parse_args()
    course = json.loads(Path(a.course).read_text(encoding="utf-8"))
    out = a.out or str(Path(a.course).with_suffix(".png"))
    render(course, out, show_path=a.path, blind=a.blind)
    print(out)
