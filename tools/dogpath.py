"""Approximate dog path through a course (used by render.py and mirrored in the tool).

The path is one smooth curve (centripetal Catmull-Rom spline) through waypoints
at the obstacles. Helper points are kept as small as possible, because a smooth
line through the obstacles is already close to how judges draw the path.

- hurdle: a short straight piece centred on the bar (take-off, centre, landing),
  crossed in the direction the course flows; limited to HURDLE_MAX_ANGLE
- spread, tyre, long jump, wall: the same, but crossed straight on (SAGI rule)
- contacts: straight along the obstacle, with a short straight run-in and run-out
- tunnel: along the tunnel
- weave: a straight line from the middle of the first gap to the last gap; if the
  dog comes from the wrong side, one extra point so the first pole is on its left
- turn-back after a hurdle: the dog crosses in the direction it came from and the
  line turns after landing; no loops are added before the jump
- if the line would pass through a hurdle's bar or wing a second time, one point
  beyond the wing is added

Lengths are an estimate of the dog's line, not the judge's official measurement.
"""
import math

STRAIGHT_JUMPS = {"spread", "tyre", "wall"}  # the long jump has a start and an end, like a contact
JUMPS = {"hurdle"} | STRAIGHT_JUMPS
CONTACTS = {"a_frame", "dog_walk", "seesaw"}

JUMP_PIECE = 0.5        # metres before and after a hurdle's bar in which the dog is on its jumping line
HURDLE_MAX_ANGLE = 65   # max degrees from straight on at a hurdle
UTURN_MAX_ANGLE = 45    # the same when the next obstacle lies behind the jump (the dog turns back after landing)
WING_CLEARANCE = 0.15   # metres the line keeps beyond a wing end when it has to pass a jump
WEAVE_SIDE = 0.7        # sideways offset of the entry helper when the dog comes from the wrong side
TURN_RADIUS = 1.2       # radius of the turn after landing when the dog turns back (metres)
BIG_TURN = 90           # degrees: a turn this large before a jump gets the loop, after a jump gets an arc


def _add(p, v, k=1.0):
    return (p[0] + k * v[0], p[1] + k * v[1])


def _unit(h):
    a = math.radians(h)
    return (math.cos(a), math.sin(a))


def _norm(v):
    l = math.hypot(v[0], v[1])
    return (v[0] / l, v[1] / l) if l > 1e-9 else None


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _turn(a, b):
    """Signed angle (radians) from direction a to direction b."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], _dot(a, b))


def _rot(v, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return (v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca)


def _clamp_dir(base, want, max_deg):
    ang = math.atan2(base[0] * want[1] - base[1] * want[0], _dot(base, want))
    lim = math.radians(max_deg)
    ang = max(-lim, min(lim, ang))
    ca, sa = math.cos(ang), math.sin(ang)
    return (base[0] * ca - base[1] * sa, base[0] * sa + base[1] * ca)


def _seg_x(p1, p2, q1, q2):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (o(q1, q2, p1) > 0) != (o(q1, q2, p2) > 0) and (o(p1, p2, q1) > 0) != (o(p1, p2, q2) > 0)


def centripetal_spline(points, step=0.1, with_index=False):
    """Centripetal Catmull-Rom curve through all points, sampled about every `step` metres.
    With with_index=True also returns, for each input point, its index in the output line."""
    pts, keep = [points[0]], [0]
    for p in points[1:]:
        if math.dist(p, pts[-1]) > 1e-3:
            pts.append(p)
        keep.append(len(pts) - 1)
    if len(pts) < 2:
        return (pts, [0] * len(points)) if with_index else pts
    ext = [_add(pts[0], (pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))] + pts + \
          [_add(pts[-1], (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]))]
    out, at = [pts[0]], [0]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + math.dist(p0, p1) ** 0.5
        t2 = t1 + math.dist(p1, p2) ** 0.5
        t3 = t2 + math.dist(p2, p3) ** 0.5
        n = max(2, int(math.dist(p1, p2) / step))
        for s in range(1, n + 1):
            t = t1 + (t2 - t1) * s / n

            def lerp(a, b, ta, tb):
                if tb - ta < 1e-9:
                    return a
                return ((tb - t) / (tb - ta) * a[0] + (t - ta) / (tb - ta) * b[0],
                        (tb - t) / (tb - ta) * a[1] + (t - ta) / (tb - ta) * b[1])
            a1, a2, a3 = lerp(p0, p1, t0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
            b1, b2 = lerp(a1, a2, t0, t2), lerp(a2, a3, t1, t3)
            out.append(lerp(b1, b2, t1, t2))
        at.append(len(out) - 1)
    return (out, [at[k] for k in keep]) if with_index else out


TANGENT_LEAD = 1.5      # metres: how far along the obstacle line the curve is already straight at its ends (weave gaps, contact run-ins)
STRAIGHT_MAX_ANGLE = 15  # degrees a spread, tyre, long jump or wall may be crossed off its own direction
CONTACT_LEAD = 0.8      # the same at a contact's plank end
HURDLE_LEAD = 0.8       # the same at a hurdle bar: short, so the line just passes the bar smoothly in its crossing direction
HURDLE_LEAD_TURN = 1.4  # added to HURDLE_LEAD for a full turnaround at the jump (scaled by the sharpest turn)


def broken_spline(points, breaks, step=0.1):
    """centripetal_spline, but at each break (index, direction) the curve passes through the point moving
    exactly in that direction. Used at the weave and at contacts: the line rounds in and arrives tangent
    to the pole line / plank (no corner, no overshoot), runs straight through and leaves the same way."""
    dirs, leads = {}, {}
    for br in breaks:
        if 0 < br[0] < len(points) - 1:
            dirs[br[0]] = br[1]
            leads[br[0]] = br[2] if len(br) > 2 else TANGENT_LEAD
    bounds = [0] + sorted(dirs) + [len(points) - 1]
    line, at = [], [0] * len(points)
    for a, b in zip(bounds, bounds[1:]):
        seg_pts = list(points[a:b + 1])
        # phantom points placed so that the curve's tangent AT the break point is exactly the given
        # direction (Catmull-Rom: tangent at an end point follows phantom minus neighbour)
        if a in dirs:
            u, q = dirs[a], points[min(a + 1, b)]
            c = max((q[0] - points[a][0]) * u[0] + (q[1] - points[a][1]) * u[1], 0.0) + leads[a]
            seg_pts.insert(0, _add(q, u, -c))
        if b in dirs:
            u, q = dirs[b], points[max(b - 1, a)]
            c = max((points[b][0] - q[0]) * u[0] + (points[b][1] - q[1]) * u[1], 0.0) + leads[b]
            seg_pts.append(_add(q, u, c))
        seg, idx = centripetal_spline(seg_pts, step, with_index=True)
        lo = idx[1] if a in dirs else 0
        hi = idx[-2] if b in dirs else idx[-1]
        piece = seg[lo:hi + 1]
        off = len(line) - 1 if line else 0
        line += piece[1:] if line else piece
        base = 1 if a in dirs else 0
        for k in range(b - a + 1):
            at[a + k] = off + idx[base + k] - lo
    return line, at


def _along(a, b, spacing=1.0):
    n = max(1, int(math.dist(a, b) / spacing))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1)]


def _resample(line, spacing=0.8):
    out, acc = [line[0]], 0.0
    for a, b in zip(line, line[1:]):
        acc += math.dist(a, b)
        if acc >= spacing:
            out.append(b)
            acc = 0.0
    if out[-1] != line[-1]:
        out.append(line[-1])
    return out


def dog_path(course, dims, tunnel_lines, entry_exit):
    """Return (through, gaps): through[i] is the line at obstacle i, gaps[i] from i to i+1."""
    obstacles = {o["id"]: o for o in course["obstacles"]}
    seq = [(obstacles[s["obstacle"]], s.get("direction", "forward")) for s in course["sequence"]]
    # hand-set control points: the gap after step i passes through control[i] and gets no automatic helpers
    control = {}
    for i, s in enumerate(course["sequence"]):
        pc = s.get("path_control")
        if pc:
            pts_ = pc if isinstance(pc, list) else [pc]
            control[i] = [(q["x"], q["y"]) for q in pts_]
    extra = {i: [("post", q) for q in ps] for i, ps in control.items()}  # step index -> list of ("pre"|"post", point)

    def travel(o, d):
        u = _unit(o["heading"])
        return u if d == "forward" else (-u[0], -u[1])

    def ref_point(j):  # where the dog heads for when going to step j
        o, d = seq[j]
        if o["type"] == "weave":
            return _add((o["x"], o["y"]), travel(o, d), -3.2)
        if o["type"] in JUMPS:
            return (o["x"], o["y"])
        if o["type"] == "tunnel":
            line = tunnel_lines[o["id"]]
            return line[0] if d == "forward" else line[-1]
        return entry_exit(o, d, tunnel_lines)[0]

    hurdle_dir = {}  # step index -> crossing direction at the bar centre (core point 0)

    def build():
        hurdle_dir.clear()
        groups = []  # per step: (pre points, core points, post points)
        for i, (o, d) in enumerate(seq):
            t = o["type"]
            pre = [p for k, p in extra.get(i, []) if k == "pre"]
            post = [p for k, p in extra.get(i, []) if k == "post"]
            if t in JUMPS:
                base = travel(o, d)
                c = (o["x"], o["y"])
                prev = (groups[-1][1] + groups[-1][2])[-1] if groups else None
                nxt = control[i][0] if i in control else (ref_point(i + 1) if i + 1 < len(seq) else None)
                vin = _norm((c[0] - prev[0], c[1] - prev[1])) if prev else None
                vout = _norm((nxt[0] - c[0], nxt[1] - c[1])) if nxt else None
                last = i == len(seq) - 1
                if t == "wall":
                    # the wall is just a point the line passes through: no direction rule at all
                    core = [c]
                elif t in STRAIGHT_JUMPS:
                    # must be taken straight: one point at the centre which the line passes in the jump's own
                    # direction (no hidden points before or after it)
                    # (a few degrees of slack, STRAIGHT_MAX_ANGLE, so a gentle arc can pass without a bump)
                    best, fix = None, base
                    for k in range(-10, 11):
                        cand = _rot(base, math.radians(STRAIGHT_MAX_ANGLE * k / 10))
                        cost = (_turn(vin, cand) ** 2 if vin else 0.0) + (_turn(cand, vout) ** 2 if vout else 0.0)
                        if best is None or cost < best - 1e-9:
                            best, fix = cost, cand
                    core = [c]
                    sharp = max(abs(_turn(vin, fix)) if vin else 0.0, abs(_turn(fix, vout)) if vout else 0.0)
                    hurdle_dir[i] = (fix, HURDLE_LEAD + HURDLE_LEAD_TURN * sharp / math.pi)
                else:
                    # A hurdle is one point, its bar centre, which the line passes smoothly in one crossing
                    # direction (like a node in a drawing program; no hidden extra points). The direction:
                    # with hand-set points beside the jump, the natural one from the points before and after;
                    # otherwise the one (within HURDLE_MAX_ANGLE of the jump's own) that balances the turn
                    # before and the turn after the jump.
                    parts = [v for v in (vin, vout) if v]
                    want = _norm((sum(v[0] for v in parts), sum(v[1] for v in parts))) if parts else None
                    hand = (i - 1) in control or i in control
                    turn_back = vout is not None and _dot(vout, base) < -0.3 and not hand and not last
                    if hand and want and _dot(want, base) > 0.05:
                        fix = want
                    else:
                        best, fix = None, base
                        cap = UTURN_MAX_ANGLE if turn_back else HURDLE_MAX_ANGLE
                        for k in range(-10, 11):
                            cand = _rot(base, math.radians(cap * k / 10))
                            cost = (_turn(vin, cand) ** 2 if vin else 0.0) + (_turn(cand, vout) ** 2 if vout else 0.0)
                            if best is None or cost < best - 1e-9:
                                best, fix = cost, cand
                    core = [c]
                    # the sharper the turn before or after the jump, the longer the line stays on its crossing
                    # direction, so a turnaround becomes a loop instead of a sharp V
                    sharp = max(abs(_turn(vin, fix)) if vin else 0.0, abs(_turn(fix, vout)) if vout else 0.0)
                    hurdle_dir[i] = (fix, HURDLE_LEAD + HURDLE_LEAD_TURN * sharp / math.pi)
                    if turn_back:
                        # next obstacle behind the jump: cross near straight on, then a rounded turn after landing
                        cr = fix
                        land = _add(c, cr, JUMP_PIECE)
                        v = _norm((nxt[0] - land[0], nxt[1] - land[1])) or vout
                        d2 = _turn(cr, v)
                        side = 1.0 if d2 > 0 else -1.0
                        perp = (-cr[1] * side, cr[0] * side)
                        sweep = abs(d2)
                        centre = _add(land, perp, TURN_RADIUS)
                        for f in ((0.5, 1.0) if sweep >= 0.75 * math.pi else (0.5,)):
                            a = f * sweep
                            core.append(_add(_add(centre, perp, -TURN_RADIUS * math.cos(a)), cr, TURN_RADIUS * math.sin(a)))
            elif t == "weave":
                # 12 poles at -3.52 .. +3.52 m (0.64 m apart). The line comes straight from the previous obstacle into the gap
                # between poles 1 and 2, runs straight along the poles and leaves from the gap between the last
                # two poles. Pole 1 must be on the dog's left; the side check further down adds a point if not.
                us = travel(o, d)
                c = (o["x"], o["y"])
                core = _along(_add(c, us, -3.2), _add(c, us, 3.2), 1.0)
            elif t == "tunnel":
                line = tunnel_lines[o["id"]]
                core = _resample(line if d == "forward" else line[::-1])
            else:  # contacts and long jump: from the start to the end of the obstacle
                entry, exit_, _ = entry_exit(o, d, tunnel_lines)
                u = _norm((exit_[0] - entry[0], exit_[1] - entry[1]))
                # no run-in/run-out points: the line reaches the plank end moving along the plank
                # (tangent break) and may turn right up to it, as in the judges' course designer
                core = _along(entry, exit_, 1.2)
            groups.append((pre, core, post))
        pts = [p for g in groups for p in g[0] + g[1] + g[2]]
        spans, k = [], 0
        for pre, core, post in groups:
            spans.append((k, k + len(pre), k + len(pre) + len(core) - 1, k + len(pre) + len(core) + len(post) - 1))
            k += len(pre) + len(core) + len(post)
        return groups, pts, spans

    def tangent_breaks(spans):
        # at the weave's first/last gap and at a contact's run-in/run-out the line moves exactly along
        # the obstacle, so it rounds in tangent to the poles or the plank and leaves the same way
        out = [(k, travel(o, d), TANGENT_LEAD if o["type"] == "weave" else CONTACT_LEAD)
               for i, (o, d) in enumerate(seq) if o["type"] in ("weave", "a_frame", "dog_walk", "seesaw", "long_jump")
               for k in (spans[i][1], spans[i][2])]
        out += [(spans[i][1], u, lead) for i, (u, lead) in hurdle_dir.items()]
        return out

    jumps = [(o, travel(o, d)) for o, d in seq if o["type"] in JUMPS]
    for _ in range(5):
        groups, pts, spans = build()
        line, at = broken_spline(pts, tangent_breaks(spans))
        changed = False
        # which line indices belong to each obstacle's own crossing(s)
        own = {}
        for i, (o, d) in enumerate(seq):
            own.setdefault(o["id"], []).append((at[spans[i][1]] - 2, at[spans[i][2]] + 2))  # a little margin round the obstacle
        gap_ranges = [(at[spans[i][2]], at[spans[i + 1][1]]) for i in range(len(seq) - 1)]
        for o, base in jumps:
            u = _unit(o["heading"])
            n = (-u[1], u[0])
            c = (o["x"], o["y"])
            w = dims[o["type"]]["width"] / 2
            bar = (_add(c, n, -w), _add(c, n, w))
            if o["type"] == "hurdle":
                # a hurdle's wings are low: only the bar itself must not be crossed (and not its very ends,
                # which the line may brush when it passes close to the wing)
                wb = (dims["hurdle"]["width"] - 2 * 0.5) / 2 - 0.15
                bar = (_add(c, n, -wb), _add(c, n, wb))
            for gi, (g0, g1) in enumerate(gap_ranges):
                if gi in control or any(k == gi for k in extra.get(("wing", o["id"]), [])):
                    continue
                for k in range(g0 + 1, g1 + 1):
                    if any(lo <= k - 1 and k <= hi for lo, hi in own[o["id"]]):
                        continue
                    if _seg_x(line[k - 1], line[k], bar[0], bar[1]):
                        # the line passes through this jump's bar or wings: go around the nearer wing end
                        p = line[k]
                        full_w = dims[o["type"]]["width"] / 2
                        ends = (_add(c, n, -full_w), _add(c, n, full_w))
                        e = min(ends, key=lambda q: math.dist(q, p))
                        out = _norm((e[0] - c[0], e[1] - c[1]))
                        detour = _add(e, out, WING_CLEARANCE)
                        near_start = math.dist(p, line[g0]) <= math.dist(p, line[g1])
                        if near_start:
                            extra.setdefault(gi, []).append(("post", detour))
                        else:
                            extra.setdefault(gi + 1, []).insert(0, ("pre", detour))
                        extra.setdefault(("wing", o["id"]), []).append(gi)
                        changed = True
                        break
        for i, (o, d) in enumerate(seq):
            if o["type"] != "weave" or i == 0 or extra.get(i) or (i - 1) in control:
                continue
            us = travel(o, d)
            right = (-us[1], us[0])
            first = _add((o["x"], o["y"]), us, -3.2)
            prev = pts[spans[i][0] - 1]
            v = _norm((first[0] - prev[0], first[1] - prev[1]))
            # the dog must reach the gap with pole 1 on its left, i.e. coming from the right of the pole line
            if v and _dot(v, right) > -0.1:
                extra.setdefault(i, []).append(("pre", _add(_add(first, us, -1.2), right, WEAVE_SIDE)))
                changed = True
        if not changed:
            break

    groups, pts, spans = build()
    line, at = broken_spline(pts, tangent_breaks(spans))
    # the line ends on the last obstacle, not beyond it
    last_o, last_d = seq[-1]
    if last_o["type"] in JUMPS:
        end = at[spans[-1][1]] if last_o["type"] == "hurdle" else at[spans[-1][2]]
        line = line[:end + 1]
        at = [min(k, end) for k in at]
    marks = [(at[c0], at[c1]) for _, c0, c1, _ in spans]
    through = [line[a:b + 1] for a, b in marks]
    gaps = [line[marks[i][1]:marks[i + 1][0] + 1] for i in range(len(marks) - 1)]
    return through, gaps

