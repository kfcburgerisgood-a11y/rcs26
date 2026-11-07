"""Finding routes over a roof: the cheapest way between stops, where climbing costs."""

import math

import numpy as np
from compas.geometry import Polyline
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .drawing import Drawing

__all__ = ["route", "network"]

ALONGSIDE = 20.0    # how much dearer a step is within a path's width of the walk: branches leave it, not shadow it
# how much a turn costs, as walking: width x (1 - cos(turn)) times this. Measured on a 1:8 slope, under about
# 0.4 a route zigzags a cell wide and smoothing leaves it straight up the slope; above it, it keeps switchbacks.
TURN_COST = 1.0

# a step to any of 16 neighbours, so a route can head 16 ways rather than 8
MOVES = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]
MOVES += [(-a, -b) for a, b in MOVES]


def route(brief, found, drawing):
    """Step 3: each drawing's stops become a path, the cheapest way between them. *Canvas: Shortest Walk.*

    Reads ``width``, ``ramp_max``, ``avoid_slope``, ``turns``, ``route_tolerance``,
    and the grid of ``found``, the slope map. A step costs its length, more the
    steeper it climbs *in the direction it goes*: ``avoid_slope x (grade / ramp_max)^2``
    more, on the roof as it is. So a route crosses a slope along its contours
    where that is cheaper than going round, as a mountain path does. Between
    each two stops it swings ``turns`` times, left and right in turn, through
    the gentlest ground near each swing (see ``_turning_points``). It keeps
    ``width / 2`` clear of holes and the roof's edge, and is drawn as one smooth
    winding line from stop to stop (see ``_wind``). Returns the drawing with
    each route added to its paths.

    Examples
    --------
    >>> from roof_circulation import Brief, steps
    >>> brief = Brief(roughness=0.0, length=6000.0, span=3000.0)
    >>> stops = Drawing(stops=[Polyline([[1000, 1500, 0], [5000, 1500, 0]])])
    >>> found = route(brief, steps.slopes(brief, steps.host(brief)), stops)
    >>> line = found.paths[0]
    >>> straight = max(abs(p[1] - 1500) for p in line) <= brief.cell / 2     # on a flat roof, to within the grid
    >>> list(line[0])[:2], list(line[-1])[:2], straight
    ([1000.0, 1500.0], [5000.0, 1500.0], True)
    """
    room, height = _terrain(brief, found)
    prefer = _preference(brief, found, drawing)
    paths, taken = list(drawing.paths), []
    for stops in drawing.stops:
        points = [(float(p[0]), float(p[1])) for p in stops]
        corners = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            via = [a] + _turning_points(brief, found, room, a, b, prefer, drawing.attractors) + [b]
            way = [a]
            for p, q in zip(via[:-1], via[1:]):
                # a route never runs over itself or another: what is taken is closed, a cell beyond a path's width
                # for the smoothing to come, but around p and q, to leave by
                clear = brief.width + brief.cell
                free = room & (~_near(found, taken, clear) | _near(found, [[p], [q]], clear))
                cells = _search(_graph(brief, found, free, height, prefer), free.shape, _cell(found, free, p),
                                _cell(found, free, q), p, q)
                part = [p] + [(float(found.x[i]), float(found.y[j])) for i, j in cells[1:-1]] + [q]
                part = _pull(brief, found, free, height, part, prefer)     # taut between swings, not across them
                taken.append(part)
                way += part[1:]
            # one smooth line from stop to stop: the swings bend it, they are not corners
            corners += _wind(brief, found, room, way)[1:]
        paths.append(Polyline([[x, y, 0.0] for x, y in corners]))
    return Drawing(paths, list(drawing.pins), [], list(drawing.entrances), list(drawing.attractors),
                   list(drawing.repellers))


def _turning_points(brief, found, room, a, b, weight, attractors=()):
    """``turns`` points for a route from ``a`` to ``b`` to swing through, left and right in turn.

    They are spaced evenly along the way and set aside by ``swing`` times that
    spacing, so each swing leans the same way off the straight line however many
    there are (0.5: about 27 degrees): more turns wind more, a wider swing bows
    further. Each moves to the gentlest ground near its place, kept from
    repellers by ``weight``. An attractor within ``influence`` of a swing's place
    captures it, the nearest swing only: the swing goes to the attractor.
    """
    if not brief.turns:
        return []
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    step = length / (brief.turns + 1)
    gx, gy = np.meshgrid(found.x, found.y, indexing="ij")
    steep = np.nan_to_num(found.slope, nan=np.inf)
    aside = step * brief.swing
    places = [(a[0] + ux * step * k - uy * (1 if k % 2 else -1) * aside,
               a[1] + uy * step * k + ux * (1 if k % 2 else -1) * aside) for k in range(1, brief.turns + 1)]
    for lure in attractors:
        k = min(range(len(places)), key=lambda n: math.hypot(places[n][0] - lure[0], places[n][1] - lure[1]))
        if math.hypot(places[k][0] - lure[0], places[k][1] - lure[1]) <= brief.influence:
            places[k] = (float(lure[0]), float(lure[1]))
    points = []
    for tx, ty in places:
        off = np.hypot(gx - tx, gy - ty)
        reach = max(aside / 2, brief.cell)
        near = room & (off <= reach)
        if near.any():  # off the roof, or no room there: no swing
            score = np.where(near, (steep + brief.ramp_max * off / reach) * weight, np.inf)
            i, j = np.unravel_index(int(score.argmin()), score.shape)
            points.append((float(found.x[i]), float(found.y[j])))
    return points


def network(brief, roof, found, drawing, landing=None):
    """Step 3, from entrances alone: a walk that branches out over the gentle ground. *Canvas: Shortest Walk, grown.*

    Reads ``branches``, ``edge_clearance``, and what ``route`` and ``grade`` read.
    Grows the walk one branch at a time from the ``roof`` and its slope map
    ``found``: each end on ground no steeper than ``ramp_max``, at least
    ``edge_clearance`` from the roof's edge and holes, as far as it can be from
    everything walked so far; each branch leaving the walk wherever that is
    cheapest, kept from running alongside it. The walk is split at every fork,
    so each piece meets the others at its ends, and a branch is kept only if
    every piece can still be walked at ``ramp_max``. Returns the drawing with
    the walk added to its paths; fewer branches than asked if no more fit.

    Examples
    --------
    >>> from roof_circulation import Brief, steps
    >>> brief = Brief(roughness=0.0, branches=3)
    >>> roof = steps.host(brief)
    >>> grown = network(brief, roof, steps.slopes(brief, roof), Drawing.sample_entrance())
    >>> len(_tips(grown.paths, Drawing.sample_entrance().entrances))
    3
    """
    room, height = _terrain(brief, found)
    graph = _graph(brief, found, room, height)
    starts = [(float(p[0]), float(p[1])) for p in drawing.entrances]
    if not starts:
        raise ValueError("A network needs at least one entrance.")
    first = [np.ravel_multi_index(_cell(found, room, s), room.shape) for s in starts]
    reach = np.isfinite(_walk(graph, room.shape, first)[0])
    edge = distance_transform_edt(np.pad(~np.isnan(found.height), 1))[1:-1, 1:-1] * brief.cell
    dx, dy = np.gradient(height, brief.cell)
    can_end = reach & (np.hypot(dx, dy) <= brief.ramp_max) & (edge >= brief.edge_clearance)
    if not can_end.any():
        raise ValueError("Nowhere to end a branch: no ground reachable from the entrances is gentler than 1:{:.0f} "
                         "and {:.0f} mm from the roof's edge.".format(1 / brief.ramp_max, brief.edge_clearance))
    from .steps import grade, ground, rest     # here, not at the top: steps imports this module

    walk, grown = [], 0
    for _ in range(4 * brief.branches):     # shortcut: a few tries per branch, then fewer branches than asked
        if grown == brief.branches:
            break
        # the next end: as far as it can be from everything walked so far
        apart = np.where(can_end, _distance(found, walk + [[s] for s in starts]), -1.0)
        if apart.max() < brief.width * 2:   # every gentle spot left is next to the walk already
            break
        i, j = np.unravel_index(int(apart.argmax()), apart.shape)
        end = (float(found.x[i]), float(found.y[j]))
        on = _near(found, walk + [[s] for s in starts], brief.cell)
        weight = np.where(_near(found, walk, brief.width) & ~on, ALONGSIDE, 1.0) * _preference(brief, found, drawing)
        _, way = _walk(_graph(brief, found, room, height, weight), room.shape, np.flatnonzero(on))
        cells = way(np.ravel_multi_index(_cell(found, room, end), room.shape))
        fork, split = _fork(walk, starts, (float(found.x[cells[0][0]]), float(found.y[cells[0][1]])), brief.width)
        leg = [fork] + [(float(found.x[i]), float(found.y[j])) for i, j in cells[1:-1]] + [end]
        trial = split + [_wind(brief, found, room, leg)]
        try:    # a branch is kept only if every piece can still be walked at ramp_max, with rests if given
            pieces = Drawing([Polyline([[x, y, 0.0] for x, y in line]) for line in trial])
            profiles = grade(brief, ground(brief, roof, pieces))
            if landing is not None:
                rest(brief, landing, roof, profiles, drawing)
        except ValueError:
            can_end &= ~_near(found, [[end]], brief.width * 2)
            continue
        walk, grown = trial, grown + 1
    paths = list(drawing.paths) + [Polyline([[x, y, 0.0] for x, y in line]) for line in walk]
    return Drawing(paths, list(drawing.pins), list(drawing.stops), list(drawing.entrances),
                   list(drawing.attractors), list(drawing.repellers))


def _fork(walk, starts, point, snap):
    """Where a branch leaves the walk, nearest ``point``: an entrance, a corner within ``snap``, or a new corner.

    Returns the fork and the walk, split there so that the fork ends every piece it touches.
    """
    best = min(((math.hypot(point[0] - s[0], point[1] - s[1]), None, None, s) for s in starts), key=lambda c: c[0])
    for n, line in enumerate(walk):
        for k, (p, q) in enumerate(zip(line[:-1], line[1:])):
            dx, dy = q[0] - p[0], q[1] - p[1]
            t = min(max(((point[0] - p[0]) * dx + (point[1] - p[1]) * dy) / (dx * dx + dy * dy), 0.0), 1.0)
            at = (p[0] + t * dx, p[1] + t * dy)
            off = math.hypot(point[0] - at[0], point[1] - at[1])
            if off < best[0]:
                best = (off, n, k, at)
    _, n, k, at = best
    if n is None:
        return at, walk
    line = walk[n]
    for corner in (line[0], line[-1]):
        if math.hypot(at[0] - corner[0], at[1] - corner[1]) <= snap:
            return corner, walk
    return at, walk[:n] + [line[:k + 1] + [at], [at] + line[k + 1:]] + walk[n + 1:]


def _tips(paths, entrances):
    """The ends of a walk: path ends that no other path shares, and that are not entrances."""
    ends = [(round(p[0], 3), round(p[1], 3)) for path in paths for p in (path[0], path[-1])]
    doors = {(round(e[0], 3), round(e[1], 3)) for e in entrances}
    return [e for e in ends if ends.count(e) == 1 and e not in doors]


def _preference(brief, found, drawing):
    """How much a step costs in each cell for the drawing's attractors and repellers: a weight, 1 elsewhere.

    Within ``influence`` of an attractor a step is up to ``attract`` cheaper, of a
    repeller up to ``1 + repel`` times dearer, fading as the square of the way out.
    """
    gx, gy = np.meshgrid(found.x, found.y, indexing="ij")
    weight = np.ones(gx.shape)
    for points, strength in ((drawing.attractors, -brief.attract), (drawing.repellers, brief.repel)):
        for p in points:
            weight *= 1 + strength * np.clip(1 - np.hypot(gx - p[0], gy - p[1]) / brief.influence, 0.0, 1.0) ** 2
    return weight


def _distance(found, lines):
    """Each cell's plan distance to the nearest of ``lines``, each a list of plan points (one point is a dot)."""
    gx, gy = np.meshgrid(found.x, found.y, indexing="ij")
    nearest = np.full(gx.shape, np.inf)
    for line in lines:
        for p, q in zip(line, line[1:] or line):
            (px, py), (dx, dy) = p, (q[0] - p[0], q[1] - p[1])
            t = np.clip(((gx - px) * dx + (gy - py) * dy) / ((dx * dx + dy * dy) or 1.0), 0.0, 1.0)
            nearest = np.minimum(nearest, np.hypot(gx - px - t * dx, gy - py - t * dy))
    return nearest


def _near(found, lines, distance):
    """The cells closer than ``distance`` to any of ``lines``."""
    return _distance(found, lines) < distance


def _terrain(brief, found):
    """Where a path fits, ``width / 2`` clear of holes and edges; and the roof's height, holes filled."""
    roof = ~np.isnan(found.height)
    room = distance_transform_edt(np.pad(roof, 1))[1:-1, 1:-1] * brief.cell > brief.width / 2
    return room, np.where(roof, found.height, np.nanmean(found.height))


def _pull(brief, found, room, height, way, weight):
    """The way found, pulled taut: straight wherever a straight line costs no more and has room.

    On a grid of 16 headings a way between them zigzags or bows; a straight
    line costs its length, its climb and ``weight`` by the same rule as the
    search, so pulling never takes a steeper shortcut, nor one past a repeller
    or away from an attractor. Returns plan points every ``cell``.
    """
    def cost(p, q):
        n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (brief.cell / 2)))
        xs, ys = np.linspace(p[0], q[0], n + 1), np.linspace(p[1], q[1], n + 1)
        i = np.abs(found.x[:, None] - xs).argmin(axis=0)
        j = np.abs(found.y[:, None] - ys).argmin(axis=0)
        step = math.hypot(q[0] - p[0], q[1] - p[1]) / n
        grade = np.abs(np.diff(height[i, j])) / step
        dear = np.maximum(weight[i, j][:-1], weight[i, j][1:])
        return float((step * (1 + brief.avoid_slope * (grade / brief.ramp_max) ** 2) * dear).sum())

    along = [0.0] + [cost(p, q) for p, q in zip(way[:-1], way[1:])]
    spent = np.cumsum(along)
    taut, k = [way[0]], 0
    while k < len(way) - 1:
        far = k + 1
        for m in range(len(way) - 1, k + 1, -1):     # the farthest point a straight line reaches as cheaply
            if cost(way[k], way[m]) <= spent[m] - spent[k] and _fits(found, room, way[k], way[m]):
                far = m
                break
        n = max(1, int(math.hypot(way[far][0] - way[k][0], way[far][1] - way[k][1]) / brief.cell))
        taut += [(way[k][0] + (way[far][0] - way[k][0]) * t, way[k][1] + (way[far][1] - way[k][1]) * t)
                 for t in np.linspace(0.0, 1.0, n + 1)[1:]]
        k = far
    return taut


def _wind(brief, found, room, way):
    """The way found, from grid step to grid step, as a smooth winding line every ``width / 6`` mm.

    Each point is averaged with its neighbours over ``bend`` mm: longer for
    sweeping curves, shorter to follow the ground closely; the ends stay where
    they are. Points this
    close keep every bend small: a sloping path is level across, so a bend leaves
    a step on its inside of about slope x width / 2 x the bend, a few mm. Where the
    smooth line would leave the room the path needs, the way is simplified to
    within ``route_tolerance`` instead.
    """
    points = np.array(way, dtype=float)
    k = max(1, int(round(brief.bend / 2 / brief.cell)))
    # mirrored past each end, so the average there is even and the line runs into its end without a kink
    n = min(k, len(points) - 1)
    padded = np.concatenate([2 * points[0] - points[n:0:-1], points, 2 * points[-1] - points[-2:-n - 2:-1]])
    smooth = np.array([padded[i:i + 2 * n + 1].mean(axis=0) for i in range(len(points))])
    at = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(smooth, axis=0).T))])
    u = np.linspace(0.0, at[-1], max(1, int(round(at[-1] / (brief.width / 6)))) + 1)
    line = [(float(x), float(y)) for x, y in zip(np.interp(u, at, smooth[:, 0]), np.interp(u, at, smooth[:, 1]))]
    line[0], line[-1] = tuple(way[0]), tuple(way[-1])
    if all(_fits(found, room, p, q) for p, q in zip(line[:-1], line[1:])):
        return line
    return _simplify(list(way), brief.route_tolerance, lambda p, q: _fits(found, room, p, q))


def _graph(brief, found, room, height, weight=None):
    """Every step between neighbouring cells with room, from every heading to every heading.

    A node is a cell and the heading it was reached by, so a step costs its
    length and its climb (times ``weight``) plus how far it turns: ``TURN_COST x
    width x (1 - cos(turn))``, two widths of walking to turn straight back.
    Without it a route could zigzag a cell wide, for nothing, and smoothing
    would undo it.
    """
    nx, ny = room.shape
    index = np.arange(nx * ny).reshape(nx, ny)
    headings = [math.atan2(dj, di) for di, dj in MOVES]
    rows, cols, costs = [], [], []
    for d, (di, dj) in enumerate(MOVES):
        here = (slice(max(-di, 0), nx - max(di, 0)), slice(max(-dj, 0), ny - max(dj, 0)))
        there = (slice(max(di, 0), nx - max(-di, 0)), slice(max(dj, 0), ny - max(-dj, 0)))
        ok = room[here] & room[there]
        step = brief.cell * math.hypot(di, dj)
        grade = np.abs(height[there] - height[here]) / step
        cost = step * (1 + brief.avoid_slope * (grade / brief.ramp_max) ** 2)
        if weight is not None:
            cost = cost * np.maximum(weight[here], weight[there])
        for e, heading in enumerate(headings):    # arriving by heading e, leaving by d
            rows.append(index[here][ok] * len(MOVES) + e)
            cols.append(index[there][ok] * len(MOVES) + d)
            costs.append(cost[ok] + TURN_COST * brief.width * (1 - math.cos(headings[d] - heading)))
    size = nx * ny * len(MOVES)
    return coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)).tocsr()


def _walk(graph, shape, sources):
    """The cheapest ways from any of the cells ``sources`` (flat indices), setting off any way.

    Returns each cell's cost, and a function giving the way to a cell (flat
    index) as grid cells from a source, or None where there is none.
    """
    headings = len(MOVES)
    starts = (np.asarray(sources, dtype=int)[:, None] * headings + np.arange(headings)).ravel()
    cost, before, _ = dijkstra(graph, indices=starts, min_only=True, return_predecessors=True)
    cost = cost.reshape(-1, headings)

    def way(cell):
        if not np.isfinite(cost[cell].min()):
            return None
        states = [int(cell) * headings + int(cost[cell].argmin())]
        while before[states[-1]] >= 0:
            states.append(int(before[states[-1]]))
        return [divmod(s // headings, shape[1]) for s in reversed(states)]

    return cost.min(axis=1).reshape(shape), way


def _cell(found, room, point):
    """The grid cell nearest a stop, which must have room for the path."""
    i, j = int(np.abs(found.x - point[0]).argmin()), int(np.abs(found.y - point[1]).argmin())
    if not room[i, j]:
        raise ValueError("The stop at ({:.0f}, {:.0f}) has no room for the path: it is within half the path's "
                         "width of a hole or the roof's edge.".format(*point))
    return i, j


def _search(graph, shape, start, end, a, b):
    """The cheapest way from cell ``start`` to cell ``end`` on a grid of ``shape``, as cells."""
    _, way = _walk(graph, shape, [np.ravel_multi_index(start, shape)])
    cells = way(np.ravel_multi_index(end, shape))
    if cells is None:
        raise ValueError("No way from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) with room for the path.".format(*a, *b))
    return cells


def _fits(found, room, p, q):
    """Whether the path has room all along the straight line from ``p`` to ``q``."""
    steps = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (found.x[1] - found.x[0]) * 2))
    for t in np.linspace(0.0, 1.0, steps):
        i = int(np.abs(found.x - (p[0] + t * (q[0] - p[0]))).argmin())
        j = int(np.abs(found.y - (p[1] + t * (q[1] - p[1]))).argmin())
        if not room[i, j]:
            return False
    return True


def _simplify(points, tolerance, fits):
    """Douglas-Peucker, keeping a straight line only where it strays less than ``tolerance`` and still ``fits``."""
    a, b = np.array(points[0]), np.array(points[-1])
    if len(points) <= 2:
        return [points[0], points[-1]]
    chord = b - a
    length = float(np.hypot(*chord)) or 1.0
    inner = np.array(points[1:-1])
    off = np.abs(chord[0] * (inner[:, 1] - a[1]) - chord[1] * (inner[:, 0] - a[0])) / length
    k = int(off.argmax()) + 1
    if off[k - 1] <= tolerance and fits(points[0], points[-1]):
        return [points[0], points[-1]]
    return _simplify(points[:k + 1], tolerance, fits)[:-1] + _simplify(points[k:], tolerance, fits)
