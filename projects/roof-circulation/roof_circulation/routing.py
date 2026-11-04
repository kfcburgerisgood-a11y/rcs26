"""Finding routes over a roof: the cheapest way between stops, where climbing costs."""

import math

import numpy as np
from compas.geometry import Polyline
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .drawing import Drawing

__all__ = ["route"]

# a step to any of 16 neighbours, so a route can head 16 ways rather than 8
MOVES = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]
MOVES += [(-a, -b) for a, b in MOVES]


def route(brief, found, drawing):
    """Step 3: each drawing's stops become a path, the cheapest way between them. *Canvas: Shortest Walk.*

    Reads ``width``, ``smoothing``, ``ramp_max``, ``avoid_slope``, ``route_tolerance``,
    and the grid of ``found``, the slope map. A step costs its length, more the
    steeper it climbs *in the direction it goes*: ``avoid_slope x (grade / ramp_max)^2``
    more, on the roof averaged over ``smoothing``. So a route crosses a slope
    along its contours where that is cheaper than going round, as a mountain
    path does. It keeps ``width / 2`` clear of holes and the roof's edge, and is
    simplified to within ``route_tolerance``, keeping every stop as a corner.
    Returns the drawing with each route added to its paths.

    Examples
    --------
    >>> from roof_circulation import Brief, steps
    >>> brief = Brief(roughness=0.0, length=6000.0, span=3000.0)
    >>> stops = Drawing(stops=[Polyline([[1000, 1500, 0], [5000, 1500, 0]])])
    >>> found = route(brief, steps.slopes(brief, steps.host(brief)), stops)
    >>> [list(p)[:2] for p in found.paths[0]]       # on a flat roof: straight
    [[1000.0, 1500.0], [5000.0, 1500.0]]
    """
    room, height = _terrain(brief, found)
    paths, taken = list(drawing.paths), []
    for stops in drawing.stops:
        points = [(float(p[0]), float(p[1])) for p in stops]
        corners = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            # a route never runs over itself or another: what is taken is closed, but around a and b, to leave by
            free = room & (~_near(found, taken, brief.width) | _near(found, [[a], [b]], brief.width + brief.cell))
            graph = _graph(brief, found, free, height)
            way = _search(graph, free.shape[1], _cell(found, free, a), _cell(found, free, b), a, b)
            leg = [a] + [(float(found.x[i]), float(found.y[j])) for i, j in way[1:-1]] + [b]
            leg = _simplify(leg, brief.route_tolerance, lambda p, q: _fits(found, free, p, q))
            corners += leg[1:]
            taken.append(leg)
        paths.append(Polyline([[x, y, 0.0] for x, y in corners]))
    return Drawing(paths, list(drawing.pins), [])


def _near(found, lines, distance):
    """The cells closer than ``distance`` to any of ``lines``, each a list of plan points (one point is a dot)."""
    gx, gy = np.meshgrid(found.x, found.y, indexing="ij")
    near = np.zeros(gx.shape, dtype=bool)
    for line in lines:
        for p, q in zip(line, line[1:] or line):
            (px, py), (dx, dy) = p, (q[0] - p[0], q[1] - p[1])
            t = np.clip(((gx - px) * dx + (gy - py) * dy) / ((dx * dx + dy * dy) or 1.0), 0.0, 1.0)
            near |= np.hypot(gx - px - t * dx, gy - py - t * dy) < distance
    return near


def _terrain(brief, found):
    """Where a path fits, ``width / 2`` clear of holes and edges; and the roof averaged over ``smoothing``."""
    roof = ~np.isnan(found.height)
    room = distance_transform_edt(np.pad(roof, 1))[1:-1, 1:-1] * brief.cell > brief.width / 2
    filled = np.where(roof, found.height, np.nanmean(found.height))
    return room, uniform_filter(filled, size=max(1, int(round(brief.smoothing / brief.cell))), mode="nearest")


def _graph(brief, found, room, height):
    """Every step between neighbouring cells with room, weighted by its length and its climb."""
    nx, ny = room.shape
    index = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, costs = [], [], []
    for di, dj in MOVES:
        here = (slice(max(-di, 0), nx - max(di, 0)), slice(max(-dj, 0), ny - max(dj, 0)))
        there = (slice(max(di, 0), nx - max(-di, 0)), slice(max(dj, 0), ny - max(-dj, 0)))
        ok = room[here] & room[there]
        step = brief.cell * math.hypot(di, dj)
        grade = np.abs(height[there] - height[here]) / step
        cost = step * (1 + brief.avoid_slope * (grade / brief.ramp_max) ** 2)
        rows.append(index[here][ok])
        cols.append(index[there][ok])
        costs.append(cost[ok])
    return coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny)).tocsr()


def _cell(found, room, point):
    """The grid cell nearest a stop, which must have room for the path."""
    i, j = int(np.abs(found.x - point[0]).argmin()), int(np.abs(found.y - point[1]).argmin())
    if not room[i, j]:
        raise ValueError("The stop at ({:.0f}, {:.0f}) has no room for the path: it is within half the path's "
                         "width of a hole or the roof's edge.".format(*point))
    return i, j


def _search(graph, ny, start, end, a, b):
    """The cheapest way from cell ``start`` to cell ``end``, as cells; ``ny`` cells to a grid row."""
    first, last = start[0] * ny + start[1], end[0] * ny + end[1]
    _, before = dijkstra(graph, indices=first, return_predecessors=True)
    if last != first and before[last] < 0:
        raise ValueError("No way from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) with room for the path.".format(*a, *b))
    way = [last]
    while way[-1] != first:
        way.append(before[way[-1]])
    return [divmod(int(k), ny) for k in reversed(way)]


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
