"""The workflow: one function per step, each one component on a canvas.

A step takes the brief and what earlier steps made, and returns its own result.
Build steps from ``compas_vol.nodes`` where a component exists, and keep Rhino
out of this package: that is what lets a step become a component later.

The module reads any host as a field: ``ground`` finds its surface under each
drawn path, ``grade`` lays a walking level on it that is never steeper than
``ramp_max``, and ``circulation`` hands back a :class:`Contribution`.
:func:`host` is only a stand-in rocky roof to try it on.
"""

import math
import os
from dataclasses import dataclass, field, replace

import compas
import numpy as np
from compas.geometry import Frame, Plane

from compas_vol import nodes
from compas_vol.meshing import write_stl
from compas_vol.primitives import VolPlane

from .contribution import Contribution, combine
from .rock import VolRock
from .routing import network, route

__all__ = [
    "Profile", "Segment", "SlopeMap", "host", "slopes", "route", "network", "ground", "grade", "rest", "segments",
    "walk", "clearance", "support", "circulation", "combine", "extent", "export",
]

OVERLAP = 1.0   # mm: neighbouring pieces overlap this much, so the seam between them never reads as a surface
SHARPEST = 160  # degrees: a sharper turn doubles back, and its two arms would lie on each other
MOST_RESTS = 200   # rests on one path before a rest brief's rise is taken to be a mistake
GENTLE = 8      # degrees: a bend this gentle climbs on as it turns; a sharper one is a level landing.
# A sloping path is level across, so a bend leaves a step on its inside of slope x width / 2 x sin(bend):
# at 8 degrees and 1:12, about 7 mm on a 1200 path, under the 10 mm path_clear allows.


@dataclass
class Profile:
    """One path, followed every ``spacing`` mm in plan: where it goes, the host under it, and the level walked on."""

    s: np.ndarray           # distance along the path, in plan
    x: np.ndarray
    y: np.ndarray
    ground: np.ndarray      # the host's top under the centre line
    floor: np.ndarray       # its lowest top across the width
    holds: list             # (from s, to s, level): ends and turns held level, and pins
    level: np.ndarray = None
    rests: list = field(default_factory=list)   # (from s, to s, side, bay bottom, bay depth); side 0 for no bay

    @property
    def length(self):
        return float(self.s[-1])


@dataclass
class Segment:
    """One straight piece of a path, at a single slope, built as one box."""

    start: tuple            # (x, y, walking level)
    end: tuple
    bottom: float           # how low its solid reaches: into the host below it
    width: float = None     # its own width, for a bay; the brief's, otherwise

    @property
    def run(self):
        """float: plan length."""
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def rise(self):
        return self.end[2] - self.start[2]

    @property
    def direction(self):
        """The plan unit vector from start to end."""
        return (self.end[0] - self.start[0]) / self.run, (self.end[1] - self.start[1]) / self.run

    def points(self, spacing):
        """``(x, y, level)`` along the centre line, about every ``spacing`` mm.

        Examples
        --------
        >>> Segment((0, 0, 0), (1000, 0, 50), -150).points(500)
        [(250.0, 0.0, 12.5), (750.0, 0.0, 37.5)]
        """
        (x, y, z), (ux, uy) = self.start, self.direction
        count = max(2, int(self.run // spacing))
        return [(x + ux * u, y + uy * u, z + self.rise * u / self.run)
                for u in (self.run * (i + 0.5) / count for i in range(count))]


def host(brief):
    """A stand-in rocky roof, to try the module on. Any field will do in its place: a mesh, through VolGrid."""
    return VolRock(brief.length, brief.span, brief.thickness, brief.roughness, brief.rock_cell, brief.rock_seed)


def _surface(brief, host, x, y, holes=False):
    """The host's highest surface over plan points: scanned down every ``voxel`` mm, then bisected.

    Where there is no host, refuses, or with ``holes`` gives NaN.
    """
    zs = np.arange(brief.search_top, brief.search_bottom, -brief.voxel)
    inside = host.get_distance_numpy(x[:, None, None], y[:, None, None], zs[None, None, :])[:, 0, :] < 0
    found = inside.any(axis=1)
    if not found.all() and not holes:
        i = int(np.argmin(found))
        raise ValueError("No host under the path at ({:.0f}, {:.0f}).".format(x[i], y[i]))
    first = inside.argmax(axis=1)
    lo, hi = zs[first], zs[np.maximum(first - 1, 0)]
    for _ in range(12):
        mid = (lo + hi) / 2
        below = host.get_distance_numpy(x[:, None, None], y[:, None, None], mid[:, None, None])[:, 0, 0] < 0
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    mid = (lo + hi) / 2
    # one step by the distance itself: exact where the field is, and never past the surface where it is a bound
    top = mid - host.get_distance_numpy(x[:, None, None], y[:, None, None], mid[:, None, None])[:, 0, 0]
    return np.where(found, top, np.nan)


@dataclass
class SlopeMap:
    """The host's top on a plan grid, and how steep it is there."""

    x: np.ndarray           # grid lines, along X
    y: np.ndarray           # and along Y
    height: np.ndarray      # the top, shape (len(x), len(y)); NaN where there is no host
    slope: np.ndarray       # rise over run, steepest way; NaN where there is no host

    def walkable(self, ramp_max):
        """Where the top is no steeper than ``ramp_max``: where a path could follow it as it is."""
        return np.nan_to_num(self.slope, nan=np.inf) <= ramp_max


def slopes(brief, host):
    """Step 2: the host's slope over its plan, every ``cell`` mm. *Canvas: a mesh coloured by slope.*

    Reads ``length``, ``span``, ``cell``, and the search numbers. Samples the plan
    box of ``length`` by ``span``, so give them the real roof's plan. Holes, such
    as another module's light openings, come back NaN: not walkable.

    Examples
    --------
    >>> from roof_circulation import Brief
    >>> brief = Brief(roughness=0.0, length=3000.0, span=2000.0)
    >>> found = slopes(brief, host(brief))
    >>> float(np.nanmax(found.slope)), bool(found.walkable(brief.ramp_max).all())
    (0.0, True)
    """
    x = np.arange(brief.cell / 2, brief.length, brief.cell)
    y = np.arange(brief.cell / 2, brief.span, brief.cell)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    height = _surface(brief, host, gx.ravel(), gy.ravel(), holes=True).reshape(gx.shape)
    dx, dy = np.gradient(height, brief.cell, brief.cell)
    return SlopeMap(x, y, height, np.hypot(dx, dy))


def ground(brief, host, drawing):
    """Step 2: the host's surface under every path. *Canvas: Divide Curve, then a ray down onto the host.*

    Reads ``width``, ``spacing``, ``voxel``, ``search_top``, ``search_bottom``. Each path
    is followed every ``spacing`` mm, and at every vertex and pin. Its ends and
    turns sharper than ``GENTLE`` are held level ``width / 2`` each way, further at a
    sharp turn, where the two arms overlap: ``width / 2 x tan(turn / 2)``; a gentler
    bend climbs on as it turns. Where paths meet at an end, they are held level
    as far as each overlaps the most acute of the others. Each pin is held at its Z.
    Returns one :class:`Profile` per path, without a level yet.
    """
    r = brief.width / 2
    profiles, placed = [], set()
    arms = _arms(drawing.paths)
    for n, path in enumerate(drawing.paths):
        corners = np.array([[p[0], p[1]] for p in path], dtype=float)
        legs = np.hypot(*np.diff(corners, axis=0).T)
        if not legs.all():
            raise ValueError("A path doubles back on itself at {}.".format(corners[1:][legs == 0][0].tolist()))
        at = np.concatenate([[0.0], np.cumsum(legs)])
        total = at[-1]
        first, last = _end_reach(r, arms, n, 0, path[0]), _end_reach(r, arms, n, -1, path[-1])
        holds = [(0.0, min(first, total), None), (max(total - last, 0.0), total, None)]
        for i in range(1, len(corners) - 1):
            a, b = corners[i] - corners[i - 1], corners[i + 1] - corners[i]
            turn = math.acos(float(np.clip(a @ b / (legs[i - 1] * legs[i]), -1.0, 1.0)))
            if turn > math.radians(GENTLE):     # a gentler bend climbs on as it turns
                if turn > math.radians(SHARPEST):
                    raise ValueError("The path turns back on itself at ({:.0f}, {:.0f}), by {:.0f} degrees; "
                                     "at most {} can be walked.".format(*corners[i], math.degrees(turn), SHARPEST))
                # the two arms overlap this far from a sharp corner: level all of it, past the next corner if need be
                reach = r * max(1.0, math.tan(turn / 2))
                holds.append((max(at[i] - reach, 0.0), min(at[i] + reach, total), None))
        for k, pin in enumerate(drawing.pins):
            s, off = _nearest(corners, at, pin)
            if off <= r:
                holds.append((s, s, float(pin[2])))
                placed.add(k)
        marks = [h[0] for h in holds] + [h[1] for h in holds] + list(at)
        # rounded so that marks a hair apart are one; clipped, not dropped, so the ends always stay
        s = np.unique(np.clip(np.round(np.concatenate([np.arange(0.0, total, brief.spacing), [total], marks]), 6),
                              0.0, total))
        x, y = np.interp(s, at, corners[:, 0]), np.interp(s, at, corners[:, 1])
        tx, ty = np.gradient(x, s), np.gradient(y, s)
        norm = np.hypot(tx, ty)
        nx, ny = -ty / norm, tx / norm
        centre = _surface(brief, host, x, y)
        edges = [_surface(brief, host, x + sign * r * nx, y + sign * r * ny) for sign in (-1, 1)]
        profiles.append(Profile(s, x, y, centre, np.minimum(centre, np.minimum(*edges)), holds))
    for k, pin in enumerate(drawing.pins):
        if k not in placed:
            raise ValueError("The pin at ({:.0f}, {:.0f}) is not on any path.".format(pin[0], pin[1]))
    return profiles


def _arms(paths):
    """Where paths end: for each end point, ``(path, end, direction)`` of every path leaving it."""
    arms = {}
    for n, path in enumerate(paths):
        for end, (a, b) in ((0, (path[0], path[1])), (-1, (path[-1], path[-2]))):
            d = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
            arms.setdefault((round(a[0], 3), round(a[1], 3)), []).append((n, end, d / (np.hypot(*d) or 1.0)))
    return arms


def _end_reach(r, arms, n, end, point):
    """How far end ``end`` of path ``n`` is held level: ``r``, or as far as it overlaps another path leaving there."""
    here = arms[(round(point[0], 3), round(point[1], 3))]
    own = next(d for m, e, d in here if (m, e) == (n, end))
    reach = r
    for m, e, other in here:
        if (m, e) == (n, end):
            continue
        between = math.acos(float(np.clip(own @ other, -1.0, 1.0)))
        if between < math.radians(180 - SHARPEST):
            raise ValueError("Two paths leave ({:.0f}, {:.0f}) {:.0f} degrees apart: at least {} can be walked."
                             .format(point[0], point[1], math.degrees(between), 180 - SHARPEST))
        reach = max(reach, r / math.tan(between / 2))
    return reach


def _nearest(corners, at, point):
    """Distance along a plan polyline to its nearest point to ``point``, and how far off it ``point`` is."""
    p = np.array([point[0], point[1]], dtype=float)
    best = (0.0, math.inf)
    for i, (a, b) in enumerate(zip(corners[:-1], corners[1:])):
        d = b - a
        t = float(np.clip((p - a) @ d / (d @ d), 0.0, 1.0))
        off = float(np.hypot(*(a + t * d - p)))
        if off < best[1]:
            best = (at[i] + t * float(np.hypot(*d)), off)
    return best


def grade(brief, profiles):
    """Step 3: the level walked on. Never steeper than ``ramp_max``, and never stairs.

    Reads ``ramp_max``, ``smoothing``, ``lift``, ``wave``, ``wavelength``. The level
    follows the ground averaged over ``smoothing`` mm and raised by ``lift``, plus a
    wave that fades out at the ends; it meets the ground (and lift) at both ends,
    keeps every pin as set, and is level
    through every turn. Where it cannot do all that at ``ramp_max``, it says
    where, so the route can be drawn longer.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> brief = Brief(roughness=0.0, wave=50.0)
    >>> profile = grade(brief, ground(brief, host(brief), Drawing.sample()))[0]
    >>> float(profile.level[0]), float(profile.level[-1])    # both ends on the flat roof, at 600
    (600.0, 600.0)
    >>> bool(np.all(np.abs(np.diff(profile.level)) <= brief.ramp_max * np.diff(profile.s) + 1e-9))
    True
    """
    m = brief.ramp_max
    result = []
    for p in profiles:
        s, total = p.s, p.length
        window = np.abs(s[:, None] - s[None, :]) <= brief.smoothing / 2
        share = np.gradient(s)[None, :] * window
        target = (share * p.ground[None, :]).sum(axis=1) / share.sum(axis=1) + brief.lift
        if brief.wave:
            fade = np.clip(np.minimum(s, total - s) / (brief.wavelength / 2), 0.0, 1.0)
            target = target + brief.wave * np.sin(2 * math.pi * s / brief.wavelength) * fade * fade * (3 - 2 * fade)
        # landings climb nothing: every distance below is measured in ramp only, landings taken out
        flats = _merged([(a, b) for a, b, _ in p.holds if a < b])
        ramp = s - sum(np.clip(s - a, 0.0, b - a) for a, b in flats)
        held = []       # (ramp distance, level, s), each within reach of all those before it
        order = p.holds[:2] + [h for h in p.holds[2:] if h[2] is not None] + [h for h in p.holds[2:] if h[2] is None]
        for k, (a, b, level) in enumerate(order):
            if k < 2:   # the ends: on the ground
                level = float(p.ground[0] if k == 0 else p.ground[-1]) + brief.lift
            here = (a + b) / 2
            r = float(np.interp(here, s, ramp))
            reach = [(c, m * abs(r - c[0])) for c in held]
            if level is None:
                lo = max([c[1] - d for c, d in reach] + [-math.inf])
                hi = min([c[1] + d for c, d in reach] + [math.inf])
                level = float(np.clip(np.interp(here, s, target), lo, hi))
            for c, d in reach:
                if abs(level - c[1]) > d + 1e-6:
                    raise ValueError(_too_steep(p, (here, r, level), (c[2], c[0], c[1]), m))
            held.append((r, level, here))
        lo = np.max([h - m * np.abs(ramp - r) for r, h, _ in held], axis=0)
        hi = np.min([h + m * np.abs(ramp - r) for r, h, _ in held], axis=0)
        z = np.clip(target, lo, hi)
        apart = m * np.abs(ramp[:, None] - ramp[None, :])
        z = ((z[None, :] + apart).min(axis=1) + (z[None, :] - apart).max(axis=1)) / 2
        result.append(replace(p, level=z))
    return result


def rest(brief, landing, host, profiles, drawing=None):
    """Step 4b, optional: a rest landing wherever a path would climb more than ``landing.rise`` without one.

    *Canvas: its own component, fed a* :class:`RestBrief`; *leave it out for no rests.* Reads
    ``landing`` (``rise``, ``length``, ``bay``), and ``influence`` and what
    ``grade`` reads. Climb is counted along one continuous climb, up or down: it
    starts again where the path turns the other way, or runs at 1:50 or gentler
    (as steep as a landing may be) for ``length`` or more. A climb too long
    gets as few landings as divide it into parts of at most ``rise``, at
    levels that divide it evenly; each goes where there is room to climb to it and
    on from it, by the drawing's attractor if one is near, else on the flattest
    ground. The path is graded again and the next climb found, until none climbs
    too far. Each landing gets a bay ``bay`` deep beside it, on whichever side
    needs less cut and fill. Returns the profiles, graded, with their rests;
    ``rise = 0`` leaves them be.

    Examples
    --------
    >>> from compas.geometry import Point, Polyline
    >>> from roof_circulation import Brief, Drawing
    >>> from roof_circulation import RestBrief
    >>> brief = Brief(roughness=0.0)
    >>> drawing = Drawing([Polyline([[1000, 2000, 0], [17000, 2000, 0]])], [Point(9000, 2000, 1050)])
    >>> roof = host(brief)
    >>> profile = rest(brief, RestBrief(rise=300.0), roof, grade(brief, ground(brief, roof, drawing)))[0]
    >>> len(profile.rests)                  # 450 up and 450 down, a rest every 300: one each way
    2
    """
    if landing.rise <= 0:
        return profiles
    lures = list(drawing.attractors) if drawing is not None else []
    result = []
    for p in profiles:
        # first between the levels held fixed, the ends and the pins, where the climb is known in full
        fixed = sorted([(0.0, float(p.ground[0])), (p.length, float(p.ground[-1]))]
                       + [(a, level) for a, b, level in p.holds[2:] if level is not None and a == b])
        held, rests = replace(p, rests=[]), []
        for foot, head in zip(fixed[:-1], fixed[1:]):
            held, rests = _place(brief, landing, host, held, rests, foot, head, lures)
        p = replace(grade(brief, [held])[0], rests=rests)
        # then wherever following the ground still climbs too far
        for _ in range(MOST_RESTS):
            climb = _long_climb(landing, p)
            if climb is None:
                break
            held, rests = _place(brief, landing, host, p, list(p.rests), *climb, lures)
            p = replace(grade(brief, [held])[0], rests=rests)
        else:
            raise ValueError("More than {} rests on one path: is the rise too small?".format(MOST_RESTS))
        result.append(p)
    return result


def _place(brief, landing, host, held, rests, foot, head, lures):
    """Rests on the climb from ``foot`` to ``head``, each ``(s, level)``: as few as divide it into parts of at most
    ``rise``, at levels dividing it evenly, each where there is ramp to climb to it and on past the rest.

    Returns the profile held at them, ungraded, and the rests so far.
    """
    (s0, z0), (s1, z1) = foot, head
    m = brief.ramp_max
    n = int(math.ceil(abs(z1 - z0) / landing.rise - 1e-9)) - 1     # as few as will do
    after = s0
    for k in range(1, n + 1):
        level = z0 + (z1 - z0) * k / (n + 1)
        # measured in ramp, landings taken out: room to climb up to it, and on from it past the rests after
        flats = _merged([(a, b) for a, b, _ in held.holds if a < b])
        ramp = held.s - sum(np.clip(held.s - a, 0.0, b - a) for a, b in flats)
        lo = np.interp(s0, held.s, ramp) + abs(level - z0) / m
        hi = np.interp(s1, held.s, ramp) - abs(z1 - level) / m - (n - k + 1) * landing.length
        clear = np.array([all(s + landing.length <= a or s >= b for a, b in flats) for s in held.s])
        fits = (ramp >= lo - 1e-6) & (ramp <= hi + 1e-6) & (held.s >= after - 1e-6) & clear
        if not fits.any():
            have = np.interp(s1, held.s, ramp) - np.interp(s0, held.s, ramp) - n * landing.length
            raise ValueError(_too_steep(held, (s0, 0.0, z0), (s1, have, z1), m)
                             + " That is with {} rest landings of {:.0f}.".format(n, landing.length))
        start = _rest_at(brief, landing, held, held.s[fits], lures)
        after = start + landing.length
        rests = rests + [(start, after) + _bay(brief, landing, host, held, (start + after) / 2, level)]
        held = _held(held, start, after, level)
    return held, rests


def _long_climb(landing, p):
    """The first climb that rises more than ``rise``: ``((s, level) at its foot, (s, level) at its head)``.

    The foot is where the level stretch below it begins and the head where the
    one above it ends: the climb may use all of the way between.
    """
    def level_from(i):     # where the level stretch from i on ends
        while i < len(p.s) - 1 and abs(p.level[i + 1] - p.level[i]) <= (p.s[i + 1] - p.s[i]) / 50 + 1e-9:
            i += 1
        return i

    def climb(a, b):
        if abs(p.level[b] - p.level[a]) > landing.rise + 1e-6:
            b = level_from(b)
            return (p.s[a], p.level[a]), (p.s[b], p.level[b])
        return None

    start, sign, flat = 0, 0, None
    for i in range(len(p.s) - 1):
        ds, dz = p.s[i + 1] - p.s[i], p.level[i + 1] - p.level[i]
        if abs(dz) <= ds / 50 + 1e-9:         # as steep as a landing may be
            flat = i if flat is None else flat
            if sign and p.s[i + 1] - p.s[flat] >= landing.length - 1e-6:     # a long enough rest ends it
                found = climb(start, flat)
                if found:
                    return found
                sign = 0
            continue
        heading = 1 if dz > 0 else -1
        if sign and heading != sign:                                            # turning the other way ends it
            found = climb(start, i)
            if found:
                return found
            sign = 0
        if not sign:
            start, sign = (i if flat is None else flat), heading               # from the foot of the level below
        flat = None
    return climb(start, len(p.s) - 1) if sign else None


def _rest_at(brief, landing, p, where, lures):
    """Which of ``where`` a rest landing starts at: by an attractor if one is near, else on the flattest ground."""
    x, y = np.interp(where, p.s, p.x), np.interp(where, p.s, p.y)
    near = [d for d in (np.hypot(x - q[0], y - q[1]) for q in lures) if d.min() <= brief.influence]
    if near:
        return float(where[min(near, key=lambda d: d.min()).argmin()])
    rise = np.abs(np.interp(where + landing.length, p.s, p.ground) - np.interp(where, p.s, p.ground))
    return float(where[rise.argmin()])


def _held(p, a, b, level=None):
    """The profile followed at ``a`` and ``b`` too, and held level between them, at ``level`` if given; ungraded."""
    s = np.unique(np.concatenate([p.s, [a, b]]))
    def at(v): return np.interp(s, p.s, v)
    return replace(p, s=s, x=at(p.x), y=at(p.y), ground=at(p.ground), floor=at(p.floor),
                   holds=p.holds + [(a, b, level)], level=None)


def _bay(brief, landing, host, p, s, level):
    """Which side of the path a rest's bay goes, and how low it reaches: ``(side, bottom)``, side 0 for none.

    The side whose ground lies nearer the landing's level, so less is cut or filled; never off the host.
    """
    if landing.bay <= 0:
        return 0, None, 0.0
    i = int(np.clip(np.searchsorted(p.s, s), 1, len(p.s) - 1))
    dx, dy = p.x[i] - p.x[i - 1], p.y[i] - p.y[i - 1]
    nx, ny = -dy / math.hypot(dx, dy), dx / math.hypot(dx, dy)
    x, y = np.interp(s, p.s, p.x), np.interp(s, p.s, p.y)
    best = (math.inf, 0, None)
    for side in (1, -1):
        reach = np.array([brief.width / 2 + landing.bay / 2, brief.width / 2 + landing.bay])
        top = _surface(brief, host, x + side * reach * nx, y + side * reach * ny, holes=True)
        if not np.isnan(top).any() and abs(top[0] - level) < best[0]:
            best = (abs(top[0] - level), side, float(min(top.min(), level)) - brief.depth)
    return best[1], best[2], landing.bay


def _merged(spans):
    """Overlapping ``(from, to)`` spans joined."""
    result = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def _too_steep(p, one, other, m):
    """Where the route is too short to climb at ``m``, in words a designer can act on."""
    (sa, ra, ha), (sb, rb, hb) = sorted([one, other])
    xa, ya, xb, yb = (float(np.interp(v, p.s, c)) for v, c in ((sa, p.x), (sa, p.y), (sb, p.x), (sb, p.y)))
    return ("Too steep for 1:{:.0f} from ({:.0f}, {:.0f}) at {:.0f} to ({:.0f}, {:.0f}) at {:.0f}: "
            "a rise of {:.0f} mm needs {:.0f} mm of ramp, and the route has {:.0f} between its landings. "
            "Draw it longer there.").format(1 / m, xa, ya, ha, xb, yb, hb, abs(hb - ha), abs(hb - ha) / m, abs(rb - ra))


def segments(brief, profiles):
    """Each graded path as straight pieces, one per slope, each reaching ``depth`` into the host below it.

    Reads ``depth``, ``width``. Returns one list of :class:`Segment` per path,
    then one per rest bay: a level piece as wide as the bay is deep, beside its
    landing, rounded at its ends, as long as the landing in all.
    """
    result = []
    for p in profiles:
        runs = []
        for i in range(len(p.s) - 1):
            # depth under the host or under the walking surface, whichever is lower: a deep cut stands on its own too
            bottom = min(p.floor[i], p.floor[i + 1], p.level[i], p.level[i + 1]) - brief.depth
            seg = Segment((p.x[i], p.y[i], p.level[i]), (p.x[i + 1], p.y[i + 1], p.level[i + 1]), bottom)
            last = runs[-1] if runs else None
            if last and _same_line(last, seg):
                runs[-1] = Segment(last.start, seg.end, min(last.bottom, bottom))
            else:
                runs.append(seg)
        result.append(runs)
    for p in profiles:
        for a, b, side, bottom, bay in p.rests:
            if not side:
                continue
            mid, half = (a + b) / 2, max(b - a - bay, 2.0) / 2   # its round ends make up the rest
            dx, dy = np.interp(b, p.s, p.x) - np.interp(a, p.s, p.x), np.interp(b, p.s, p.y) - np.interp(a, p.s, p.y)
            ux, uy = dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)
            aside = side * ((brief.width + bay) / 2 - OVERLAP)
            x, y = np.interp(mid, p.s, p.x) - uy * aside, np.interp(mid, p.s, p.y) + ux * aside
            level = float(np.interp(mid, p.s, p.level))
            result.append([Segment((x - ux * half, y - uy * half, level), (x + ux * half, y + uy * half, level),
                                   bottom, bay)])
    return result


def _straight(a, b):
    """Whether the path keeps its direction in plan from piece ``a`` to piece ``b``."""
    (ax, ay), (bx, by) = a.direction, b.direction
    return ax * bx + ay * by > 1 - 1e-9


def _bend(a, b):
    """How far the path turns in plan from piece ``a`` to piece ``b``, in radians."""
    (ax, ay), (bx, by) = a.direction, b.direction
    return math.acos(min(max(ax * bx + ay * by, -1.0), 1.0))


def _same_line(a, b):
    """Whether two pieces continue each other: one direction in plan, one slope."""
    return _straight(a, b) and abs(a.rise / a.run - b.rise / b.run) < 1e-9


def _band(brief, runs, below, above):
    """The solid from ``below`` under the walking surface (or down to each piece's bottom) to ``above`` over it."""
    parts = []
    r = brief.width / 2

    def reach(a, b):
        """How far a piece runs on past its end into a gentle bend, to close the bend's outside: a mitre."""
        turn = _bend(a, b)
        return OVERLAP + (r * math.tan(turn / 2) if turn <= math.radians(GENTLE) else 0.0)

    for i, seg in enumerate(runs):
        (x, y, z), (ux, uy) = seg.start, seg.direction
        back = reach(runs[i - 1], seg) if i > 0 else OVERLAP
        on = reach(seg, runs[i + 1]) if i + 1 < len(runs) else OVERLAP
        slope = seg.rise / seg.run
        up = [-ux * slope, -uy * slope, 1]
        if below is None:   # down to a level bottom, in the host
            lo, underside = seg.bottom, VolPlane(Plane([x, y, seg.bottom], [0, 0, -1]))
        else:               # a band of even depth under the walking surface
            lo, underside = min(seg.start[2], seg.end[2]) - below, VolPlane(Plane([x, y, z - below], [-v for v in up]))
        hi = max(seg.start[2], seg.end[2]) + above
        middle = (seg.run + on - back) / 2
        frame = Frame([x + ux * middle, y + uy * middle, (lo + hi) / 2], [ux, uy, 0], [-uy, ux, 0])
        parts.append(nodes.boolean([
            nodes.box(frame, seg.run + back + on, seg.width or brief.width, hi - lo, 0),
            VolPlane(Plane([x, y, z + above], up)),
            underside,
        ], "intersection", 0))
    # a round joint where the path turns sharply or ends; the grade holds it level there
    for i in range(len(runs) + 1):
        touching = runs[max(i - 1, 0):i + 1]
        if len(touching) == 2 and _bend(*touching) <= math.radians(GENTLE):
            continue
        x, y, z = runs[i].start if i < len(runs) else runs[-1].end
        down = z - min(t.bottom for t in touching) if below is None else below
        frame = Frame([x, y, z + (above - down) / 2], [1, 0, 0], [0, 1, 0])
        parts.append(nodes.cylinder(frame, (touching[0].width or brief.width) / 2, down + above))
    return nodes.boolean(parts, "union", 0)


def walk(brief, segments):
    """The path as solid, from its walking surface down into the host. *Canvas: Volumetric Boolean.*"""
    return nodes.boolean([_band(brief, runs, None, 0.0) for runs in segments], "union", 0)


def clearance(brief, segments, walk):
    """The space kept empty over the path, ``headroom`` high, less the path itself: where it cuts into the host."""
    space = nodes.boolean([_band(brief, runs, 0.0, brief.headroom) for runs in segments], "union", 0)
    return nodes.boolean([space, walk], "difference", 0)


def support(brief, segments):
    """The top ``support`` mm under the walking surface, which no later process may remove."""
    return nodes.boolean([_band(brief, runs, brief.support, 0.0) for runs in segments], "union", 0)


def circulation(brief, segments):
    """The module's result, for any host: a :class:`Contribution`.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> brief = Brief(roughness=0.0)
    >>> runs = segments(brief, grade(brief, ground(brief, host(brief), Drawing.sample())))
    >>> part = circulation(brief, runs)
    >>> round(part.add.get_distance([5000, 2000, 700]), 3)     # 100 mm over the path, on the flat roof at 600
    100.0
    >>> part.cut.get_distance([5000, 2000, 1600]) < 0         # a metre above it
    True
    """
    solid = walk(brief, segments)
    return Contribution(add=solid, cut=clearance(brief, segments, solid), keep=support(brief, segments))


def extent(brief, profiles):
    """The box to sample: the host and every path, ``margin`` around. ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    Before the grade, the paths are taken at the ground's height.
    """
    r = brief.width / 2 + max([rest[4] for p in profiles for rest in p.rests] + [0.0])   # bays beside paths
    xs = np.concatenate([[0.0, brief.length]] + [np.concatenate([p.x - r, p.x + r]) for p in profiles])
    ys = np.concatenate([[0.0, brief.span]] + [np.concatenate([p.y - r, p.y + r]) for p in profiles])
    zs = np.concatenate([[0.0, brief.thickness + brief.roughness]] +
                        [np.concatenate([p.floor - brief.depth, p.ground if p.level is None else p.level])
                         for p in profiles])
    lo, hi = (xs.min(), ys.min(), zs.min()), (xs.max(), ys.max(), zs.max())
    return tuple(float(v) - brief.margin for v in lo) + tuple(float(v) + brief.margin for v in hi)


def export(brief, solid, bounds, stem):
    """Write ``<stem>.stl`` (the mesh) and ``<stem>.json`` (the field), sampled every ``voxel`` mm.

    Returns ``(vertices, faces)``. *Canvas: Volumetric Mesh.*
    """
    folder = os.path.dirname(stem)
    if folder:
        os.makedirs(folder, exist_ok=True)
    resolution = tuple(int(math.ceil((bounds[i + 3] - bounds[i]) / brief.voxel)) + 1 for i in range(3))
    vertices, faces = nodes.mesh(solid, bounds, resolution, 0)
    write_stl(stem + ".stl", vertices, faces)
    compas.json_dump(solid, stem + ".json")
    return vertices, faces
