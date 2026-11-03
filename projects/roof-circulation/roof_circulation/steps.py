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
from dataclasses import dataclass, replace

import compas
import numpy as np
from compas.geometry import Frame, Plane

from compas_vol import nodes
from compas_vol.meshing import write_stl
from compas_vol.primitives import VolPlane

from .contribution import Contribution, combine
from .rock import VolRock

__all__ = [
    "Profile", "Segment", "host", "ground", "grade", "segments", "walk", "clearance", "support", "circulation",
    "combine", "extent", "export",
]

OVERLAP = 1.0   # mm: neighbouring pieces overlap this much, so the seam between them never reads as a surface


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

    @property
    def length(self):
        return float(self.s[-1])


@dataclass
class Segment:
    """One straight piece of a path, at a single slope, built as one box."""

    start: tuple            # (x, y, walking level)
    end: tuple
    bottom: float           # how low its solid reaches: into the host below it

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


def _surface(brief, host, x, y):
    """The host's highest surface over plan points: scanned down every ``voxel`` mm, then bisected."""
    zs = np.arange(brief.search_top, brief.search_bottom, -brief.voxel)
    inside = host.get_distance_numpy(x[:, None, None], y[:, None, None], zs[None, None, :])[:, 0, :] < 0
    if not inside.any(axis=1).all():
        i = int(np.argmin(inside.any(axis=1)))
        raise ValueError("No host under the path at ({:.0f}, {:.0f}).".format(x[i], y[i]))
    first = inside.argmax(axis=1)
    lo, hi = zs[first], zs[np.maximum(first - 1, 0)]
    for _ in range(12):
        mid = (lo + hi) / 2
        below = host.get_distance_numpy(x[:, None, None], y[:, None, None], mid[:, None, None])[:, 0, 0] < 0
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    mid = (lo + hi) / 2
    # one step by the distance itself: exact where the field is, and never past the surface where it is a bound
    return mid - host.get_distance_numpy(x[:, None, None], y[:, None, None], mid[:, None, None])[:, 0, 0]


def ground(brief, host, drawing):
    """Step 2: the host's surface under every path. *Canvas: Divide Curve, then a ray down onto the host.*

    Reads ``width``, ``spacing``, ``voxel``, ``search_top``, ``search_bottom``. Each path
    is followed every ``spacing`` mm, and at every vertex and pin. Its ends and
    turns are held level ``width / 2`` each way, and each pin at its Z.
    Returns one :class:`Profile` per path, without a level yet.
    """
    r = brief.width / 2
    profiles, placed = [], set()
    for path in drawing.paths:
        corners = np.array([[p[0], p[1]] for p in path], dtype=float)
        legs = np.hypot(*np.diff(corners, axis=0).T)
        if not legs.all():
            raise ValueError("A path doubles back on itself at {}.".format(corners[1:][legs == 0][0].tolist()))
        at = np.concatenate([[0.0], np.cumsum(legs)])
        total = at[-1]
        holds = [(0.0, min(r, total), None), (max(total - r, 0.0), total, None)]
        for i in range(1, len(corners) - 1):
            a, b = corners[i] - corners[i - 1], corners[i + 1] - corners[i]
            if a @ b < (1 - 1e-9) * legs[i - 1] * legs[i]:
                holds.append((at[i] - r, at[i] + r, None))
        for k, pin in enumerate(drawing.pins):
            s, off = _nearest(corners, at, pin)
            if off <= r:
                holds.append((s, s, float(pin[2])))
                placed.add(k)
        marks = [h[0] for h in holds] + [h[1] for h in holds] + list(at)
        s = np.unique(np.round(np.concatenate([np.arange(0.0, total, brief.spacing), [total], marks]), 6))
        s = s[(s >= 0) & (s <= total)]
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

    Reads ``ramp_max``, ``smoothing``, ``wave``, ``wavelength``. The level follows
    the ground averaged over ``smoothing`` mm, plus a wave that fades out at the
    ends; it meets the ground at both ends, keeps every pin, and is level
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
        target = (share * p.ground[None, :]).sum(axis=1) / share.sum(axis=1)
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
                level = float(p.ground[0] if k == 0 else p.ground[-1])
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

    Reads ``depth``. Returns one list of :class:`Segment` per path.
    """
    result = []
    for p in profiles:
        runs = []
        for i in range(len(p.s) - 1):
            bottom = min(p.floor[i], p.floor[i + 1]) - brief.depth
            seg = Segment((p.x[i], p.y[i], p.level[i]), (p.x[i + 1], p.y[i + 1], p.level[i + 1]), bottom)
            last = runs[-1] if runs else None
            if last and _same_line(last, seg):
                runs[-1] = Segment(last.start, seg.end, min(last.bottom, bottom))
            else:
                runs.append(seg)
        result.append(runs)
    return result


def _straight(a, b):
    """Whether the path keeps its direction in plan from piece ``a`` to piece ``b``."""
    (ax, ay), (bx, by) = a.direction, b.direction
    return ax * bx + ay * by > 1 - 1e-9


def _same_line(a, b):
    """Whether two pieces continue each other: one direction in plan, one slope."""
    return _straight(a, b) and abs(a.rise / a.run - b.rise / b.run) < 1e-9


def _band(brief, runs, below, above):
    """The solid from ``below`` under the walking surface (or down to each piece's bottom) to ``above`` over it."""
    parts = []
    for seg in runs:
        (x, y, z), (ux, uy) = seg.start, seg.direction
        slope = seg.rise / seg.run
        up = [-ux * slope, -uy * slope, 1]
        if below is None:   # down to a level bottom, in the host
            lo, underside = seg.bottom, VolPlane(Plane([x, y, seg.bottom], [0, 0, -1]))
        else:               # a band of even depth under the walking surface
            lo, underside = min(seg.start[2], seg.end[2]) - below, VolPlane(Plane([x, y, z - below], [-v for v in up]))
        hi = max(seg.start[2], seg.end[2]) + above
        frame = Frame([x + ux * seg.run / 2, y + uy * seg.run / 2, (lo + hi) / 2], [ux, uy, 0], [-uy, ux, 0])
        parts.append(nodes.boolean([
            nodes.box(frame, seg.run + 2 * OVERLAP, brief.width, hi - lo, 0),
            VolPlane(Plane([x, y, z + above], up)),
            underside,
        ], "intersection", 0))
    # a round joint where the path turns or ends; the grade holds it level width / 2 each way
    for i in range(len(runs) + 1):
        touching = runs[max(i - 1, 0):i + 1]
        if len(touching) == 2 and _straight(*touching):
            continue
        x, y, z = runs[i].start if i < len(runs) else runs[-1].end
        down = z - min(t.bottom for t in touching) if below is None else below
        frame = Frame([x, y, z + (above - down) / 2], [1, 0, 0], [0, 1, 0])
        parts.append(nodes.cylinder(frame, brief.width / 2, down + above))
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
    r = brief.width / 2
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
