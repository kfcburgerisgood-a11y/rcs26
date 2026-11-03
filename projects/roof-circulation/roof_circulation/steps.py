"""The workflow: one function per step, each one component on a canvas.

A step takes the brief and what earlier steps made, and returns its own result.
Build steps from ``compas_vol.nodes`` where a component exists, and keep Rhino
out of this package: that is what lets a step become a component later.

The module is :func:`circulation`: drawn paths in, a :class:`Contribution` out,
for any host. :func:`host` is only a stand-in roof to try it on.
"""

import math
import os
from dataclasses import dataclass

import compas
from compas.geometry import Frame, Plane, Polyline

from compas_vol import nodes
from compas_vol.meshing import write_stl
from compas_vol.primitives import VolPlane

from .contribution import Contribution, combine
from .drawing import Drawing

__all__ = [
    "Segment", "host", "segments", "undulate", "walk", "clearance", "support", "circulation", "combine",
    "extent", "export",
]


@dataclass
class Segment:
    """One straight run of a path, between two drawn vertices: level, a ramp, or stairs.

    A run that is not level keeps a level landing where the path turns or ends,
    and a flight keeps one at both ends, so a turn never lands on a slope or a step.
    """

    start: tuple            # (x, y, walking level)
    end: tuple
    risers: int = 0         # 0 for a ramp
    flat_start: float = 0.0     # level plan length at the start
    flat_end: float = 0.0       # and at the end

    @property
    def run(self):
        """float: plan length."""
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def rise(self):
        return self.end[2] - self.start[2]

    @property
    def kind(self):
        return "stairs" if self.risers else "ramp"

    @property
    def sloped(self):
        """float: plan length between the landings."""
        return self.run - self.flat_start - self.flat_end

    @property
    def going(self):
        """float: plan depth of each step, for stairs; the sloped length, for a ramp."""
        return self.sloped / self.risers if self.risers else self.sloped

    def low_end(self):
        """The lower end, the plan unit vector from it to the other end, and the landings at the low and high ends."""
        if self.rise >= 0:
            a, b, low, high = self.start, self.end, self.flat_start, self.flat_end
        else:
            a, b, low, high = self.end, self.start, self.flat_end, self.flat_start
        return a, ((b[0] - a[0]) / self.run, (b[1] - a[1]) / self.run), low, high

    def level(self, u):
        """float: the walking level ``u`` mm in plan from the low end."""
        (_, _, z), _, low, _ = self.low_end()
        if self.risers:
            k = min(max(int((u - low) // self.going), -1), self.risers - 1)
            return float(z + (k + 1) * abs(self.rise) / self.risers)
        return float(z + abs(self.rise) * min(max((u - low) / self.sloped, 0.0), 1.0))

    def points(self, spacing):
        """``(x, y, level)`` along the centre line: every landing's and step's middle, or every ``spacing`` mm.

        Examples
        --------
        >>> Segment((0, 0, 0), (1000, 0, 300), 2, 100, 100).points(100)
        [(50.0, 0.0, 0.0), (300.0, 0.0, 150.0), (700.0, 0.0, 300.0), (950.0, 0.0, 300.0)]
        """
        (x, y, _), (ux, uy), low, high = self.low_end()
        if self.risers:
            us = [low / 2, self.run - high / 2] + [low + (k + 0.5) * self.going for k in range(self.risers)]
        else:
            count = max(2, int(self.run // spacing))
            us = [self.run * (i + 0.5) / count for i in range(count)]
        return [(x + ux * u, y + uy * u, self.level(u)) for u in sorted(us)]


def host(brief):
    """A stand-in roof slab, to try the module on. *Canvas: Volumetric Box.*"""
    centre = Frame([brief.length / 2, brief.span / 2, brief.thickness / 2], [1, 0, 0], [0, 1, 0])
    return nodes.box(centre, brief.length, brief.span, brief.thickness, 0)


def _joints(path):
    """For each vertex, whether the path turns or ends there: where a run that is not level needs a landing."""
    directions = []
    for a, b in zip(path[:-1], path[1:]):
        run = math.hypot(b[0] - a[0], b[1] - a[1])
        if run == 0:
            raise ValueError("A path rises straight up at {}: give it some plan length.".format(list(a)))
        directions.append(((b[0] - a[0]) / run, (b[1] - a[1]) / run))
    turns = [d[0] * e[0] + d[1] * e[1] < 1 - 1e-9 for d, e in zip(directions[:-1], directions[1:])]
    return [True] + turns + [True]


def segments(brief, drawing):
    """Each path as its runs: a ramp up to ``ramp_max``, stairs of at most ``riser`` beyond.

    Reads ``ramp_max``, ``riser``, ``width``. A run that is not level keeps a level
    landing ``width / 2`` long where the path turns or ends, and a flight keeps one
    at both ends. Returns one list of :class:`Segment` per path.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> [s.kind for s in segments(Brief(), Drawing.sample())[0]]
    ['ramp', 'ramp', 'stairs', 'ramp', 'ramp', 'stairs', 'ramp']
    """
    r = brief.width / 2
    result = []
    for path in drawing.paths:
        joints = _joints(path)
        runs = []
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            seg = Segment(tuple(a), tuple(b))
            if seg.rise:
                seg.flat_start, seg.flat_end = (r if joints[i] else 0.0), (r if joints[i + 1] else 0.0)
                if abs(seg.rise) > brief.ramp_max * seg.sloped:
                    seg.risers = int(math.ceil(abs(seg.rise) / brief.riser))
                    seg.flat_start = seg.flat_end = r
                if seg.sloped <= 0:
                    raise ValueError("The run from {} to {} is too short for its landings.".format(list(a), list(b)))
            runs.append(seg)
        result.append(runs)
    return result


def undulate(brief, drawing):
    """A wave along every level run and ramp, ``wave`` high and about ``wavelength`` long. *Canvas: Divide Curve.*

    Reads ``wave``, ``wavelength``, ``wave_step``, ``ramp_max``, ``width``. Stairs,
    landings and every drawn level stay as drawn: each run holds a whole number of
    waves between its landings. Where the full ``wave`` would make a run steeper than
    ``ramp_max``, that run's wave is lowered until it is not, so the wave never makes
    stairs. Returns a new :class:`Drawing`; ``wave = 0`` returns the drawing itself.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> level = Drawing([Polyline([[0, 0, 600], [8000, 0, 600]])])
    >>> wavy = undulate(Brief(wave=50.0, wavelength=6800.0), level).paths[0]
    >>> wavy[0][2], wavy[-1][2], round(max(p[2] for p in wavy))
    (600.0, 600.0, 650)
    """
    if not brief.wave:
        return drawing
    r = brief.width / 2
    paths = []
    for path, runs in zip(drawing.paths, segments(brief, drawing)):
        joints = _joints(path)
        points = [list(path[0])]
        for i, seg in enumerate(runs):
            u0, u1 = (r if joints[i] else 0.0), seg.run - (r if joints[i + 1] else 0.0)
            length = u1 - u0
            if seg.risers or length <= 0:
                points.append(list(seg.end))
                continue
            waves = max(1, int(round(length / brief.wavelength)))
            room = max(brief.ramp_max - abs(seg.rise) / length, 0.0)
            height = min(brief.wave, room * length / (2 * math.pi * waves))
            count = max(2, int(math.ceil(length / brief.wave_step)))
            (sx, sy, sz), (ex, ey, _) = seg.start, seg.end
            ux, uy = (ex - sx) / seg.run, (ey - sy) / seg.run
            for j in range(0 if u0 > 0 else 1, count + 1):
                t = j / count
                z = sz + seg.rise * t + height * math.sin(2 * math.pi * waves * t)
                points.append([sx + ux * (u0 + length * t), sy + uy * (u0 + length * t), z])
            if u1 < seg.run:
                points.append(list(seg.end))
        paths.append(Polyline(points))
    return Drawing(paths)


def _band(brief, runs, below, above):
    """The solid from ``below`` under the walking surface to ``above`` over it, ``width`` wide."""
    parts = []
    for seg in runs:
        (x, y, z), (ux, uy), low, high = seg.low_end()
        across = [-uy, ux, 0]
        top = z + abs(seg.rise)
        # level pieces, as (middle, plan length, level): the landings, and each step
        flats = [(low / 2, low, z), (seg.run - high / 2, high, top)]
        flats += [(low + (k + 0.5) * seg.going, seg.going, z + (k + 1) * abs(seg.rise) / seg.risers)
                  for k in range(seg.risers)]
        for u, length, level in flats:
            if length > 0:
                frame = Frame([x + ux * u, y + uy * u, level + (above - below) / 2], [ux, uy, 0], across)
                parts.append(nodes.box(frame, length, brief.width, below + above, 0))
        if not seg.risers:
            # the slope between the landings: a plan box, cut above and below by planes parallel to it
            x0, y0 = x + ux * low, y + uy * low
            lo, hi = z - below, top + above
            frame = Frame([x0 + ux * seg.sloped / 2, y0 + uy * seg.sloped / 2, (lo + hi) / 2], [ux, uy, 0], across)
            slope = abs(seg.rise) / seg.sloped
            up = [-ux * slope, -uy * slope, 1]
            down = [-v for v in up]
            parts.append(nodes.boolean([
                nodes.box(frame, seg.sloped, brief.width, hi - lo, 0),
                VolPlane(Plane([x0, y0, z + above], up)),
                VolPlane(Plane([x0, y0, z - below], down)),
            ], "intersection", 0))
    # a round joint where the path turns or ends: every run is level within width / 2 of one
    path = [runs[0].start] + [seg.end for seg in runs]
    for (x, y, z), joint in zip(path, _joints(path)):
        if joint:
            frame = Frame([x, y, z + (above - below) / 2], [1, 0, 0], [0, 1, 0])
            parts.append(nodes.cylinder(frame, brief.width / 2, below + above))
    return nodes.boolean(parts, "union", 0)


def walk(brief, segments):
    """The path as solid, down to ``depth`` below its lowest level. *Canvas: Volumetric Boolean.*

    Reads ``width``, ``depth``. Raised parts stand on solid, so no path floats.
    """
    solids = []
    for runs in segments:
        levels = [z for seg in runs for z in (seg.start[2], seg.end[2])]
        floor = min(levels) - brief.depth
        band = _band(brief, runs, max(levels) - floor, 0.0)
        solids.append(nodes.boolean([band, VolPlane(Plane([0, 0, floor], [0, 0, -1]))], "intersection", 0))
    return nodes.boolean(solids, "union", 0)


def clearance(brief, segments, walk):
    """The space kept empty over the path, ``headroom`` high, less the path itself.

    Reads ``width``, ``headroom``.
    """
    space = nodes.boolean([_band(brief, runs, 0.0, brief.headroom) for runs in segments], "union", 0)
    return nodes.boolean([space, walk], "difference", 0)


def support(brief, segments):
    """The top ``support`` mm under the walking surface, which no later process may remove."""
    return nodes.boolean([_band(brief, runs, brief.support, 0.0) for runs in segments], "union", 0)


def circulation(brief, drawing):
    """The module: drawn paths in, a :class:`Contribution` out, for any host.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> brief = Brief()
    >>> part = circulation(brief, Drawing.sample())
    >>> round(part.add.get_distance([14000, 7000, 1650]), 3)   # 100 mm over the terrace, at 1550
    100.0
    >>> part.cut.get_distance([14000, 7000, 2550]) < 0        # a metre above it
    True
    """
    runs = segments(brief, drawing)
    solid = walk(brief, runs)
    return Contribution(add=solid, cut=clearance(brief, runs, solid), keep=support(brief, runs))


def extent(brief, drawing):
    """The box to sample: the host and every path, ``margin`` around. ``(xmin, ymin, zmin, xmax, ymax, zmax)``."""
    points = [p for path in drawing.paths for p in path]
    r = brief.width / 2
    lo = [min([0.0] + [p[0] - r for p in points]), min([0.0] + [p[1] - r for p in points]),
          min([0.0] + [p[2] - brief.depth for p in points])]
    hi = [max([brief.length] + [p[0] + r for p in points]), max([brief.span] + [p[1] + r for p in points]),
          max([brief.thickness] + [p[2] for p in points])]
    return tuple(v - brief.margin for v in lo) + tuple(v + brief.margin for v in hi)


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
