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
from compas.geometry import Frame, Plane

from compas_vol import nodes
from compas_vol.meshing import write_stl
from compas_vol.primitives import VolPlane

from .contribution import Contribution, combine

__all__ = [
    "Segment", "host", "segments", "walk", "clearance", "support", "circulation", "combine", "extent", "export",
]


@dataclass
class Segment:
    """One straight run of a path, between two drawn vertices: a ramp, or stairs."""

    start: tuple        # (x, y, walking level)
    end: tuple
    risers: int         # 0 for a ramp
    landing: float = 0.0    # level plan length at each end of a flight, so a turn never lands on a step

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
    def going(self):
        """float: plan depth of each step, for stairs."""
        return (self.run - 2 * self.landing) / self.risers if self.risers else self.run

    def low_end(self):
        """The lower end, and the plan unit vector from it to the other end."""
        a, b = (self.start, self.end) if self.rise >= 0 else (self.end, self.start)
        return a, ((b[0] - a[0]) / self.run, (b[1] - a[1]) / self.run)

    def points(self, spacing):
        """``(x, y, level)`` along the centre line: every step's middle, or every ``spacing`` mm of a ramp.

        Examples
        --------
        >>> Segment((0, 0, 0), (1000, 0, 300), 2, landing=100).points(100)
        [(300.0, 0.0, 150.0), (700.0, 0.0, 300.0)]
        """
        (x, y, z), (ux, uy) = self.low_end()
        if self.risers:
            step = abs(self.rise) / self.risers
            us = [self.landing + (k + 0.5) * self.going for k in range(self.risers)]
            return [(x + ux * u, y + uy * u, z + (k + 1) * step) for k, u in enumerate(us)]
        count = max(2, int(self.run // spacing))
        return [(x + ux * u, y + uy * u, z + abs(self.rise) * u / self.run)
                for u in (self.run * (i + 0.5) / count for i in range(count))]


def host(brief):
    """A stand-in roof slab, to try the module on. *Canvas: Volumetric Box.*"""
    centre = Frame([brief.length / 2, brief.span / 2, brief.thickness / 2], [1, 0, 0], [0, 1, 0])
    return nodes.box(centre, brief.length, brief.span, brief.thickness, 0)


def segments(brief, drawing):
    """Each path as its runs: a ramp up to ``ramp_max``, stairs of at most ``riser`` beyond.

    Reads ``ramp_max``, ``riser``, ``width``: a flight keeps a landing
    ``width / 2`` long at each end. Returns one list of :class:`Segment` per path.

    Examples
    --------
    >>> from roof_circulation import Brief, Drawing
    >>> [s.kind for s in segments(Brief(), Drawing.sample())[0]]
    ['ramp', 'ramp', 'stairs', 'ramp']
    """
    result = []
    for path in drawing.paths:
        runs = []
        for a, b in zip(path[:-1], path[1:]):
            seg = Segment(tuple(a), tuple(b), 0)
            if seg.run == 0:
                raise ValueError("A path rises straight up at {}: give it some plan length.".format(list(a)))
            if abs(seg.rise) > brief.ramp_max * seg.run:
                seg.risers = int(math.ceil(abs(seg.rise) / brief.riser))
                seg.landing = brief.width / 2
            runs.append(seg)
        result.append(runs)
    return result


def _band(brief, runs, below, above):
    """The solid from ``below`` under the walking surface to ``above`` over it, ``width`` wide."""
    parts = []
    for seg in runs:
        (x, y, z), (ux, uy) = seg.low_end()
        across = [-uy, ux, 0]
        if seg.risers:
            # one box per step, from the low end up
            step = abs(seg.rise) / seg.risers
            for k in range(seg.risers):
                u, top = seg.landing + (k + 0.5) * seg.going, z + (k + 1) * step
                frame = Frame([x + ux * u, y + uy * u, top + (above - below) / 2], [ux, uy, 0], across)
                parts.append(nodes.box(frame, seg.going, brief.width, below + above, 0))
        else:
            # a plan box, cut above and below by planes parallel to the ramp
            lo, hi = z - below, z + abs(seg.rise) + above
            frame = Frame([x + ux * seg.run / 2, y + uy * seg.run / 2, (lo + hi) / 2], [ux, uy, 0], across)
            slope = abs(seg.rise) / seg.run
            up = [-ux * slope, -uy * slope, 1]
            down = [-v for v in up]
            parts.append(nodes.boolean([
                nodes.box(frame, seg.run, brief.width, hi - lo, 0),
                VolPlane(Plane([x, y, z + above], up)),
                VolPlane(Plane([x, y, z - below], down)),
            ], "intersection", 0))
    # a round landing at every vertex joins the runs, at any angle
    ends = [runs[0].start] + [seg.end for seg in runs]
    for x, y, z in ends:
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
    >>> round(part.add.get_distance([2000, 1500, 700]), 3)     # 100 mm over the walking surface
    100.0
    >>> part.cut.get_distance([2000, 1500, 1600]) < 0         # a metre above it
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
