"""The rules, measured. Each one is kept by a step and checked here."""

import numpy as np

__all__ = ["steepest", "cut_and_fill", "path_clear"]


def steepest(segments):
    """The steepest slope on any path: never more than ``ramp_max``, since there are no stairs."""
    return max(abs(seg.rise) / seg.run for runs in segments for seg in runs)


def cut_and_fill(profiles):
    """How far the path goes below the host's surface (cut) and above it (fill), at most, in mm."""
    cut = max(float(np.max(p.ground - p.level)) for p in profiles)
    fill = max(float(np.max(p.level - p.ground)) for p in profiles)
    return {"cut": max(cut, 0.0), "fill": max(fill, 0.0)}


def path_clear(brief, segments, solid, spacing=250.0, inset=50.0, toe=10.0):
    """Walk every path on the finished solid: there must be floor underfoot and nothing overhead.

    Samples the centre line and both edges, ``inset`` in from them; anything ``toe``
    above the walking surface counts as in the way, so a lip trips it too. Returns
    ``{"samples": n, "unsupported": n, "blocked": n}``; a sound design has
    zero of each, whatever the other modules added.
    """
    offsets = (0.0, brief.width / 2 - inset, inset - brief.width / 2)
    counts = {"samples": 0, "unsupported": 0, "blocked": 0}
    for runs in segments:
        for seg in runs:
            ux, uy = seg.direction
            for x, y, z in seg.points(spacing):
                for o in offsets:
                    px, py = x - uy * o, y + ux * o
                    counts["samples"] += 1
                    if solid.get_distance([px, py, z - inset]) >= 0:
                        counts["unsupported"] += 1
                    if any(solid.get_distance([px, py, z + h]) <= 0
                           for h in (toe, brief.headroom / 2, brief.headroom - inset)):
                        counts["blocked"] += 1
    return counts
