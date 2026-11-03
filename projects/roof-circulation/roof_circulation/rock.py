"""A stand-in rocky roof, to try the module on until it meets the real one."""

import numpy as np

from compas_vol.base import VolObject
from compas_vol.processes import fbm

__all__ = ["VolRock"]


class VolRock(VolObject):
    """A slab from ``z = 0`` up to a rocky top: ``thickness`` lifted by fractal noise of ``roughness``.

    The top is defined in world coordinates, so it is the same rock however
    it is sampled. The distance is the vertical offset scaled down by a bound
    on the top's slope: below the true distance, as a search for the surface
    needs, and exact for a flat top.

    Examples
    --------
    >>> flat = VolRock(10000, 6000, 600, roughness=0)
    >>> flat.get_distance([5000, 3000, 700]), flat.get_distance([5000, 3000, 500])
    (100.0, -100.0)
    """

    def __init__(self, length, span, thickness, roughness=0.0, cell=6000.0, seed=0, name=None):
        super().__init__(name=name)
        self.length, self.span, self.thickness = float(length), float(span), float(thickness)
        self.roughness, self.cell, self.seed = float(roughness), float(cell), int(seed)

    @property
    def __data__(self):
        return {"length": self.length, "span": self.span, "thickness": self.thickness,
                "roughness": self.roughness, "cell": self.cell, "seed": self.seed}

    @classmethod
    def __from_data__(cls, data):
        return cls(**data)

    def __repr__(self):
        sizes = (self.length, self.span, self.thickness, self.roughness)
        return "VolRock({:.0f} x {:.0f}, {:.0f} +- {:.0f})".format(*sizes)

    def top(self, x, y):
        """The rock's top at plan points."""
        level = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
        if not self.roughness:
            return level + self.thickness
        return self.thickness + self.roughness * fbm(x, y, level, cell_size=self.cell, seed=self.seed)

    def get_distance(self, point):
        p = self._as_point(point)
        # one-element arrays: the noise's hashing warns on wraparound for plain numbers, not for arrays
        return float(self.get_distance_numpy(np.array([p.x]), np.array([p.y]), np.array([p.z]))[0])

    def get_distance_numpy(self, x, y, z):
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        slope = 3.0 * self.roughness / self.cell     # a bound on how steep the top gets
        top = (z - self.top(x, y)) / np.sqrt(1.0 + slope * slope)
        sides = np.maximum(np.abs(x - self.length / 2) - self.length / 2, np.abs(y - self.span / 2) - self.span / 2)
        return np.maximum(np.maximum(top, sides), -z)
