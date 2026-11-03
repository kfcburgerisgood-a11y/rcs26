"""The drawing: where people walk."""

from dataclasses import dataclass, field

import compas
from compas.geometry import Polyline

__all__ = ["Drawing"]


@dataclass
class Drawing:
    """The drawn inputs, as COMPAS geometry.

    Attributes
    ----------
    paths : list[:class:`compas.geometry.Polyline`]
        Centre lines of the paths, in world coordinates. Each vertex's Z is
        the walking level there; between vertices the level ramps, or steps
        where a ramp would be too steep.

    Examples
    --------
    >>> drawing = Drawing.sample()
    >>> len(drawing.paths), len(drawing.paths[0])
    (1, 5)
    >>> Drawing.from_data(compas.json_loads(compas.json_dumps(drawing.to_data()))).paths[0][1] == drawing.paths[0][1]
    True
    """

    paths: list = field(default_factory=list)

    def to_data(self):
        return {"paths": self.paths}

    @classmethod
    def from_data(cls, data):
        return cls([Polyline(p) for p in data.get("paths", [])])

    def save(self, path):
        compas.json_dump(self.to_data(), path, pretty=True)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_data(compas.json_load(path))

    @classmethod
    def sample(cls):
        """A path over the stand-in roof: level, a ramp, a turn, stairs up to a terrace."""
        return cls([Polyline([
            [1000, 1500, 600], [4500, 1500, 600], [7500, 1500, 800], [7500, 4500, 1400], [10500, 4500, 1400],
        ])])
