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
    (1, 8)
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
        """A path over the stand-in roof that rises and falls.

        A long level stretch on the deck (at 600), a ramp up, stairs up to a terrace,
        a ramp down, then stairs down to 200 mm below the deck, ending in the trench it cuts.
        """
        return cls([Polyline([
            [1000, 2000, 600], [9000, 2000, 600], [14000, 2000, 950], [14000, 5500, 1550], [14000, 9000, 1550],
            [8000, 9000, 1200], [8000, 6000, 400], [3000, 6000, 400],
        ])])
