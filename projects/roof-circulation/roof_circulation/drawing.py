"""The drawing: where people walk."""

from dataclasses import dataclass, field

import compas
from compas.geometry import Point, Polyline

__all__ = ["Drawing"]


@dataclass
class Drawing:
    """The drawn inputs, as COMPAS geometry.

    Attributes
    ----------
    paths : list[:class:`compas.geometry.Polyline`]
        Centre lines of the paths, in plan: their Z is ignored. The level
        follows the host's surface.
    pins : list[:class:`compas.geometry.Point`]
        Levels held where they are drawn: a lookout, a bridge. Each sits on a
        path, within half its width, and its Z is the level there.

    Examples
    --------
    >>> drawing = Drawing.sample()
    >>> len(drawing.paths), len(drawing.paths[0]), len(drawing.pins)
    (1, 6, 1)
    >>> again = Drawing.from_data(compas.json_loads(compas.json_dumps(drawing.to_data())))
    >>> again.paths[0][1] == drawing.paths[0][1], again.pins[0] == drawing.pins[0]
    (True, True)
    """

    paths: list = field(default_factory=list)
    pins: list = field(default_factory=list)

    def to_data(self):
        return {"paths": self.paths, "pins": self.pins}

    @classmethod
    def from_data(cls, data):
        return cls([Polyline(p) for p in data.get("paths", [])], [Point(*p) for p in data.get("pins", [])])

    def save(self, path):
        compas.json_dump(self.to_data(), path, pretty=True)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_data(compas.json_load(path))

    @classmethod
    def sample(cls):
        """A walk over the stand-in rock: along it, round a corner, up to a lookout held at 1300, and back."""
        return cls(
            [Polyline([[1500, 2000, 0], [9000, 2000, 0], [14500, 3500, 0], [14500, 9500, 0], [7000, 9500, 0],
                       [3000, 6500, 0]])],
            [Point(14500, 9500, 1300)],
        )
