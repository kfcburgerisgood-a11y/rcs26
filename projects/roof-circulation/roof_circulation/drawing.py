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
    stops : list[:class:`compas.geometry.Polyline`]
        Routes to find rather than draw: each from its first point, through
        the others in order, to its last. ``route`` turns each into a path.
    entrances : list[:class:`compas.geometry.Point`]
        Where people come onto the roof. ``network`` grows branching paths
        from them over the gentle ground.
    attractors, repellers : list[:class:`compas.geometry.Point`]
        Places a route bends towards (a view, shade, a planter) or keeps away
        from (a drain, an opening's edge), within the brief's ``influence``.

    Examples
    --------
    >>> drawing = Drawing.sample()
    >>> len(drawing.paths), len(drawing.paths[0]), len(drawing.pins)
    (1, 6, 1)
    >>> again = Drawing.from_data(compas.json_loads(compas.json_dumps(Drawing.sample_stops().to_data())))
    >>> len(again.stops[0]), again.pins[0] == Drawing.sample_stops().pins[0]
    (3, True)
    """

    paths: list = field(default_factory=list)
    pins: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    entrances: list = field(default_factory=list)
    attractors: list = field(default_factory=list)
    repellers: list = field(default_factory=list)

    def to_data(self):
        return {"paths": self.paths, "pins": self.pins, "stops": self.stops, "entrances": self.entrances,
                "attractors": self.attractors, "repellers": self.repellers}

    @classmethod
    def from_data(cls, data):
        return cls(
            [Polyline(p) for p in data.get("paths", [])],
            [Point(*p) for p in data.get("pins", [])],
            [Polyline(p) for p in data.get("stops", [])],
            [Point(*p) for p in data.get("entrances", [])],
            [Point(*p) for p in data.get("attractors", [])],
            [Point(*p) for p in data.get("repellers", [])],
        )

    def save(self, path):
        compas.json_dump(self.to_data(), path, pretty=True)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_data(compas.json_load(path))

    @classmethod
    def sample(cls):
        """A walk drawn over the stand-in rock: along it, round a corner, up to a lookout held at 1300, and back."""
        return cls(
            [Polyline([[1500, 2000, 0], [9000, 2000, 0], [14500, 3500, 0], [14500, 9500, 0], [7000, 9500, 0],
                       [3000, 6500, 0]])],
            [Point(14500, 9500, 1300)],
        )

    @classmethod
    def sample_stops(cls):
        """The same walk to be found instead: an entrance, the lookout, an exit."""
        return cls(
            pins=[Point(14500, 9500, 1300)],
            stops=[Polyline([[1500, 2000, 0], [14500, 9500, 0], [3000, 6500, 0]])],
        )

    @classmethod
    def sample_crossing(cls):
        """A way across the stand-in rock, from one corner to the other."""
        return cls(stops=[Polyline([[1500, 1500, 0], [16500, 10500, 0]])])

    @classmethod
    def sample_lookout(cls):
        """The way across, by a lookout held at 850 on its way: a climb for its rests."""
        return cls(pins=[Point(9000, 7000, 850)],
                   stops=[Polyline([[1500, 1500, 0], [9000, 7000, 0], [16500, 10500, 0]])])

    @classmethod
    def sample_entrance(cls):
        """One way onto the stand-in rock, for a network to grow from."""
        return cls(entrances=[Point(1500, 2000, 0)])
