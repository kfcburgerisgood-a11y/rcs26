"""The brief: every number the workflow depends on, in one place."""

import dataclasses
import json
from dataclasses import dataclass

__all__ = ["Brief", "RestBrief", "RAMP_LIMIT", "LANDING"]

RAMP_LIMIT = 1 / 12     # the steepest a ramp may ever be: no brief may ask for more
WIDTH_LEAST = 1200.0    # the narrowest a path may be
LANDING = 1500.0        # by code, a level platform 1500 x 1500 at every end and every turn


@dataclass
class Brief:
    """Every number the workflow depends on, with defaults, in millimetres.

    The path numbers are the module; the host numbers only make a stand-in
    rocky roof to try it on, until it is put on the real one.

    Examples
    --------
    >>> brief = Brief(width=1500.0)
    >>> Brief.from_dict(brief.to_dict()) == brief
    True
    >>> Brief(ramp_max=1 / 8)
    Traceback (most recent call last):
    ...
    ValueError: ramp_max is 0.125; a ramp may be at most 1:12 (0.0833)
    """

    # the host: a stand-in rocky slab, the size of the grid roof, its top at thickness +- roughness
    length: float = 18000.0
    span: float = 12000.0
    thickness: float = 600.0
    roughness: float = 400.0        # how far the rock's top rises and falls; 0 for a flat roof
    rock_cell: float = 6000.0       # size of the rock's largest humps
    rock_seed: int = 3

    # the path: a ramp everywhere, never stairs
    width: float = 1200.0           # clear walking width, at least 1200; 1500 where there is no stair beside it
    ramp_max: float = RAMP_LIMIT    # steepest the path may be: 1:12, or gentler
    smoothing: float = 3000.0       # the path follows the rock averaged over this length
    lift: float = 0.0               # how far the walking surface rides above what it follows; pins stay as set
    spacing: float = 250.0          # the path is followed every this many mm
    depth: float = 150.0            # the path's solid reaches this far into the rock below it
    headroom: float = 2100.0        # kept empty above the walking surface
    support: float = 300.0          # below the walking surface, no later process may remove

    # the wave along the path, fading out at its ends; lowered wherever it would pass ramp_max
    wave: float = 0.0               # height of the wave: 0 for none
    wavelength: float = 6000.0      # crest to crest

    # reading the roof, and finding routes over it
    cell: float = 250.0             # plan grid for the slope map and the route search
    avoid_slope: float = 4.0        # how strongly a route avoids climbing: 0 goes straight
    turns: int = 0                  # how many times a route swings side to side between stops: 0 for none
    swing: float = 0.5              # how far each swing goes aside, over the spacing between swings
    bend: float = 2400.0            # length a route's bends are smoothed over: longer sweeps, shorter hugs the ground
    attract: float = 0.7            # how much cheaper a step is at an attractor, 0 to under 1: a route detours
                                    # to one or it does not, so this acts more as a switch than a dial
    repel: float = 9.0              # how much dearer a step is at a repeller: 1 + this, times
    influence: float = 3000.0       # how far an attractor or repeller reaches, fading to nothing
    route_tolerance: float = 300.0  # where a winding line will not fit, how far a straightened one may stray
    branches: int = 3               # how many ends a network grows to from its entrances
    edge_clearance: float = 500.0   # how far a path's edge keeps from the roof's edge and holes, all along it

    # where to look for the host's surface, and how to sample it
    search_top: float = 6000.0      # above any roof
    search_bottom: float = -2000.0  # below any roof
    margin: float = 200.0           # room around the model to sample
    voxel: float = 50.0             # sample spacing: 50 to look, less to make

    def __post_init__(self):
        if not 0 < self.ramp_max <= RAMP_LIMIT + 1e-12:
            message = "ramp_max is {:.4g}; a ramp may be at most 1:12 ({:.4f})"
            raise ValueError(message.format(self.ramp_max, RAMP_LIMIT))
        if self.width < WIDTH_LEAST:
            raise ValueError("width is {:.0f}; a path is at least {:.0f} wide".format(self.width, WIDTH_LEAST))
        if not 0 <= self.attract < 1:
            raise ValueError("attract is {}; it is 0 to under 1, a share cheaper".format(self.attract))

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError("Not in the brief: {}".format(", ".join(sorted(unknown))))
        return cls(**data)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class RestBrief:
    """The numbers of the rest landings step, apart from the brief: leave the step out, and there are none.

    By code a ramp rests every 750 of climb on a landing 1500 long.

    Examples
    --------
    >>> landing = RestBrief(rise=300.0)
    >>> RestBrief.from_dict(landing.to_dict()) == landing
    True
    """

    rise: float = 750.0             # the most a path climbs between rests: 0 for none
    length: float = 1500.0          # each rest landing's length along the path
    bay: float = 1500.0             # a bay beside each, this deep, to sit in or turn a wheelchair: 0 for none

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError("Not in the rest brief: {}".format(", ".join(sorted(unknown))))
        return cls(**data)
