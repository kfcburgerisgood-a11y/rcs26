"""The brief: every number the workflow depends on, in one place."""

import dataclasses
import json
from dataclasses import dataclass

__all__ = ["Brief", "RAMP_LIMIT"]

RAMP_LIMIT = 1 / 12     # the steepest a ramp may ever be: no brief may ask for more


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
    width: float = 1200.0           # clear walking width; also the level landing at every turn and end
    ramp_max: float = RAMP_LIMIT    # steepest the path may be: 1:12, or gentler
    smoothing: float = 3000.0       # the path follows the rock averaged over this length
    spacing: float = 250.0          # the path is followed every this many mm
    depth: float = 150.0            # the path's solid reaches this far into the rock below it
    headroom: float = 2100.0        # kept empty above the walking surface
    support: float = 300.0          # below the walking surface, no later process may remove

    # the wave along the path, fading out at its ends; lowered wherever it would pass ramp_max
    wave: float = 0.0               # height of the wave: 0 for none
    wavelength: float = 6000.0      # crest to crest

    # reading the roof
    cell: float = 250.0             # plan grid for the slope map

    # where to look for the host's surface, and how to sample it
    search_top: float = 6000.0      # above any roof
    search_bottom: float = -2000.0  # below any roof
    margin: float = 200.0           # room around the model to sample
    voxel: float = 50.0             # sample spacing: 50 to look, less to make

    def __post_init__(self):
        if not 0 < self.ramp_max <= RAMP_LIMIT + 1e-12:
            message = "ramp_max is {:.4g}; a ramp may be at most 1:12 ({:.4f})"
            raise ValueError(message.format(self.ramp_max, RAMP_LIMIT))

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
