"""The brief: every number the workflow depends on, in one place."""

import dataclasses
import json
from dataclasses import dataclass

__all__ = ["Brief"]


@dataclass
class Brief:
    """Every number the workflow depends on, with defaults, in millimetres.

    The path numbers are the module; the host numbers only make a stand-in
    roof to try it on, until it is put on the real one.

    Examples
    --------
    >>> brief = Brief(width=1500.0)
    >>> Brief.from_dict(brief.to_dict()) == brief
    True
    """

    # the host: a stand-in roof slab, top at z = thickness, the size of the grid roof
    length: float = 18000.0
    span: float = 12000.0
    thickness: float = 600.0

    # the path
    width: float = 1200.0           # clear walking width
    depth: float = 150.0            # solid below the lowest walking level of a path
    ramp_max: float = 1 / 12        # steepest a ramp may be; steeper becomes stairs
    riser: float = 150.0            # highest a step may be
    going_min: float = 280.0        # shallowest a step may be
    headroom: float = 2100.0        # kept empty above the walking surface
    support: float = 300.0          # below the walking surface, no later process may remove

    # the wave along level runs and ramps; lowered wherever it would pass ramp_max,
    # so at most about ramp_max x wavelength / 6.3 on a level run: 80 mm for a 6 m wave at 1:12
    wave: float = 0.0               # height of the wave: 0 for none
    wavelength: float = 4000.0      # crest to crest, roughly: each run holds whole waves
    wave_step: float = 250.0        # how finely the wave is followed

    # sampling
    margin: float = 200.0           # room around the model to sample
    voxel: float = 50.0             # sample spacing: 50 to look, less to make

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
